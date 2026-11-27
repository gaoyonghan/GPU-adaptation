"""Measure native MX calls; never synthesize measurements from old metrics."""
from collections import Counter
from contextlib import contextmanager
import time


class NativeTimer:
    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.active = False
        self.warming = False
        self.events = []
        self.load_seconds = 0.0
        self.load_depth = 0
        self.warmup_seconds = None
        self.warmup_batches = 0
        self.pipeline_seconds = 0.0
        self.pipeline_batches = 0
        self.e2e_seconds = 0.0
        self.e2e_batches = 0
        self.e2e_pending = None
        self.pending = None
        self.evaluation_started = None

    def sync(self):
        self.torch.cuda.synchronize(self.device)

    @contextmanager
    def loading(self):
        outer = self.load_depth == 0
        self.load_depth += 1
        if outer:
            self.sync()
            begin = time.perf_counter()
        try:
            yield
        finally:
            self.load_depth -= 1
            if outer:
                self.sync()
                self.load_seconds += time.perf_counter() - begin

    @contextmanager
    def warmup(self):
        self.sync()
        begin = time.perf_counter()
        self.warming = True
        try:
            yield
        finally:
            self.sync()
            self.warmup_seconds = time.perf_counter() - begin
            self.warming = False
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def invoke(self, model, *args, **kwargs):
        if self.warming:
            self.warmup_batches += 1
            return model(*args, **kwargs)
        if not self.active:
            return model(*args, **kwargs)
        values = list(args) + list(kwargs.values())
        tensor = next((v for v in values if self.torch.is_tensor(v) and v.ndim), None)
        if tensor is None:
            raise ValueError('Cannot establish native forward sample count')
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record(self.torch.cuda.current_stream(self.device))
        result = model(*args, **kwargs)
        end.record(self.torch.cuda.current_stream(self.device))
        self.events.append((start, end, int(tensor.shape[0])))
        return result

    def begin_batch(self):
        if self.warming:
            return
        self.sync()
        self.pending = time.perf_counter()
        self.e2e_pending = self.pending
        self.active = True

    def prediction_ready(self):
        if self.e2e_pending is not None:
            self.sync()
            self.e2e_seconds += time.perf_counter() - self.e2e_pending
            self.e2e_batches += 1
            self.e2e_pending = None

    def end_batch(self):
        if self.pending is not None:
            self.sync()
            self.pipeline_seconds += time.perf_counter() - self.pending
            self.pipeline_batches += 1
            self.pending = None

    def summary(self, phase, wall_seconds):
        self.sync()
        seconds = sum(a.elapsed_time(b) / 1000.0 for a, b, n in self.events)
        count = sum(n for a, b, n in self.events)
        sizes = Counter(n for a, b, n in self.events)
        torch = self.torch
        return dict(
            model_load_seconds_this_segment=self.load_seconds,
            model_load_scope='Sum of instrumented native model/processor construction, loading and placement blocks; excludes dataset setup. Different boundary from Example contiguous load block.',
            warmup_seconds_this_segment=self.warmup_seconds,
            warmup_batches=self.warmup_batches,
            warmup_scope='Native repeated preprocessed first batch; synchronized before/after; native count retained. Preprocessing, transfer and publisher softmax-to-CPU are not inside this warmup timer, unlike Example.',
            timed_images=count, forward_seconds=seconds,
            inference_e2e_seconds=self.e2e_seconds if self.e2e_batches else None,
            forward_amortized_ms_per_image=1000 * seconds / count if count else None,
            forward_images_per_second=count / seconds if seconds else None,
            e2e_images_per_second=count / self.e2e_seconds if self.e2e_seconds else None,
            actual_batch_sizes={str(k): v for k, v in sorted(sizes.items())},
            forward_method='MX torch.cuda events on current stream around the unchanged outer model call; synchronize before elapsed_time; seconds.',
            e2e_scope='Synchronized wall time before native DataLoader next() through native prediction/ranking CPU availability; excludes record writes and subsequent metric updates. Worker prefetch can start decode before this interval. SPIDER starts after composite image construction. Native metric operations before prediction readiness remain included and are listed in models.json timing_alignment. Semantically compatible, not identical to Example serial read/decode boundary.',
            native_pipeline_seconds=self.pipeline_seconds,
            native_pipeline_batches=self.pipeline_batches,
            native_pipeline_scope='DataLoader next() wait (generator: after composite construction), transfer, forward, native postprocessing and native per-batch metric updates, through CPU prediction capture; excludes JSONL/status writes. Includes prefetch differences. Not Example inference_e2e_seconds.',
            resume_segments=1,
            instrumentation_predictions_unchanged_this_segment=None,
            peak_allocated_bytes_this_segment=torch.cuda.max_memory_allocated(self.device),
            evaluation_seconds=None,
            run_wall_seconds_this_segment=wall_seconds,
            gpu_at_end=dict(name=torch.cuda.get_device_name(self.device), device=str(self.device), torch_version=torch.__version__),
            phase=phase, performance_comparison_valid=False,
            backend_flags=dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                               cudnn_benchmark=torch.backends.cudnn.benchmark,
                               cudnn_deterministic=torch.backends.cudnn.deterministic),
            timed_e2e_batches=self.e2e_batches,
            measurement_state='measured_native_boundaries; Example_e2e_boundary_differs; public_evaluation_separate_not_measured_here')
