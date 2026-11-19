"""Observe the published forward call without replacing its implementation."""
import time


class ForwardTimer:
    def __init__(self, model, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.active = False
        self.records = []
        self.pending = None
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True),
                        model.register_forward_hook(self.after, with_kwargs=True)]

    def before(self, model, args, kwargs):
        if not self.active:
            return
        tensor = next((v for v in list(args)+list(kwargs.values()) if self.torch.is_tensor(v) and v.ndim), None)
        count = int(tensor.shape[0]) if tensor is not None else None
        if count is None:
            raise ValueError('Cannot establish timed forward batch size')
        if self.device.type == 'cuda':
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record(self.torch.cuda.current_stream(self.device))
            self.pending = (start, end, count)
        else:
            self.pending = (time.perf_counter(), None, count)

    def after(self, model, args, kwargs, output):
        if not self.active:
            return
        start, end, count = self.pending
        if end is not None:
            end.record(self.torch.cuda.current_stream(self.device))
            self.records.append((start, end, count))
        else:
            self.records.append((time.perf_counter()-start, None, count))
        self.pending = None

    def summary(self, skip=0):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        records = self.records[skip:]
        seconds = [a.elapsed_time(b)/1000 if b is not None else a for a,b,n in records]
        count = sum(n for a,b,n in records)
        total = sum(seconds)
        return {'forward_seconds':total,'timed_images':count,'timed_batches':len(records),
                'actual_batch_sizes':[n for a,b,n in records],
                'forward_amortized_ms_per_image':1000*total/count if count else None,
                'forward_images_per_second':count/total if total else None,
                'forward_method':'CUDA events on current stream; synchronized before reading' if self.device.type=='cuda' else 'CPU perf_counter',
                'batch_forward_seconds':seconds}

    def close(self):
        for handle in self.handles:
            handle.remove()


def synchronize(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
