"""Preserve native forward measurements and expose the common warmup option."""
import os

def warmup_steps(default):
    steps=int(os.environ.get('MODEL_WARMUP_STEPS',default))
    if steps<0:raise ValueError('Warmup iterations cannot be negative')
    return steps

def native_timing(native_metrics,batches,run_wall_seconds):
    timing=dict(native_metrics.get('timing') or {})
    timing.update(record_export_batches=batches,run_wall_seconds=run_wall_seconds,
      timing_note='Native forward timing is preserved; wall time includes record writes.',
      requested_warmup_steps=int(os.environ.get('MODEL_WARMUP_STEPS','0')))
    return timing

