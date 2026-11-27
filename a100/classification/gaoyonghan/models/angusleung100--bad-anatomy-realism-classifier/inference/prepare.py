#!/usr/bin/env python3
"""Model-local conversion invoked automatically by inference.py; no model loads."""
import argparse
import json
from pathlib import Path
import sys
MODEL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODEL/'inference'))
from common import save
from output_format.prepare_classification import finalize_run
CONFIG=json.loads(Path(__file__).with_name('entrypoint.json').read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--phase',choices=['full','smoke'],default='full')
    a=p.parse_args();root=(a.run_dir if a.run_dir.is_absolute() else MODEL/a.run_dir).resolve()
    if not root.is_relative_to(MODEL/'result'):p.error('Run directory must stay inside this model result/')
    result=dict(state='failed',task=CONFIG['task'],sequence=CONFIG['sequence'],phase=a.phase,bundles=[])
    try:
        report=finalize_run(root,CONFIG['sequence'],a.phase,successful=True)
        if report['state']!='exported':raise RuntimeError(json.dumps(report,ensure_ascii=False))
        bundles=[Path(x['output']).resolve() for x in report['items']]
        if not bundles or any(not b.is_relative_to(root) for b in bundles):raise RuntimeError('Invalid standardized bundle paths')
        result.update(state='completed',scoring_ready=True,bundles=[str(b.relative_to(root)) for b in bundles],
          native_status=str((root/a.phase/('m%03d'%CONFIG['sequence'])/'status.json').relative_to(root)))
        save(root/'inference_output.json',result)
    except BaseException as exc:
        save(root/'inference_output.json',dict(result,reason=str(exc)));raise

if __name__=='__main__':main()

