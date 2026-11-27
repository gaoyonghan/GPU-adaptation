"""Run explicitly repaired models once, after the existing validation queue exits."""
from pathlib import Path
import fcntl,subprocess,json,time
r=Path('/data/zhoujianjie/delivery_validation_20261126')
with (r/'validation.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 # The queued manager is done; release before starting its own locked invocation.
requests=json.loads((r/'repaired_retry_requests.json').read_text())
command=['/usr/bin/python',str(r/'run_delivery_validation.py'),'--root',str(r)]
for mid in requests:command+=['--only',mid]
print('REPAIRED_RETRY',requests,flush=True)
raise SystemExit(subprocess.call(command,cwd=r))
