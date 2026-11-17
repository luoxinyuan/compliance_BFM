"""Fail-closed project-local prelaunch guard; never manages another process."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

EXPECTED_GPU='GPU-3ae548cd-2131-2b5d-42ad-5d0f40ff9da0'


def check(stage,allowed_owned_pids=(),evidence_kind='live'):
    assert evidence_kind in ('live','fixture')
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    fields=next(csv.reader(gpu.splitlines()))
    uuid=fields[0].strip();total,used,util=(int(v.strip()) for v in fields[1:])
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    allowed={os.getpid(),*allowed_owned_pids}
    external={}
    for line in csv.reader(apps.splitlines()):
        if len(line)>=3 and line[0].strip()==EXPECTED_GPU:
            pid=int(line[1].strip())
            if pid not in allowed:external[pid]=line[2].strip()
    free=shutil.disk_usage(stage).free
    result=dict(evidence_kind=evidence_kind,checked_unix=time.time(),gpu_uuid=uuid,total_mib=total,used_mib=used,utilization_percent=util,
        external_gpu_processes=external,allowed_owned_pids=sorted(allowed),shared_disk_free_bytes=free,
        contract='GPU0 exclusive project job +100GiB shared disk floor; no other processes touched')
    result['pass']=uuid==EXPECTED_GPU and not external and free>=100*2**30 and total-used>=max(6*1024,.08*total)
    trace=Path(stage)/'prep/resource-preflight-traces';trace.mkdir(parents=True,exist_ok=True)
    (trace/f'{time.time_ns()}-{os.getpid()}.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['pass']:
        raise RuntimeError('RESOURCE_PREFLIGHT_REFUSED: '+json.dumps(result,sort_keys=True))
    return result
