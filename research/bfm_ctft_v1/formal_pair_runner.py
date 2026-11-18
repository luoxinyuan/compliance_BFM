"""Serial single-GPU U/T execution; no automatic retry after a failed arm."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
from resource_guard import check
from cas_checkpoint import durable


def main():
    out=STAGE/'runs/formal-pair-01'
    out.mkdir(exist_ok=False)
    lock=json.loads((STAGE/'locks/common-N-resource-lock-output250-v4.json').read_text())
    settled=json.loads((STAGE/'locks/s0-plus90m-closeout-v1.json').read_text())
    assert lock['N']==131072 and settled['N']==131072
    started=time.time()
    wall_limit=144000-settled['stage_conservative_charge_s']-36000-900
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'research/bfm_ctft_v1').glob('*.py')}
    record=dict(status='RUNNING',started_unix=started,pid=os.getpid(),
        runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        N_per_arm=131072,single_GPU0_serial=True,formal_source_sha256=source,
        max_pair_wall_s=wall_limit,S0_settled_job_s=settled['stage_conservative_charge_s'],arms=[])
    stopped=False
    def handler(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,handler);signal.signal(signal.SIGTERM,handler)
    def save():
        record['elapsed_s']=time.time()-started
        record['stage_job_s_conservative']=settled['stage_conservative_charge_s']+record['elapsed_s']
        durable(out/'ledger.json',(json.dumps(record,indent=2)+'\n').encode())
    for arm in ['U','T']:
        assert not stopped
        check(STAGE)
        assert {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'research/bfm_ctft_v1').glob('*.py')}==source
        directory=STAGE/'runs'/f'formal-{arm}-01'
        assert not directory.exists()
        with (out/f'{arm}.log').open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('formal_train.py')),
                '--arm',arm,'--output',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            item=dict(arm=arm,pid=proc.pid,output=str(directory.relative_to(STAGE)),started_unix=time.time())
            record['arms'].append(item);save()
            terminate_at=None
            while proc.poll() is None:
                if stopped or time.time()-started>wall_limit:
                    if terminate_at is None:
                        proc.send_signal(signal.SIGTERM);terminate_at=time.time()
                        record['stop_requested']=True;save()
                    elif time.time()-terminate_at>900:
                        record['status']='STOP_SAVE_NOT_FINISHED';save()
                        raise RuntimeError('Owned child still saving after900s; manual inspection required; not SIGKILLed')
                time.sleep(5)
                save()
            item['returncode']=proc.returncode;item['ended_unix']=time.time();save()
            progress=directory/'progress.json'
            valid=progress.exists() and json.loads(progress.read_text())['collected']==131072
            if proc.returncode or not valid:
                record['status']='STOPPED_ARM_FAILURE_NO_AUTOMATIC_NEXT_ARM';save();return
    record['status']='U_T_FIXED_ENDPOINTS_COMPLETE_EVALUATION_NOT_STARTED';save()


if __name__=='__main__':main()
