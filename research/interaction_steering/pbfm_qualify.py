"""Pre-registered native ERI free-motion qualification. No training/search."""
import json,sys,time,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from .pbfm_native import Native

def main():
    s=Path(sys.argv[1]);out=s/'runs/free-01';out.mkdir(parents=True,exist_ok=False);n=Native(s);rows=[];t=time.time()
    for speed in [0.,.09,.18]:
        for rep in [0,1]:
            name=f'v{speed:.2f}-r{rep}';row=n.trial(speed,rep,out/name);rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    checks=[]
    for speed in [0.,.09,.18]:
        a=torch.load(out/f'v{speed:.2f}-r0/trace.pt',weights_only=False);b=torch.load(out/f'v{speed:.2f}-r1/trace.pt',weights_only=False)
        exact=len(a['physics'])==len(b['physics']) and all(np.array_equal(x['qpos'],y['qpos']) and np.array_equal(x['qvel'],y['qvel']) for x,y in zip(a['physics'],b['physics']))
        exact=exact and all(np.array_equal(x['raw'],y['raw']) and all(np.array_equal(v,y['obs'][k]) for k,v in x['obs'].items()) for x,y in zip(a['controls'],b['controls']))
        checks.append({'speed':speed,'repeat_exact':exact})
    core=[r for r in rows if r['repeat']==0];per=[]
    for r in core:
        final=r['phases'].get('final',{});per.append({'speed':r['speed'],'pass':r['boundary'] is None and r['actual_ticks']==2400 and final.get('horizontal_speed_rms',1e9)<=.05 and abs(final.get('yaw_end_deg',1e9))<=20 and r['max_slip'][0]<=.45 and (abs(r['root_dx'])<=.1 if r['speed']==0 else r['phases'].get('moving',{}).get('mean_vx',-1)>0)})
    order=core[1]['phases'].get('moving',{}).get('mean_vx',-1)<core[2]['phases'].get('moving',{}).get('mean_vx',-1)
    passed=all(x['pass'] for x in per) and order and all(x['repeat_exact'] for x in checks)
    result={'gate':passed,'per_condition':per,'speed_order':order,'repeats':checks,'seconds':time.time()-t,'total_reserved_ticks':n.budget.n,'actual_ticks':sum(r['actual_ticks'] for r in rows),'runtime':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'provider':n.session.get_providers(),'physics_backend':'MuJoCo'+__import__('mujoco').__version__,'contact_runs':0};(out/'gate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
