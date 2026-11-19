import csv,gzip,itertools,json,pickle,sys
from pathlib import Path
import numpy as np
from reward_v2 import step_reward
s=Path(__file__).resolve().parent
prev=s.parent/'short_direction_01'
sys.path.insert(0,str(prev))
from near_paths import names,path
def read(p):
    with gzip.open(p) as f:return json.load(f)
def save(name,rows):
    with (s/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def calc(n,group,ph,a,start,hold,release):
    arr=np.array([[r['force_time'] for r in ph[i:i+4]] for i in range(0,len(ph),4)]).round(6)
    out=[];steps=[]
    assert len(arr)==len(a)
    for i,(native,f) in enumerate(a):
        block=ph[4*i:4*i+4]
        new,burden,pos=step_reward(native,1,f,block)
        bare,_,_=step_reward(native,1,f,block,position=False)
        assert abs(bare-new-pos)<1e-10
        assert step_reward(native,0,f,block)[0]==native
        if not any(r['active'] for r in block):assert pos==0
        steps.append([bare,new])
    for label,lo,hi in [('onset',start,release),('move',start,hold),('hold',hold,release)]:
        keep=(arr[:,0]>=round(lo,6))&(arr[:,-1]<round(hi,6));ww=.98**np.arange(keep.sum())
        vals=np.array(steps)[keep]
        for gamma,(bare,new) in [(.98,ww@vals),(1.,vals.sum(axis=0))]:
            out.append(dict(group=group,model=n,stage=label,gamma=gamma,bare=bare,new=new))
    return out
if __name__=='__main__':
    rows=[]
    for n in names:
        x=read(path(n));a=read(prev/'aux01'/(n+'.json.gz'));p=x['physics'];c=x['condition']
        inputs=[(sum(r['native'].values()),tick['force']) for r,tick in zip(a,p[3::4],strict=True)]
        for r,(_,f) in zip(a,inputs):assert abs(.0025*np.dot(f,f)-r['parts']['burden'])<1e-10
        rows+=calc(n,'train_short08',p,inputs,c['onset'],c['onset']+c['ramp'],c['release'])
    for p in sorted((s.parent/'ctft_start_pair_01/paired').glob('*-stop-*.gz')):
        with gzip.open(p) as f:x=pickle.load(f)
        ph=x['physics'];t=x['start'];c=x['spec']['partner']
        hold=t+c['distance_m']/c['plateau_speed_mps']+c['accel_s']
        inputs=[(r['reward_components']['native_aux'],tick['force']) for r,tick in zip(x['controls'],ph[3::4],strict=True)]
        rows+=calc(x['model'],x['spec']['parent'],ph,inputs,t,hold,hold+c['hold_s'])
    save('scores.csv',rows)
    phys={}
    for f in ['short.csv','stop.csv']:
        for z in csv.DictReader((prev/f).open()):phys[z['group'],z['controller'],z['window']]=z
    pairs=[]
    for a,b in itertools.permutations(rows,2):
        if (a['group'],a['stage'],a['gamma'])!=(b['group'],b['stage'],b['gamma']):continue
        A,B=[phys[z['group'],z['model'],z['stage']] for z in [a,b]]
        de=float(B['error_cm'])-float(A['error_cm']);df=float(B['force_n'])-float(A['force_n'])
        if min(de,df)<0 or de<1 and df<.5:continue
        pairs.append(dict(group=a['group'],stage=a['stage'],gamma=a['gamma'],better=a['model'],other=b['model'],bare=int(a['bare']>b['bare']),new=int(a['new']>b['new'])))
    save('pairs.csv',pairs)
    for group in ['near','stop']:
        for stage in ['onset','move','hold']:
            rr=[z for z in pairs if (z['group']=='train_short08')==(group=='near') and z['stage']==stage and z['gamma']==.98]
            print(group,stage,len(rr),sum(z['bare'] for z in rr),sum(z['new'] for z in rr))
