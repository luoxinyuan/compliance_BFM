import csv,gzip,json,pickle,hashlib,itertools,math
from pathlib import Path
import numpy as np
s=Path(__file__).resolve().parent
e=s.parent
def gz(p):
 with gzip.open(p) as f:return pickle.load(f)
def jgz(p):
 with gzip.open(p) as f:return json.load(f)
def write(n,rows):
 with (s/(n+'.csv')).open('w') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def hist(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];files={};pairs=[];checks=[]
def episode(p,version,group,branch,ancestor,old=False):
 if old:
  x=jgz(p);sp=x['condition'];ph=x['physics'];end=sp['end'];start=sp['onset'];hold=start+sp['ramp'];release=sp['release']
  a=e/'ctft_positive50_01/aux03'/(version+'.json.gz');files[str(a.relative_to(e))]=hist(a)
  aux=np.array([v['contact']+sum(v['native'].values()) for v in jgz(a)]);ct=np.round([v['time'] for v in x['control']],6);q0=x['initial']['qpos'];hard=False;slip='PASS' if version=='rolling_continue' else 'FAIL';stopped=None
 else:
  x=gz(p);sp=x['spec'];partner=sp['partner'];start=0;hold=partner['distance_m']/partner['plateau_speed_mps']+partner['accel_s'];release=hold+partner['hold_s'];end=release+4
  ph=x['physics'];cs=x['controls'];ct=np.round([v['time']-x['start'] for v in cs],6);q0=x['initial_qpos']
  aux=np.array([sum(v['reward_components'][k] for k in ('contact_aux','native_aux')) if 'reward_components' in v else np.nan for v in cs])
  met=json.loads(p.with_suffix('.json').read_text());hard=met['hard'];slip=met['slip_result'];stopped=met.get('stopped')
 files[str(p.relative_to(e))]=hist(p)
 t=np.round([v['force_time']-(0 if old else x['start']) for v in ph],6)
 anchor=np.array([v['anchor'] for v in ph]);wrist=np.array([v['wrist'] for v in ph]);root=np.array([v['root'] for v in ph]);forces=np.array([v['force'] for v in ph]);active=np.array([v['active'] for v in ph],bool);err=np.linalg.norm(anchor-wrist,axis=1)
 direction=np.array([1.,0.]) if old else np.array([math.cos(math.radians(partner['direction_deg'])),math.sin(math.radians(partner['direction_deg']))])
 count=min(len(t)//4,len(ct)); n=4*count
 assert np.array_equal(ct[:count],t[:n:4])
 extra=(active[:n]*(err[:n]/.10)**2).reshape(-1,4).mean(axis=1)
 if branch=='no_force':assert not np.any(extra)
 for stage,l,u in [('onset',start,release),('midmove',(hold+start)/2,release),('hold',hold,release),('release',release,end),('move_only',start,hold)]+([('whole',start,end)] if branch=='no_force' else []):
  k=int(np.ceil((l-ct[0])/.02-1e-7));m=int(np.ceil((u-ct[0])/.02-1e-7))
  # For release-only checks, the first eligible control interval must contain
  # four actually disconnected physics ticks. Keep mixed intervals identifiable.
  mixed=0
  if stage=='release':
   while k<min(m,count) and active[4*k:4*k+4].any():
    assert not active[4*k:4*k+4].all(),(group,branch,k)
    mixed+=1;k+=1
   assert not active[4*k:4*min(count,m)].any()
  record=dict(mixed_release_blocks=mixed,parent=group,ancestor=ancestor,branch=branch,version=version,stage=stage,complete=False,hard=hard,slip=slip,stopped=stopped,error_cm=None,force_N=None,root_cm=None,hold_forward_cm=None,old=None,lag=None,long=None,both=None,time_only=None)
  if 0<=k<m<=count and np.all(np.isfinite(aux[k:m])):
   sl=slice(4*k,4*m);g=.98**np.arange(m-k);h=.995**np.arange(m-k);d0=np.asarray(q0)[:2] if k==0 else root[4*k-1,:2]
   hi=int(np.ceil((hold-ct[0])/.005-1e-6));ri=int(np.ceil((release-ct[0])/.005-1e-6))
   record.update(complete=True,error_cm=float(np.sqrt(np.mean(err[sl]**2))*100) if stage!='release' and branch!='no_force' else None,force_N=float(np.sqrt(np.mean(np.sum(forces[sl]**2,axis=1)))),root_cm=float(100*(root[4*m-1,:2]-d0)@direction),hold_forward_cm=float(100*(root[ri-1,:2]-root[hi-1,:2])@direction) if ri<=n else None,old=float(g@aux[k:m]),lag=float(g@(aux[k:m]-extra[k:m])),long=float(h@aux[k:m]),both=float(h@(aux[k:m]-extra[k:m])),time_only=float(aux[k:m].sum()))
  rows.append(record)
 return sp,np.asarray(q0),anchor,t
def compare(group,branch,alist):
 a=alist[0]
 for v in alist[1:]:
  n=min(len(a[2]),len(v[2]));assert a[0]==v[0] and np.array_equal(a[1],v[1]) and np.array_equal(a[2][:n],v[2][:n]) and np.array_equal(a[3][:n],v[3][:n]),(group,branch)
 checks.append(dict(group=group,branch=branch,matched=len(alist)))
# #50 common-state/trajectory identity already audited, real spec handoff differs by metadata only.
old=e/'native-rolling-h100-20261107/runs'
aa=[]
for v in ['rolling_continue','I__speed0.15','C32__speed0.15','motion__speed0.15']:
 aa.append(episode(old/('rolling-02' if v=='rolling_continue' else 'qualification-03')/(v+'.json.gz'),v,'50','continue','50-standing',True))
for a in aa:a[0].pop('handoff_time',None)
compare('50','continue',aa)
lock=json.loads((e/'ctft_d1_68/forward/lock/lock.json').read_text())
for pick in lock['picks']:
 i=pick['source']['case'];idx=f'{i:02}';group='d68-'+idx
 for branch in ['continue','stop','no_force']:
  key=idx+'-'+branch
  aa=[episode(e/d/(key+suf+'.gz'),v,group,branch,pick['source']['ancestor']) for d,suf,v in [('ctft_d2_69/reference','','U'),('ctft_d2_69/d2','-walk','weak'),('ctft_d1_68/forward/d2','-walk','full')]]
  compare(group,branch,aa)
for branch in ['continue','stop','no_force']:
 aa=[episode(e/'ctft_quiet_01'/d/(v+'-'+branch+'.gz'),v,'quiet',branch,'quiet') for d,v in [('paired','B0'),('resumed','U')]]
 compare('quiet',branch,aa)
origins=json.loads((e/'ctft_start_pair_01/origins.json').read_text())
for origin in origins:
 idx=f"{origin['case']:02}"
 if not (e/'ctft_start_pair_01/paired'/f'{idx}-continue-unchanged.gz').exists():continue
 group='start-'+idx
 for branch in ('continue','stop','no_force'):
  aa=[episode(e/'ctft_start_pair_01/paired'/f'{idx}-{branch}-{v}.gz',v,group,branch,origin['ancestor']) for v in ('unchanged','early','late')]
  compare(group,branch,aa)
for key,grp in itertools.groupby(sorted(rows,key=lambda r:(r['parent'],r['branch'],r['stage'])),key=lambda r:(r['parent'],r['branch'],r['stage'])):
 rr=list(grp);ref=next(r for r in rr if r['version']==('rolling_continue' if key[0]=='50' else 'B0' if key[0]=='quiet' else 'unchanged' if key[0].startswith('start-') else 'U'))
 for b in rr:
  if b is ref:continue
  sign=0
  if ref['complete'] and b['complete'] and not (ref['hard'] or b['hard']) and ref['error_cm'] is not None:
   err=b['error_cm']-ref['error_cm'];force=b['force_N']-ref['force_N']
   if err<=0 and force<=0 and (err<=-1 or force<=-.5):sign=1
   if err>=0 and force>=0 and (err>=1 or force>=.5):sign=-1
  d=dict(parent=key[0],ancestor=ref['ancestor'],branch=key[1],stage=key[2],contrast=b['version'],winner=(b['version'] if sign==1 else ref['version'] if sign==-1 else ''),hard=ref['hard'] or b['hard'])
  win,lose=(b,ref) if sign==1 else (ref,b)
  d['qualified']=bool(sign and win['slip']=='PASS' and (key[1]!='stop' or max(0,win['hold_forward_cm'])<=max(0,lose['hold_forward_cm'])))
  for metric in ['old','lag','long','both']:
   d[metric]=None if not sign else float((1+np.sign(sign*(b[metric]-ref[metric])))/2)
  pairs.append(d)
write('scores',rows);write('pairs',pairs);write('checks',checks);(s/'input_hashes.json').write_text(json.dumps(files,indent=1))
