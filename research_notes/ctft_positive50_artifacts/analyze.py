from low2 import names,path,s,m,load,np,json
import sys,csv
sys.path.insert(0,str(s.parent/'ctft_quiet_01'))
from score_endpoint import stance_segments,material
from reward import ForceEMA,contact_cost
def dump(name,rows):
 with (s/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
st=[];phases=[];windows=[];matched=[];coords={}
for n in names:
 x=load(path(n));ph=x['physics'];c=x['condition'];initial=x['initial'];control=x['control']
 for r,cs in zip(ph,load(s/'low01'/(n+'.json.gz')),strict=True):r['ctft_material_contacts']=cs
 get=lambda k:np.array([r[k] for r in ph])
 t=get('force_time').round(5);forces=get('force');lag=get('anchor')-get('wrist');root=get('root');wrist=get('wrist');shoulder=get('shoulder_xyz')
 yaw=np.unwrap(np.arctan2(get('pelvis_matrix')[:,1,0],get('pelvis_matrix')[:,0,0]))*180/np.pi
 seg=material(m,ph,initial['qpos'],stance_segments(get('foot_normal_force_n'),sum(m.body_mass)*np.linalg.norm(m.opt.gravity)))
 for row in seg:
  row.update(model=n,t0=t[row['start']],t1=round(t[row['end_exclusive']-1]+.005,5),passed=bool(row['measured'] and row['p95']<=.05 and row['path']<=.02))
 st.extend(seg)
 auxfile=s/'aux03'/(n+'.json.gz')
 aux=load(auxfile) if auxfile.exists() else None
 ema=ForceEMA();cost=[]
 for j,cc in enumerate(control):
  f=forces[4*j+3];r=contact_cost(filtered_force_n=ema.update(f),applied_force_n=f,wrist_velocity_mps=cc['post_velocity'],root_velocity_mps=cc['post_root_velocity'],guidance_mode=1)
  cost.append(r['contact_native_reward'])
 if aux:assert np.allclose(cost,[a['contact'] for a in aux],atol=1e-12,rtol=0)
 tc=np.round([r['time'] for r in control],5);hold=c.get('stop_time',c['onset']+c['ramp'])+c.get('deceleration',0);release=c['release']
 def metrics(label,a,b):
  idx=np.flatnonzero((t>=a)&(t<b));j,k=idx[0],idx[-1];ids=np.flatnonzero((tc>=a)&(tc<b))
  def dx(v,key):
   return v[k,0]-(v[j-1,0] if j else initial[key][0])
  inside=[r for r in seg if r['t0']>=a and r['t1']<=b]
  disc=.98**np.arange(len(ids));native=sum(d*sum(aux[i]['native'].values()) for d,i in zip(disc,ids)) if aux else None
  contact=disc@np.array(cost)[ids]
  return dict(model=n,stage=label,t0=a,t1=b,root_cm=100*dx(root,'root'),wrist_cm=100*dx(wrist,'wrist'),arm_rel_shoulder_cm=100*(dx(wrist,'wrist')-dx(shoulder,'shoulder_xyz')),yaw_change_deg=yaw[k]-yaw[max(j-1,0)],err_cm=float(100*np.sqrt(np.mean(np.sum(lag[idx]**2,axis=1)))) if a>=5 and b<=release else None,force_N=float(np.sqrt(np.mean(np.sum(forces[idx]**2,axis=1)))),stance_n=len(inside),pass_n=sum(r['passed'] for r in inside),unmeasured=sum(not r['measured'] for r in inside),contact_return=contact,native_return=native,aux_return=contact+native if aux else None)
 for label,a,b in [('pre',4,5),('move',5,hold),('hold',hold,release),('release',release,c['end'])]:
  phases.append(metrics(label,a,b))
 if n!='rolling_stop':
  for a in [5,6,7,8,hold,hold+1]:
   windows.append(metrics('move' if a<hold else 'hold',a,a+1))
 coords[n]=[dict(t=float(t[i]),f=float(np.linalg.norm(forces[i])),stance=' / '.join(('?' if not r['measured'] else 'OK' if r['passed'] else 'FAIL') for r in seg if r['t0']<=t[i]<r['t1']) or '-',active=bool(ph[i]['active'])) for i in range(0,len(ph),8)]
for a in windows[:6]:
 for b in windows[6:]:
  if a['t0']!=b['t0']:continue
  def cls(w):
   return '?' if not w['stance_n'] or w['unmeasured'] else 'clean' if w['stance_n']==w['pass_n'] else 'fail'
  eligible=(a['stage']!='move' or min(a['root_cm'],a['wrist_cm'])>=2) and all(abs(a[k]-b[k])<=max(2,.2*abs(a[k])) for k in ['root_cm','wrist_cm'])
  matched.append(dict(other=b['model'],t0=a['t0'],matched=eligible,MPC_quality=cls(a),other_quality=cls(b),MPC_aux=a['aux_return'],other_aux=b['aux_return']))
for label,data in [('stances',st),('phases',phases),('windows',windows),('matched',matched)]:dump(label,data)
(s/'video_channels.json').write_text(json.dumps(coords))
