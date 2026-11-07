"""Finite native hot-start shooting and receding-horizon privileged control.
Predictions use CURRENT partner position/velocity only. No PPO samples/training.
"""
import argparse,copy,gzip,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
STAGE=ROOT.parent
BACKEND=STAGE.parent
ASSET_ROOT=Path(os.environ.get('BFM_ROLLING_ASSET_ROOT','/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs'))
PLAN=json.loads(Path(__file__).with_name('rolling_protocol.json').read_text())
PREDICTION=None

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def study(env,model,contract,out):
    global PREDICTION
    import torch,numpy as np,mujoco
    import native_env,geometry_contact
    from native_env import Task,spec,install
    from clean_policy import direct_command,Expert as OldExpert
    from geometry_policy import expand
    from native_policy import Expert as NativeExpert
    from native_policy_coordinate import Expert as CoordinateExpert
    from locomotion_rollout import spherical
    from m9_cyclic import sample
    from rolling_snapshot import BranchState
    from m11_runtime_speed import install as speed
    torch.set_num_threads(1);torch.set_grad_enabled(False);install();write(out/'runtime-speed.json',speed(env))
    from rolling_backend import install as backend_difference
    write(out/'backend-difference.json',backend_difference(env,model,STAGE/'compatibility-prefix'))
    write(out/'software.json',dict(torch=torch.__version__,torch_cuda=torch.version.cuda,mujoco=mujoco.__version__,python=sys.version,device=str(env.standing.device),scope='explicit post-restart runtime difference; shared environment read-only'))
    native_env.RUN=ASSET_ROOT/'native-contact-20261106'
    original_frame=geometry_contact.frame
    def predicted_frame(s,t):
        if PREDICTION is None:return original_frame(s,t)
        c=PREDICTION;dt=max(0.,t-c['time'])
        return c['position']+dt*c['velocity'],c['velocity'].copy(),c['angle'],0.
    geometry_contact.frame=predicted_frame
    task=Task(env,model,contract,5001)
    counts=dict(candidate_physics_steps=0,actual_physics_steps=0,qualification_physics_steps=0,reset_source_physics_steps=0,searches=0,native_inference_samples=0)
    identities={};nets={}
    info=[('native_initial','native-contact-20261106/native-initial.pt','d399174cc2ddbd3efb858953653c9134495070948341381565f5f918f58da5cd',NativeExpert),('C32','native-contact-20261106/ppo-coordinate/last.pt','39cd0db55fddb174de3f77104acbd8d7574fb3a5051f743e4ca39bb6b7d24ad0',CoordinateExpert),('I','m39a-demand-20261104/B-ppo-03/job/best.pt','72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a',OldExpert),('C0','prior-recovery-20261105/correct-C0/C0.pt','ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c',OldExpert)]
    for name,path,digest,cls in info:
        p=ASSET_ROOT/path;assert sha(p)==digest,(name,'identity mismatch')
        ck=torch.load(p,map_location='cuda',weights_only=True);n=cls(1088,'direct').cuda()
        if name in ['I','C0']:expand(n,ck)
        else:n.load_state_dict(ck['state_dict'])
        n.eval().requires_grad_(False);nets[name]=n;identities[name]=dict(path=str(p),sha256=digest,runtime=ck.get('runtime'),iteration=ck.get('iteration'),role='frozen evaluation only' if name in ['I','C0'] else 'native independent reference')
    write(out/'models.json',identities)
    rp=STAGE/'runs/reward-inference-01/reward-candidates.json'
    reward_z={}
    if rp.exists():
        manifest=json.loads(rp.read_text());assert manifest['model_sha256']==contract['model_sha256']
        for row in manifest['candidates']:
            z=torch.tensor(row['z'],device='cuda',dtype=torch.float32);assert abs(float(z.norm())-16)<1e-4
            reward_z[row['name']]=z
    write(out/'reward-source-status.json',dict(status='original_replay_verified_and3configs_inferred' if reward_z else 'no_new_reward_sources',new_reward_z_generated=len(reward_z),manifest_sha256=sha(rp) if rp.exists() else None,no_contact_reward_prompt_claim=True))
    exploration_reward_names=['forward015_ordinary','forward022_ordinary'] if reward_z else []
    selected_reward=None
    selection=STAGE/'runs/reward-selection.json'
    if selection.exists():
        selected_reward=json.loads(selection.read_text())['selected']
        assert selected_reward is None or selected_reward in reward_z
    def raw_net(name):
        n=nets[name];x=task.features()
        return n.raw_mean(x) if hasattr(n,'raw_mean') else n.distribution(x).mean
    def condition(distance=.72,average=.15,stop=None):
        s=spec(distance,distance/average if distance else 4.,5.0,until=4.)
        if stop is not None:s.update(stop_time=stop,deceleration=.7,release=stop+2.7,end=stop+6.7)
        return s
    def native_target(kind,t,origin=5.0,phase=0.):
        if kind in ['native_initial','C32','I','C0']:return direct_command(raw_net(kind),env.standing)
        if kind in reward_z:
            u=float(np.clip((t-origin)/1.5,0,1));return spherical(env.standing,reward_z[kind],u*u*(3-2*u))
        if kind=='standing':return env.standing
        if kind in task.native_z:return task.native_z[kind]
        assert kind=='motion'
        u=float(np.clip((t-origin)/1.5,0,1));u=u*u*(3-2*u)
        return spherical(env.standing,sample(task.refs['straight'],phase+max(0.,t-origin-1.5)/task.refs['straight']['period']),u)
    def metrics(rows,controls):
        active=[r for r in rows if r['active']]
        f=np.array([r['force'] for r in active]);err=np.array([np.array(r['anchor'])-r['wrist'] for r in active])
        root=np.array([r['root'] for r in rows]);vel=np.array([r['qvel'][:3] for r in rows]);feet=np.array([r['feet_xyz'] for r in rows]);q=np.array([r['qpos'][7:] for r in rows]);qd=np.array([r['q_PD'] for r in controls])
        yaw=np.unwrap(np.array([np.arctan2(np.array(r['pelvis_matrix'])[1,0],np.array(r['pelvis_matrix'])[0,0]) for r in rows]))
        ans=dict(physics_steps=len(rows),error_cm=float(100*np.sqrt(np.mean(np.sum(err*err,1))))) if len(active) else dict(physics_steps=len(rows),error_cm=None)
        ans.update(force_rms=float(np.sqrt(np.mean(np.sum(f*f,1)))) if len(active) else 0.,force_xyz_rms=np.sqrt(np.mean(f*f,0)).tolist() if len(active) else [0]*3,signed_error_cm=(100*err.mean(0)).tolist() if len(active) else [0]*3,force_xyz_mean=f.mean(0).tolist() if len(active) else [0]*3,cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=9.999)) if len(active) else 0.,root_delta_cm=(100*(root[-1]-root[0])).tolist(),root_vx_mean=float(vel[:,0].mean()),root_speed_mean=float(np.linalg.norm(vel[:,:2],axis=1).mean()),pelvis_yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])),foot_paths_cm=(100*np.linalg.norm(np.diff(feet,axis=0),axis=2).sum(0)).tolist(),slip_proxy_mean=float(np.mean([r['contact_tangent_speed_max_mps'] for r in rows])),qpd_max_step_deg=float(np.degrees(np.abs(np.diff(qd,axis=0))).max()) if len(qd)>1 else 0.,q_actual_max_step_deg=float(np.degrees(np.abs(np.diff(q,axis=0))).max()) if len(q)>1 else 0.,min_height=float(min(r['height'] for r in rows)),max_tilt=float(max(r['tilt_deg'] for r in rows)),torque_saturation_max=float(max(r['torque_saturation'] for r in rows)),work_J=float(sum(r['external_power']*.005 for r in rows)))
        return ans
    summaries=[]
    def save_episode(label,extra=None):
        task.label=label;task.save(out)
        ep_path=out/(label+'.json.gz')
        with gzip.open(ep_path,'rt') as f:ep=json.load(f)
        ep.update(scope='frozen-source qualification or real native hot-start rolling control; NOT new PPO training',formal_RL_decisions=0)
        with gzip.open(ep_path,'wt') as f:json.dump(ep,f)
        row=dict(label=label,boundary=task.boundary,condition=copy.deepcopy(task.spec),all=metrics(task.physics,task.controls),role='real executed physical trajectory',**(extra or {}))
        s=task.spec
        for key,a,b in [('moving',s['onset'],s.get('stop_time',s['onset']+s['ramp'])),('decelerating',s.get('stop_time',s['onset']+s['ramp']),s.get('stop_time',s['onset']+s['ramp'])+s.get('deceleration',0)),('holding',s.get('stop_time',s['onset']+s['ramp'])+s.get('deceleration',0),s['release']),('released',s['release'],s['end'])]:
            rs=[r for r in task.physics if a<=r['force_time']<b];cs=[r for r in task.controls if a<=r['time']<b]
            if rs:row[key]=metrics(rs,cs)
        summaries.append(row);write(out/'summaries.json',summaries);write(out/'counts.json',counts)
        print('ACTUAL',label,(round(row['all']['error_cm'],3) if row['all']['error_cm'] is not None else 'NA'),round(row['all']['force_rms'],3),row['all']['root_delta_cm'],task.boundary,flush=True)
        return row
    onset_identity=None
    def run_fixed(label,kind,s):
        nonlocal onset_identity
        task.reset(s,record=True,label=label);counts['reset_source_physics_steps']+=600+len(task.prefix_data['physics']);start=time.monotonic();onset_sha=None;exit_z=None
        while not task.done:
            t=float(env.tether.time)
            if t<s['onset']-1e-7:z=env.standing
            else:
                if onset_sha is None:
                    onset_sha=BranchState(task).fingerprint
                    if onset_identity is None:onset_identity=onset_sha
                    assert onset_sha==onset_identity,'speed/controller onset physical state differs'
                z=native_target(kind,t)
                if (kind=='motion' or kind in reward_z) and t>=s['onset']+s['ramp']-1e-7:
                    if exit_z is None:exit_z=task.last_command.clone()
                    u=float(np.clip((t-s['onset']-s['ramp'])/1.5,0,1));z=spherical(exit_z,env.standing,u*u*(3-2*u))
            _,_,_,st=task.step(z/16);counts['qualification_physics_steps']+=st['physics_steps']
        return save_episode(label,dict(controller=kind,wall_s=time.monotonic()-start,model=identities.get(kind),onset_complete_state_sha=onset_sha,source_contract='common standing until onset; motion scripted1.5s return at partner stop is qualification reference ONLY, not feedback Teacher'))
    # A deterministic compact search in native parameter coordinates. No force→skill rule.
    rng=np.random.default_rng(5001)
    def command(c,j,phase):
        kind=c['kind']
        if kind in nets:return direct_command(raw_net(kind),env.standing)
        if kind=='reward':return reward_z[c['name']]
        if kind=='hold':return c['z']
        if kind in ['standing','goal_arm5','goal_arm15']:return env.standing if kind=='standing' else task.native_z[kind]
        u=j/(PLAN['search']['horizon_decisions']-1);nodes=np.array(c['alpha']);a=float(np.interp(u,np.linspace(0,1,len(nodes)),nodes))
        rate=c['rate'];base_name=c.get('base','motion')
        walk=reward_z[base_name] if base_name in reward_z else sample(task.refs['straight'],phase+(j*.1*rate)/task.refs['straight']['period'])
        base=spherical(env.standing,walk,a)
        return spherical(base,task.native_z['goal_arm15'],float(c.get('arm',0.)))
    def propose(phase,previous=None):
        # Throughput-calibrated seven candidates, fixed before rolling results.
        # Shift the previous plan by one actually executed100ms command.
        prior=dict(kind='sequence',base=('forward015_ordinary' if exploration_reward_names else 'motion'),alpha=[.2,.7,1.,1.],rate=1.,arm=0.)
        if previous and previous['kind']=='sequence':
            prior=copy.deepcopy(previous);xx=np.linspace(0,1,len(prior['alpha']))
            prior['alpha']=np.interp(np.minimum(xx+1/(PLAN['search']['horizon_decisions']-1),1),xx,prior['alpha']).tolist()
        if exploration_reward_names:
            # Stable native forward motor sources can be useful proposals even
            # when their contact force is not below Standing. NOT good Teachers.
            cs=[dict(kind='standing'),dict(kind='goal_arm15'),
                *[dict(kind='reward',name=n) for n in exploration_reward_names],
                dict(kind='sequence',alpha=[1.,1.,1.,1.],rate=1.,arm=0.),prior]
        else:
            cs=[dict(kind='standing'),dict(kind='goal_arm15'),dict(kind='C32'),dict(kind='sequence',alpha=[1.,1.,1.,1.],rate=1.,arm=0.),prior]
        while len(cs)<7:
            cs.append(dict(kind='sequence',base=prior.get('base','motion'),alpha=np.clip(np.array(prior['alpha'])+rng.normal(0,.15,4),0,1).tolist(),rate=float(np.clip(prior['rate']+rng.normal(0,.15),.5,1.4)),arm=float(np.clip(prior.get('arm',0.)+rng.normal(0,.08),0,.2))))
        return cs
    search_rows=[]
    def search(label,phase,previous=None,save_predictions=False):
        global PREDICTION
        counts['searches']+=1;assert counts['searches']<=PLAN['search']['max_rollout_plans_total']+len(PLAN['representative_states'])
        state=BranchState(task);n0=len(task.physics);c0=len(task.controls);now=float(env.tether.time)
        p,v,angle,_=original_frame(task.spec,now)
        prediction=dict(time=now,position=p.copy(),velocity=np.array(v).copy(),angle=angle)
        candidates=propose(phase,previous);records=[];best=None;best_score=-float('inf');best_physics=None
        start=time.monotonic()
        for i,c in enumerate(candidates):
            state.restore();PREDICTION=prediction
            # Erase generator-only future fields, including legacy distance-scaled
            # heading-cost metadata. Fixed nominal heading scale for ALL predictions.
            for key in ['stop_time','deceleration','change_time','after_curvature','rotation_onset','rotation_return','rotation_return_ramp']:
                task.spec.pop(key,None)
            task.spec.update(anchor_mode='fixed_prediction_origin',distance=.72,ramp=1.,curvature=0.,rotation_deg=0.,onset=-1000.,release=100000.,end=now+PLAN['search']['horizon_decisions']*.1+1.)
            task.spec['enabled']=bool(env.tether.active) # CURRENT connection, never future release.
            total=0.;sequence=[]
            for j in range(PLAN['search']['horizon_decisions']):
                z=command(c,j,phase);sequence.append(z.detach().clone());_,r,done,st=task.step(z/16);counts['candidate_physics_steps']+=st['physics_steps'];total+=r
                if done:break
            rr=task.physics[n0:];cc=task.controls[c0:]
            if task.boundary:total=-1e6-abs(total) # reject unsafe predicted branches
            row=dict(index=i,kind=c['kind'],score=float(total),boundary=task.boundary,metrics=metrics(rr,cc),steps=len(sequence),parameters={k:v for k,v in c.items() if k!='z'})
            records.append(row)
            if total>best_score:
                best_score=total;best=(copy.deepcopy(c),sequence);best_physics=copy.deepcopy(rr) if save_predictions else None
        PREDICTION=None;state.restore()
        chosen,sequence=best
        # Predicted first command must produce same real physical first100ms;
        # prediction-vs-actual later differences are recorded, not hidden.
        result=dict(label=label,time=now,phase=phase,current_partner_position=env.tether.anchor.tolist(),current_partner_velocity=np.array(v).tolist(),state_fingerprint=state.fingerprint,candidates=records,selected=max(range(len(records)),key=lambda i:records[i]['score']),wall_s=time.monotonic()-start,prediction='constant current velocity, no future actual path/stop/release',model_future_access=False,prediction_reward_distance_metadata=.72,generator_future_fields_erased=True,privileged_input=task.features()[0].cpu().tolist(),restricted_ideal_force_input=task.student_features()[0].cpu().tolist(),previous_executed_z=task.last_command[0].cpu().tolist(),selected_first_z=sequence[0][0].cpu().tolist(),label_scope='search query; actual executed first command separately recorded in real physical controls')
        search_rows.append(result);write(out/'searches.json',search_rows);write(out/'counts.json',counts)
        if save_predictions:
            with gzip.open(out/(label+'-selected-PREDICTION.json.gz'),'wt') as f:json.dump(dict(scope='prediction only, NOT real execution',physics=best_physics,result=result),f)
        return sequence[0],chosen,result
    if ARGS.mode=='compatibility':
        task.reset(condition(),record=True,label='version_difference_smoke');xs=[]
        for j in range(40):
            xs.append(task.features().clone());task.step(native_target('C32',float(env.tether.time))/16)
            assert not task.boundary,task.boundary
        torch.save(torch.cat(xs).cpu(),out/'actual-observations.pt')
        save_episode('version_difference_smoke',dict(scope='targeted CUDA/MuJoCo version difference check, not new task success'))
    elif ARGS.mode=='stop_references':
        # Frozen same-condition stop baselines, no new search or learned weights.
        for kind in ['C32','I']:run_fixed(kind+'__stop7.8',kind,condition(stop=7.8))
    elif ARGS.mode=='free':
        for name in ['forward015_ordinary','forward015_weighted']:
            s=condition();s['enabled']=False
            run_fixed(name+'__NO_FORCE',name,s)
    elif ARGS.mode=='qualification':
        # Only 3 speed points × core native references; not a cap/seed grid.
        for v in PLAN['speed']['averages']:
            for kind in ['native_initial','C32','motion']:run_fixed(f'{kind}__speed{v:.2f}',kind,condition(average=v))
        run_fixed('standing__speed0.15','standing',condition())
        run_fixed('I__speed0.15','I',condition())
        run_fixed('goal_arm5__local','goal_arm5',condition(.07,.09))
        for name in reward_z:run_fixed(name+'__speed0.15',name,condition())
        standing=next(r for r in summaries if r['label']=='standing__speed0.15')
        valid=[r for r in summaries if r['label'].split('__')[0] in reward_z and not r['boundary'] and r['moving']['error_cm']<standing['moving']['error_cm'] and r['moving']['force_rms']<standing['moving']['force_rms']]
        ranked=sorted(valid,key=lambda r:r['all']['force_rms']+r['all']['error_cm']/10.)
        write(STAGE/'runs/reward-selection.json',dict(selected=ranked[0]['label'].split('__')[0] if ranked else None,rule='no safety boundary and better moving error AND force than standing, then rank same-condition force+error_cm/10; exploration proposal only, not a qualified contact Expert claim',ranked=[r['label'] for r in ranked],qualification_runtime=json.loads((out/'launch.json').read_text())['runtime']))
    elif ARGS.mode in ['diagnostic','rolling']:
        # Full MjData/cache/history branching verification on actual contact state.
        s=condition();task.reset(s,record=True,label='restore_test')
        while env.tether.time<6.3-1e-7:task.step(native_target('C32',float(env.tether.time))/16)
        ss=BranchState(task);outputs=[]
        for _ in range(2):
            ss.restore()
            for k in range(3):task.step(env.standing/16)
            outputs.append(dict(state=BranchState(task).fingerprint,physics=copy.deepcopy(task.physics[-60:]),control=copy.deepcopy(task.controls[-15:])))
        assert outputs[0]==outputs[1],'full physical restored replay is not exact'
        ss.restore();counts['branch_validation_physics_steps']=1380;write(out/'restoration.json',dict(status='passed',same_full_source_state=ss.fingerprint,repeat_real_100ms_commands=3,all_physics_control_outputs_exact=True,solver_warmstart_preserved=True,not_qpos_only=True))
        if ARGS.mode=='diagnostic':
            for case in PLAN['representative_states']:
                task.reset(condition(case['distance'],case['average']),record=True,label=case['name'])
                while env.tether.time<case['time']-1e-7:task.step(native_target(case['prefix'],float(env.tether.time))/16)
                phase=max(0.,case['time']-6.5)/task.refs['straight']['period'] if case['prefix']=='motion' else 0.
                z,c,result=search(case['name'],phase,save_predictions=True)
                # One real first action (not an open-loop optimized whole horizon).
                _,_,_,st=task.step(z/16);counts['actual_physics_steps']+=st['physics_steps']
                write(out/(case['name']+'-first-actual.json'),dict(physics=task.physics[-st['physics_steps']:],search_label=case['name'],scope='real first action, no claim of full rolling validation'))
                print('SEARCH',case['name'],result['selected'],round(result['candidates'][result['selected']]['score'],4),round(result['wall_s'],2),flush=True)
        else:
            run_specs=[('rolling_continue',condition()),('rolling_stop',condition(stop=7.8))]
            initial_search_rng=copy.deepcopy(rng.bit_generator.state)
            for label,s in run_specs:
                rng.bit_generator.state=copy.deepcopy(initial_search_rng)
                task.reset(s,record=True,label=label);counts['reset_source_physics_steps']+=600+len(task.prefix_data['physics']);phase=0.;previous=None
                while not task.done:
                    now=float(env.tether.time)
                    z,previous,res=search(label,phase,previous)
                    phase+=(.1*previous.get('rate',0.))/task.refs['straight']['period'] if previous['kind']=='sequence' else 0.
                    if round(now*10)%10==0:print('ROLLING',label,round(now,2),res['selected'],res['candidates'][res['selected']]['kind'],round(res['wall_s'],2),flush=True)
                    _,_,_,st=task.step(z/16);counts['actual_physics_steps']+=st['physics_steps']
                    if round(env.tether.time*10)%10==0:
                        with gzip.open(out/(label+'-partial.json.gz'),'wt') as f:json.dump(dict(initial=task.initial,physics=task.physics,control=task.controls,condition=task.spec,anchor=task.anchor.tolist(),scope='in-progress actual execution, not predicted trajectory'),f)
                save_episode(label,dict(controller='native hot-start receding-horizon search; no learned online contact controller'))
    write(out/'counts.json',counts)
    return dict(status='completed',mode=ARGS.mode,episodes=len(summaries),counts=counts,formal_RL_decisions=0,student_optimizer_updates=0,bfm_updates=0)

def main():
    global ARGS
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['qualification','diagnostic','rolling','compatibility','free','stop_references'],required=True);p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args()
    assert not ARGS.output.exists();ARGS.output.mkdir()
    import subprocess
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    write(ARGS.output/'launch.json',dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('rolling_protocol.json')),mode=ARGS.mode,actual_backend='previously validated H100-class NVIDIA environment',new_PPO_or_Student=False))
    os.environ['BFM_RECOVERY_OUTPUT']=str(ARGS.output);os.chdir(ARGS.output)
    import m10_physics,m10_stage
    m10_physics.OUT=ARGS.output;m10_physics.MODEL=Path(os.environ.get('BFM_ROLLING_MODEL_VIEW',str(ASSET_ROOT/'issue19-recovery-20261028T045520Z/model-view')));m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
