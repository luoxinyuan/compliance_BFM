"""CPU-only independent reconstruction of optimized native-time archive reward."""
import argparse,ast,gzip,hashlib,json,math,time
from pathlib import Path
import numpy as np


def run(root,out):
    source=(Path(__file__).with_name('dual_track_task.py')).read_text()
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_terms')
    namespace=dict(np=np,math=math);exec(compile(ast.Module(body=[node],type_ignores=[]),'archived-native-terms','exec'),namespace)
    terms_fn=namespace['native_terms'];pairs=[];evaluations={}
    paths=sorted(root.glob('*-i0.json.gz'))
    assert paths,'No saved exact paired physical trajectories'
    for before in paths:
        after=before.with_name(before.name.replace('-i0.','-i48.'))
        if not after.exists():continue
        rows=[]
        for path in (before,after):
            x=json.load(gzip.open(path,'rt'));s=x['condition'];physics=x['physics'];controls=x['control']
            total=0.;discounted_terms={};stages={};mismatch=0.;last=np.asarray(controls[0]['pre_qpos'][7:])
            # First q_PD predecessor is the last real source control, never actual q.
            prefix=path.with_name(path.name.replace('.json.gz','-source-prefix.json.gz'))
            if prefix.exists():
                px=json.load(gzip.open(prefix,'rt'))
                pc=px.get('control',px.get('controls',[]))
                if pc:last=np.asarray(pc[-1]['q_PD'])
            base_time=controls[0]['time']
            for i,c in enumerate(controls):
                end=controls[i+1]['time'] if i+1<len(controls) else physics[-1]['force_time']+.005
                pp=[r for r in physics if c['time']-1e-8<=r['force_time']<end-1e-8]
                assert pp
                terms=terms_fn(pp,c['q_PD'],last,s);last=np.asarray(c['q_PD']);dt=len(pp)*.005
                # The first predecessor may be absent in older prefix archives;
                # do not hide that uncertainty as exact continuity validation.
                checked=terms if i or prefix.exists() else {k:v for k,v in terms.items() if k!='continuity'}
                for key,value in checked.items():mismatch=max(mismatch,abs(value-c['reward_terms'][key]))
                saved=-dt*sum(c['reward_terms'].values())-c.get('absorbing_failure_penalty',0.)
                mismatch=max(mismatch,abs(saved-c['reward_integral']))
                gamma=.99**((c['time']-base_time)/.1);total+=gamma*saved
                t=c['time'];stop=s.get('stop_time',s['onset']+s['ramp']);end_move=stop+s.get('deceleration',0.)
                phase=('precontact' if t<s['onset'] else 'startup' if t<min(s['onset']+1.,stop)
                       else 'moving' if t<stop else 'deceleration' if t<end_move
                       else 'hold' if t<s['release'] else 'released')
                stages[phase]=stages.get(phase,0.)+gamma*saved
                for key,value in c['reward_terms'].items():discounted_terms[key]=discounted_terms.get(key,0.)-gamma*dt*value
                if c.get('absorbing_failure_penalty'):discounted_terms['terminal']=discounted_terms.get('terminal',0.)-gamma*c['absorbing_failure_penalty']
            active=[r for r in physics if r['active']]
            force=math.sqrt(np.mean([sum(np.dot(z['force'],z['force']) for z in r['contacts']) for r in active]))
            error=100*math.sqrt(np.mean([np.mean([np.sum((np.array(z['anchor'])-z['point'])**2) for z in r['contacts']]) for r in active]))
            quality=dict(loaded_slip_peak=max(r.get('stage_loaded_slip_mps',0.) for r in physics),
                tilt_peak=max(r['tilt_deg'] for r in physics),q_PD_step_peak=max(c['q_PD_step_max_rad'] for c in controls),
                boundary=x['boundary'])
            row=dict(file=str(path),condition=s,source=s['source'],length_s=physics[-1]['force_time']+.005-base_time,
                optimized_discounted_return=total,stage_return=stages,discounted_terms=discounted_terms,
                error_cm=error,force_RMS_N=force,quality=quality,reconstruction_max_error=mismatch)
            rows.append(row);evaluations[path.name]=row
        a,b=rows
        matched=a['condition']==b['condition'] and abs(a['length_s']-b['length_s'])<1e-7
        def dominates(v,w):
            return (v['error_cm']<w['error_cm'] and v['force_RMS_N']<w['force_RMS_N']
                and v['quality']['boundary'] is None
                and all(v['quality'][k]<=w['quality'][k]+1e-8 for k in ('loaded_slip_peak','tilt_peak','q_PD_step_peak')))
        better=0 if dominates(a,b) else 1 if dominates(b,a) else None
        correct=None if better is None else rows[better]['optimized_discounted_return']>rows[1-better]['optimized_discounted_return']
        pairs.append(dict(parent=before.name.split('-i0')[0],matched=matched,before=a,after=b,
            quality_unambiguous_better=better,reward_orders_better_feedback=correct))
    assert all(v['before']['reconstruction_max_error']<1e-8 and v['after']['reconstruction_max_error']<1e-8 for v in pairs),'Actual optimized reward differs from source reconstruction'
    eligible=[p for p in pairs if p['matched'] and p['quality_unambiguous_better'] is not None]
    # No unambiguous pair => explicitly insufficient, not a reward correctness claim.
    status=('clear_reward_counterexample' if any(p['reward_orders_better_feedback'] is False for p in eligible)
            else 'matched_archive_support' if eligible else 'insufficient_quality_matched_archive')
    result=dict(status=status,pairs=pairs,unambiguous_matched_pairs=len(eligible),
        actual_reward_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        limitations='Finite archive; mixed-quality Pareto pairs do not prove reward ranking. No new force cap/reward or training data.',
        formal_RL_launch_allowed=False)
    out.write_text(json.dumps(result,indent=2,allow_nan=False));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    start=time.monotonic();r=run(a.root,a.output);print(r['status'],r['unambiguous_matched_pairs'],'wall_s',time.monotonic()-start)
