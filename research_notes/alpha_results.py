"""Saved trace analysis only. No simulator/model or parameter fitting."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from contact_gate_summary import summarize
from alpha_memory import ForceMemory


def main(run):
    read=lambda p:json.loads(Path(p).read_text())
    notes=Path(__file__).resolve().parent
    protocol=read(notes/'alpha_student_protocol.json');gate_protocol=read(notes/'contact_gate_protocol.json')
    ledger=read(run/'compute-ledger.json')
    train=next(Path(a['path'])/'outputs' for a in ledger['attempts'] if a['stage']=='train' and a.get('result_status')=='completed')
    phys=next(Path(a['path'])/'outputs' for a in ledger['attempts'] if a['stage']=='physics' and a.get('result_status')=='completed')
    source=run.parent/protocol['teacher_source'];d={};raw={};manifest=[]
    conditions=gate_protocol['conditions']+[protocol['physical']['new_final_test']]
    for cond in conditions:
        for ctl in ['standing','clock','contact','A','B']:
            label=cond['name']+'__'+ctl
            path=(phys if ctl in ('A','B') or cond['name']=='final_new' else source)/(label+'.json')
            ep=read(path);raw[label]=ep;d[label]=summarize(ep,gate_protocol['gate'])
            if ctl in ('A','B'):d[label]['shadow_gate_events']=d[label].pop('gate_events')
            assert ep['boundary'] is None and len(ep['physics'])==round((cond['end']-3)*200)
            manifest.append(dict(label=label,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reused_prior_control=path.parent==source))
    improvements={};timing={};diagnostics={}
    for cond in conditions:
        name=cond['name'];base=d[name+'__standing'];teacher=d[name+'__contact']
        for ctl in ['standing','clock','contact','A','B']:
            label=name+'__'+ctl;v=d[label];cs=raw[label]['control'];release=cond['onset']+cond['duration']
            at=lambda r:r['alpha']>.01
            timing[label]=dict(first_alpha_over_0_01=v['first_alpha_over_0_01_time'],last_alpha_over_0_01=v['last_alpha_over_0_01_time'],
                onset_delay_s=None if v['first_alpha_over_0_01_time'] is None else v['first_alpha_over_0_01_time']-cond['onset'],
                alpha_end=cs[-1]['alpha'],alpha_max=max(r['alpha'] for r in cs),
                precontact_alpha_max=max((r['alpha'] for r in cs if r['time']<cond['onset']-1e-8),default=None),
                postrelease_alpha_max=max((r['alpha'] for r in cs if r['time']>=release-1e-8),default=None))
            if cond['enabled']:
                p=v['phases']['all_contact'];b=base['phases']['all_contact'];t=teacher['phases']['all_contact']
                gain=b['tracking_rms_cm']-t['tracking_rms_cm']
                improvements[label]=dict(error_reduction=1-p['tracking_rms_cm']/b['tracking_rms_cm'],force_reduction=1-p['force_rms_n']/b['force_rms_n'],
                    recovered_teacher_error_gain=(b['tracking_rms_cm']-p['tracking_rms_cm'])/gain if abs(gain)>1e-8 else None)
            if ctl in ('A','B'):
                alpha=np.array([c['alpha'] for c in cs]);shadow=np.array([c['shadow_alpha'] for c in cs]);t=np.array([c['time'] for c in cs])
                qq=np.array([c['q_PD'] for c in cs]);sq=np.array([c['shadow_q_PD'] for c in cs])
                masks={'whole':np.ones(len(cs),bool),'pre':t<cond['onset']-1e-8,'contact':(t>=cond['onset']-1e-8)&(t<release-1e-8) if cond['enabled'] else np.zeros(len(cs),bool),'recovery':t>=release-1e-8}
                diagnostics[label]={k:dict(samples=int(m.sum()),alpha_shadow_rmse=float(np.sqrt(np.mean((alpha[m]-shadow[m])**2))),alpha_shadow_max=float(abs(alpha[m]-shadow[m]).max()),
                    q_PD_shadow_rms_deg=float(np.degrees(np.sqrt(np.mean((qq[m]-sq[m])**2))))) for k,m in masks.items() if m.any()}
    checks={}
    for ctl in ['A','B']:
        gain=improvements['final_new__'+ctl];vv=[v for k,v in d.items() if k.endswith('__'+ctl)]
        checks[ctl]=dict(final_tracking=gain['error_reduction']>=.3,final_force=gain['force_reduction']>=.05,
            no_contact_false_trigger=timing['no_contact__'+ctl]['alpha_max']<=.01,
            no_fall=all(v['phases']['whole']['failed_samples']==0 for v in vv),
            no_support_loss=all(v['phases']['whole']['samples_without_either_foot_load']==0 for v in vv),
            release2s_recovery=all(v['root_speed_release_plus_2s_mps']<=.02 for v in vv if v['condition']['enabled']))
    # Raw deployment feature nearest-neighbor audit, training scaling only; no model fit.
    # Do not claim this metric alone establishes support of the full physical distribution.
    def features(ep):
        memory=ForceMemory(320);x=[]
        for c in ep['control']:
            force=np.asarray(memory.append(c['measured_force_xyz'])).reshape(-1)/5
            x.append(np.concatenate([np.asarray(c['bfm_obs'][k]).reshape(-1) for k in protocol['memory']['keys']]+[force]))
        return np.asarray(x,dtype=np.float32)
    refs=[raw[c+'__contact'] for c in protocol['training_conditions']]
    tx=np.concatenate([features(e) for e in refs]);ty=np.concatenate([[c['alpha'] for c in e['control']] for e in refs])
    center=tx.mean(0);scale=np.maximum(tx.std(0),.1);xt=(tx-center)/scale
    neighbor={}
    for cond in conditions:
        for ctl in ['contact','A','B']:
            label=cond['name']+'__'+ctl;ep=raw[label];x=(features(ep)-center)/scale;distance=[];targetgap=[]
            for start in range(0,len(x),32):
                xx=x[start:start+32];ds=np.maximum((xx*xx).sum(1)[:,None]+(xt*xt).sum(1)[None,:]-2*xx@xt.T,0)/xt.shape[1]
                ids=ds.argmin(1);distance.extend(np.sqrt(ds[np.arange(len(ids)),ids]).tolist())
                targets=np.asarray([c.get('shadow_alpha',c['alpha']) for c in ep['control'][start:start+32]])
                targetgap.extend(abs(targets-ty[ids]).tolist())
            neighbor[label]=dict(raw_sensor_feature_nn_rms_quantiles=np.quantile(distance,[.5,.9,1]).tolist(),nearest_teacher_alpha_gap_quantiles=np.quantile(targetgap,[.5,.9,1]).tolist())
    summary=dict(episodes=d,improvements=improvements,timing=timing,visited_state_shadow_diagnostics=diagnostics,raw_feature_distribution_diagnostic=neighbor,
        qualification_checks=checks,qualified={k:all(v.values()) for k,v in checks.items()},manifest=manifest,
        checks=dict(episodes=30,new_physical_branches=15,reused_prior_branches=15,all_complete=True,shared_initial_state=len({ep['common_state_sha256'] for ep in raw.values()})==1),
        fit=read(train/'paired-fit-result.json'),gradient=read(train/'gradient-contract.json'),compute=ledger)
    assert summary['checks']['shared_initial_state']
    (run/'analysis/alpha-results.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(dict(qualification=checks,improvements=improvements,timing=timing),indent=2))

if __name__=='__main__':main(Path(sys.argv[1]))
