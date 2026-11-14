"""Versioned v1.1 physical gate semantics; no reward or controller changes."""
import math
import numpy as np

def metrics(rows):
    if not rows:return None
    yaw=np.unwrap([math.atan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0]) for p in rows])
    root=np.asarray([p['root'][:2] for p in rows]);dt=np.diff([p['time'] for p in rows])
    assert (dt>0).all(), 'Noncausal physical records'
    dy=np.degrees(np.diff(yaw))
    return dict(yaw_excursion_deg=float(np.max(np.abs(np.degrees(yaw-yaw[0])))),
        yaw_total_variation_deg=float(np.abs(dy).sum()),yaw_net_deg=float(np.degrees(yaw[-1]-yaw[0])),
        yaw_range_deg=float(np.degrees(np.ptp(yaw))),yaw_rate_p95_deg_s=float(np.quantile(np.abs(dy/dt),.95)) if len(dt) else 0.,
        root_path_m=float(np.linalg.norm(np.diff(root,axis=0),axis=1).sum()),root_net_m=float(np.linalg.norm(root[-1]-root[0])),
        slip_peak=max(p['stage_loaded_slip_mps'] for p in rows),slip_p95=float(np.quantile([p['stage_loaded_slip_mps'] for p in rows],.95)),
        unsupported_ticks=sum(not any(p['support']) for p in rows),max_joint_excess_rad=max(p['joint_limit_excess_max_rad'] for p in rows),
        start_s=rows[0]['time'],end_s=rows[-1]['time'],physics_ticks=len(rows))

def actual_release_index(rows):
    seen=False
    for i,p in enumerate(rows):
        if p['active']:seen=True
        elif seen:
            assert all(float(x)==0. for x in p['force']), 'Inactive but nonzero force'
            return i
    return None

def physical_failures(m,domain,reference=None):
    if m is None:return ['UNMEASURED']
    errors=[]
    if m['max_joint_excess_rad']>.02:errors.append('G04_real_joint')
    if m['unsupported_ticks']>0:errors.append('G05_support')
    if m['slip_peak']>(.10 if domain=='none_zero' else .45):errors.append('G06_slip' if domain=='none_zero' else 'G12_slip')
    if m['yaw_excursion_deg']>15.:errors.append('G07_yaw' if domain=='none_zero' else 'G10_yaw')
    if domain=='none_zero' and m['root_path_m']>.10:errors.append('G08_root')
    if domain=='recovery' and reference is not None and m['root_path_m']>reference['root_path_m']+.10:errors.append('G11_root')
    return errors

class OnlineMonitor:
    """Checks each actual physics row. Missing recovery reference never means PASS."""
    def __init__(self,family):self.family=family;self.rows=[];self.release=None;self.seen=False;self.first_failure=None
    def observe(self,row):
        if self.first_failure:return self.first_failure
        self.rows.append(row)
        if row['active']:self.seen=True
        elif self.seen and self.release is None:self.release=len(self.rows)-1
        errors=[]
        if row['joint_limit_excess_max_rad']>.02:errors.append('G04_real_joint')
        if self.family in ('none','zero'):errors+=physical_failures(metrics(self.rows),'none_zero')
        elif self.release is not None:errors+=physical_failures(metrics(self.rows[self.release:]),'recovery')
        if errors:self.first_failure=dict(gates=sorted(set(errors)),time=row['time'])
        return self.first_failure

def angle_deg(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)),-1,1)))

def q_gate(transitions,families,released,parents,correct,nontrivial,support=True):
    if transitions<3500 or not {'none','near','mid','zero','stop','continue'}<=set(families) or not released:return 'PENDING_COVERAGE'
    if parents<4 or nontrivial==0:return 'UNDERPOWERED'
    return 'PASS' if support and correct/nontrivial>=.70 else 'FAIL'

def probe_gate(parents,positive,negative,increment,behavior,ablation):
    if parents<2 or positive==0 or negative==0:return 'UNDERPOWERED'
    return 'SUPPORTED' if increment>=.05 and behavior and ablation else 'NOT_SUPPORTED'

def fixtures():
    def rows(degrees):
        return [dict(time=i*.005,pelvis_matrix=[[math.cos(math.radians(v)),-math.sin(math.radians(v)),0],[math.sin(math.radians(v)),math.cos(math.radians(v)),0],[0,0,1]],root=[0,0,1],support=[True,True],joint_limit_excess_max_rad=0.,stage_loaded_slip_mps=0.,active=False,force=[0,0,0]) for i,v in enumerate(degrees)]
    small=rows([0,3,-3]*20);m=metrics(small);assert m['yaw_total_variation_deg']>15 and not physical_failures(m,'none_zero')
    assert 'G07_yaw' in physical_failures(metrics(rows([0,10,20,0])),'none_zero')
    assert metrics(rows(list(range(0,1082,10))))['yaw_excursion_deg']>1000
    assert actual_release_index(small) is None
    rr=rows([0,0,0]);rr[0]['active']=True;rr[0]['force']=[1,0,0];assert actual_release_index(rr)==1
    path=rows([0,0,0]);path[1]['root'][0]=.06;assert 'G08_root' in physical_failures(metrics(path),'none_zero')
    for field,value,gate in [('joint_limit_excess_max_rad',.020001,'G04_real_joint'),('stage_loaded_slip_mps',.100001,'G06_slip'),('support',[False,False],'G05_support')]:
        row=rows([0,0]);row[-1][field]=value;assert gate in physical_failures(metrics(row),'none_zero')
    mm=metrics(rows([0,0]));mm['root_path_m']=.12
    assert not physical_failures(mm,'recovery',dict(root_path_m=.02))
    mm['root_path_m']=.120001;assert 'G11_root' in physical_failures(mm,'recovery',dict(root_path_m=.02))
    for threshold in (5.,10.):
        a=[1.,0.];theta=math.radians(threshold+1);assert angle_deg(a,[math.cos(theta),math.sin(theta)])>threshold
    assert q_gate(3500,['none','near','mid','zero','stop','continue'],True,4,7,10)=='PASS'
    assert q_gate(3500,['none','near','mid','zero','stop','continue'],True,4,6,10)=='FAIL'
    assert q_gate(3500,['none','near','mid','zero','stop','continue'],True,2,7,10)=='UNDERPOWERED'
    assert probe_gate(4,0,10,.1,True,True)=='UNDERPOWERED'
    assert probe_gate(4,10,10,.06,True,True)=='SUPPORTED'
    assert probe_gate(4,10,10,.06,False,True)=='NOT_SUPPORTED'
    return dict(status='PASS',grade='synthetic_logic_only',physical_rollouts=0,training=0,
        checked=['yaw excursion vs TV vs wrapped','root path vs net','actual release vs none','real q','support','slip','same-state root comparator','U 6deg/11deg rejection','Q group denominator','probe missing classes'])

if __name__=='__main__':
    import json
    print(json.dumps(fixtures(),indent=2))
