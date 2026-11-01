import json,gzip,hashlib,subprocess
from pathlib import Path
import numpy as np
from m9b_stage import RUN,OLD
from m9_cyclic import CyclicFeedback
from m9b_feedback import CadenceFeedback,GuideFilteredFeedback

def main():
 refs=json.loads((OLD/'native-cycles.json').read_text())['cycles'];ledger=json.loads((RUN/'compute-ledger.json').read_text());result={};pairs={}
 for a in ledger['attempts']:
  for p in (Path(a['path'])/'outputs').glob('*.json.gz'):
   with gzip.open(p,'rt') as f:e=json.load(f)
   if not e.get('control'):continue
   cs=e['control'];rs=e['physics'];ctl=cs[0]['controller'];fb=(CadenceFeedback if ctl=='cadence' else GuideFilteredFeedback if ctl=='filtered' else CyclicFeedback)({k:r['period'] for k,r in refs.items()},ctl!='feedback');err=[];memory=[];zs=np.array([c['executed_z'][0] for c in cs]);forceparity=[]
   for i,c in enumerate(cs):
    s=fb.step(c['measured_force_xyz'],np.array(c['sensor']['world_R_body']),guide=np.array(c['expert_query_guide_world']));ref=c['oracle_controller_state'];err.append([abs(s[k]-ref[k]) for k in ['alpha','phase','turn']])
    if i:memory.append(float(abs(np.array(c['own_previous_z'][0])-zs[i-1]).max()))
   for r in rs:forceparity.append(float(abs(np.array(r['force'])-np.array(r['applied_contact_wrench'][:3])).max()))
   maximum=np.max(err,axis=0).tolist();mem=max(memory,default=0.);fp=max(forceparity);assert max(maximum)<1e-8 and mem==0 and fp<1e-10
   assert abs(np.linalg.norm(zs,axis=1)-16).max()<1e-4
   assert max(np.linalg.norm(r['force']) for r in rs)<5.00001
   result[a['stage']+'/'+e['label']]=dict(query_replay_max=maximum,previous_executed_command_parity=mem,force_applied_parity=fp,complete=e['boundary'] is None,controls=len(cs),physics=len(rs))
   if a['stage']=='final':
    key=e['label'].split('__')[0];v=np.r_[e['initial']['qpos'],e['initial']['qvel'],e['anchor']]
    if key in pairs:assert np.max(abs(v-pairs[key]))<1e-10
    else:pairs[key]=v
 assert all(not a.get('remaining_processes') for a in ledger['attempts'] if a['status']=='finished')
 out=dict(episodes=result,paired_final_conditions=len(pairs),remaining_owned=[a for a in ledger['attempts'] if a['status']=='running']);(RUN/'audit.json').write_text(json.dumps(out,indent=2));print('AUDIT',len(result),'episodes',len(pairs),'final conditions')
if __name__=='__main__':main()
