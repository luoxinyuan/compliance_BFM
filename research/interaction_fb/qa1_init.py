"""Archive fixed prelock, register bounded cost and validate footprint before GPU."""
import hashlib,json,os,subprocess,sys,shutil,time
from pathlib import Path
import torch
torch.set_num_threads(1)
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'experiments/r2-execution-20261114/runtime';S=P/'experiments/r2-g17-qa1-20261115';O=P/'experiments/r2-v11-qentry-20261114'
sys.path[:0]=[str(R),str(R/'research_notes')]
from research.interaction_fb.journal import read,dump
from research.interaction_fb.approved_journal import load_state
lock=json.loads((S/'prep/fixed-schedule-01/lock.json').read_text());ss=read(S/'prep/fixed-schedule-01/schedule.ptz',lock['schedule'])
counts=ss['historical_counts'].clone();groups={}
for mm in ss['metadata']:groups.setdefault(mm['stratum'],[]).append(mm['index'])
for k in range(456):
 for slot in range(64):
  i=int(ss['C'][k,slot]);u=int(ss['U'][k,slot]);assert ss['metadata'][i]['stratum']==ss['metadata'][u]['stratum']
  group=groups[ss['metadata'][i]['stratum']];assert counts[i]==min(counts[j] for j in group);counts[i]+=1
src=load_state(Path(lock['source']));meta=dump(S/'prep/footprint-agent56.ptz',src['agent'])
# Eight complete agent packets plus growing per-update audit, traces/numerical tables and conditional physical assets.
projected=8*meta['bytes']*1.3+200*2**20
assert projected<1024**3,('Output projection fails',projected)
lock['footprint_projection']=dict(actual_agent56_bytes=meta['bytes'],eight_packets_growth_factor=1.3,reserve_bytes=200*2**20,projected_bytes=projected)
lock['independent_schedule_audit']='PASS: all29184 Cslots rechecked against current minimum cumulative row exposure and exact Ustratum'
(R/'research_notes/r2_ei_qa1_schedule_lock.json').write_text(json.dumps(lock,indent=2))
for d in ('jobs','runs','tmp'): (S/d).mkdir(exist_ok=True)
assert not (S/'ledger.json').exists()
ledger=dict(stage='Issue63-G17-QA1',approval='Yi current conversation 2026-11-15',hard_job_s=3600,hard_output_bytes=1024**3,
 hard_physics_5ms_ticks=18000,charged_upper_s=600,jobs=[dict(name='archive_metadata_schedule_fixtures_preparation',wall_s=600,cost_type='conservative_upper_bound',
 actual_schedule_s=lock['wall_s'],includes='CPU/source reads, archive publication, metadata/footprint tests and first empty-round0 IndexError; no physics/optimizer')],
 source_manifest_sha256=lock['source_manifest_sha256'],status='PRELOCKED_NO_NEW_UPDATE',output_roots=[str(S)])
(S/'ledger.json').write_text(json.dumps(ledger,indent=2))
old=json.loads((O/'ledger.json').read_text());old['output_roots'].append(str(S));old['job_entries'].append(dict(name='G17-QA1-preparation',category='G17-QA1',wall_s=600,cost_type='conservative_upper_bound'))
old['total_charged_upper_s']+=600;old['status']='G17_QA1_APPROVED_PRELOCKED';(O/'ledger.json').write_text(json.dumps(old,indent=2))
print(json.dumps(dict(footprint=lock['footprint_projection'],projected_total_R2_s=old['total_charged_upper_s']+1500+1000+300,schedule_audit=lock['independent_schedule_audit']),indent=2))
