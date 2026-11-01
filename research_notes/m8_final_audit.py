"""Provenance/resource and data-split audit; never run BFM or simulation."""
import ast,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m8-contact-body-20261101A'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def main():
 ledger=json.loads((RUN/'compute-ledger.json').read_text());a=ledger['attempts'];constants={}
 for n in ast.parse((ROOT/'research_notes/g1_action_contract_check.py').read_text()).body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PINS','WEIGHT_SHA']:constants[n.targets[0].id]=ast.literal_eval(n.value)
 pins={f:sha(ROOT/f)==h and sha(P/'repo'/f)==h for f,h in constants['PINS'].items()};assert all(pins.values())
 weight=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model/model.safetensors';assert sha(weight)==constants['WEIGHT_SHA']
 alive=[]
 for x in a:
  for process in x.get('initial_processes',[]):
   f=Path('/proc')/str(process['pid'])/'stat'
   if f.exists() and f.read_text().split()[21]==str(process['start_ticks']):alive.append(process)
 assert not alive
 records=[];ck={}
 for x in a:
  out=Path(x['path'])/'outputs';f=out/'result.json';r=json.loads(f.read_text()) if f.exists() else {};records.append(dict(stage=x['stage'],commit=x['commit'],exit=x.get('exit_code'),status=('failed' if x.get('exit_code',0)!=0 else r.get('status')),raw_result_status=r.get('status'),bfm_unchanged=r.get('original_bfm_unchanged'),updates=r.get('student_optimizer_updates'),wall=x.get('wall_seconds'),owned_pid=x.get('pid'),resource_boundary=x.get('resource_boundary')))
  for f in out.glob('*-final.pt'):ck[str(f)]=sha(f)
  f=out/'new-data-projection.json'
  if f.exists():
   for s in json.loads(f.read_text())['source']:assert not any(k in Path(s['path']).name for k in ['down_test','down_position','wrist_sealed_mid'])
 old_ck={
  'm7-continuous-20261101A/train2-attempt-02/outputs/alignment-final.pt':'4366d30071814a31dd9295fa58b8b8f0ee8046ae6ddb4092e7ebe6c5c4970b1e',
  'm7-continuous-20261101A/train2-attempt-02/outputs/distill-final.pt':'ba3aa67e3d452579e1cb2e40848aebc4d50fce7bb6d428e0b23b9e6726ac316e',
  'm6-multibehavior-20261031A/train2-attempt-01/outputs/alignment-final.pt':'605bed10d1c543e7c399ee14b993c2b8fa5a6fdcddc24c1169404837d8a72053',
  'm6-multibehavior-20261031A/train2-attempt-01/outputs/distill-final.pt':'4c97a85346092956b47823a90f6643fb133d13b193a9a54faa0b687d2a41f694',
  'm5-contact-lpp-20261031A/train2-attempt-01/outputs/alignment-final.pt':'0e14d9f699540b3f5db979d70450fda80e07b307c9a087f8f140fe164d12fe86',
  'm5-contact-lpp-20261031A/train2-attempt-01/outputs/distill-final.pt':'6475261f6109a348bfd50bab1c29f09d94f7f9f0602e7bdb44f507bb4b8c4891',
  'm1-state-response-20261031A/aggregate-attempt-01/outputs/distill-final.pt':'c9036c46a0559c57fc2897419f06c66ef6113cb0fe4d57762df9a59c8b9e1ac7',
  'm1-state-response-20261031A/aggregate-attempt-01/outputs/alignment-final.pt':'e8121fbeb048de67a8147d03b8adbb7d8f302e3d6506bd1a1e0b54d41656683f',
  'm3-continuous-body-20261031A/train-attempt-01/outputs/projected-z-final.pt':'81300c1a1d4abcf25eb58a36a0a92924f92526771ac2c2bbbf9c8eb8f578cf81'}
 preserved={k:sha(RUN.parent/k)==v for k,v in old_ck.items()};assert all(preserved.values())
 summary=dict(preserved_checkpoints=preserved,records=records,total_wall=sum(x.get('wall_seconds',0) for x in a),peak_rss=max(x.get('peak_rss_bytes',0) for x in a),peak_owned_gpu_mib=max(x.get('peak_gpu_mib',0) for x in a),owned_alive=alive,core_pins=pins,bfm_weight_sha256=constants['WEIGHT_SHA'],checkpoints=ck,shared_head=git(P/'repo','rev-parse','HEAD'),shared_status=git(P/'repo','status','--porcelain'),private_head=git(ROOT,'rev-parse','HEAD'),private_status=git(ROOT,'status','--porcelain'),changed_files=git(ROOT,'diff','--name-only','17bbc25c2cb3962273d878038d0eb6c2a2c4eee8','HEAD').splitlines())
 assert not summary['shared_status'];assert all(f.startswith('research_notes/m8_') or f.startswith('research_notes/M8_') or f=='research_notes/test_m8.py' for f in summary['changed_files'])
 (RUN/'analysis/final-audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
