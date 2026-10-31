"""Provenance/resource and data-split audit; never run BFM or simulation."""
import ast,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m5-contact-lpp-20261031A'
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
   for s in json.loads(f.read_text())['source']:assert not s['condition']['name'].startswith('test_')
 summary=dict(records=records,total_wall=sum(x.get('wall_seconds',0) for x in a),peak_rss=max(x.get('peak_rss_bytes',0) for x in a),peak_owned_gpu_mib=max(x.get('peak_gpu_mib',0) for x in a),owned_alive=alive,core_pins=pins,bfm_weight_sha256=constants['WEIGHT_SHA'],checkpoints=ck,shared_head=git(P/'repo','rev-parse','HEAD'),shared_status=git(P/'repo','status','--porcelain'),private_head=git(ROOT,'rev-parse','HEAD'),private_status=git(ROOT,'status','--porcelain'),changed_files=git(ROOT,'diff','--name-only','f538511b9b9aa8ca7b6751aef5f2f5959b730726','HEAD').splitlines())
 assert not summary['shared_status'];assert all(f.startswith('research_notes/m5_') or f.startswith('research_notes/M5_') or f=='research_notes/test_m5.py' for f in summary['changed_files'])
 (RUN/'analysis/final-audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
