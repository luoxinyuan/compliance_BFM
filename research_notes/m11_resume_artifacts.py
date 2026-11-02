"""Copy immutable continuation evidence; never append through a hard link to old logs."""
import json,shutil
from pathlib import Path
from m11_selection import atomic,sha

def restore(old,new,batches,source_sha,plan_sha):
 old,new=Path(old),Path(new)
 launch=json.loads((old/'launch.json').read_text())
 assert launch['plan_sha256']==plan_sha and launch['source_sha256']==source_sha
 rows=[json.loads(x) for x in (old/'training.jsonl').read_text().splitlines()]
 kept=[x for x in rows if x['batch']<=batches]
 assert len(kept)==batches and [x['batch'] for x in kept]==list(range(1,batches+1))
 (new/'training.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in kept))
 for folder in [old/'baseline']+sorted(old.glob('dev-*')):
  if folder.name!='baseline' and (int(folder.name.split('-')[1])>batches or not (folder/'score.json').exists()):continue
  shutil.copytree(folder,new/folder.name,copy_function=shutil.copy2)
 selected=json.loads((old/'selection.json').read_text());assert selected['metadata']['batches']<=batches
 assert sha(old/'best.pt')==selected['checkpoint_sha256']
 shutil.copy2(old/'best.pt',new/'best.pt');atomic(new/'selection.json',selected)
 atomic(new/'resume-evidence.json',dict(previous_job=str(old),completed_batches=batches,previous_log_sha256=sha(old/'training.jsonl'),preserved_selection_sha256=selected['checkpoint_sha256'],old_files_modified=False,worker_state='Fresh worker reset; matched reset at same batch in B; partial old rollouts not claimed resumed'))
 return selected,kept[-1]['elapsed_s']
