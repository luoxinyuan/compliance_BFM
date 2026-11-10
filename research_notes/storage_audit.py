"""Read-only checkpoint/stage inventory. No deletion, relocation, or file content reads."""
import argparse,json,os,shutil,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 start=time.monotonic();rows=[];stages={};seen=set();errors=[]
 roots=[a.project/'experiments',a.project/'bridge-v2-owned/runs']
 for root in roots:
  for folder,dirs,files in os.walk(root,followlinks=False):
   dirs[:]=[d for d in dirs if d not in ('.git','.cache','__pycache__','torch-cache','compile-cache')]
   for name in files:
    path=Path(folder)/name
    try:st=path.lstat()
    except OSError as exc:errors.append(dict(path=str(path),error=str(exc)));continue
    if path.is_symlink():continue
    rel=path.relative_to(root);stage=str(root.relative_to(a.project))+'/'+rel.parts[0]
    data=stages.setdefault(stage,dict(logical_bytes=0,unique_allocated_bytes=0,files=0,checkpoint_files=0,checkpoint_bytes=0))
    data['logical_bytes']+=st.st_size;data['files']+=1
    inode=(st.st_dev,st.st_ino)
    if inode not in seen:data['unique_allocated_bytes']+=st.st_blocks*512;seen.add(inode)
    checkpoint=(path.suffix in ('.pt','.pth','.safetensors') or name in ('manifest.json','latest-complete.json'))
    if checkpoint:
     data['checkpoint_files']+=1;data['checkpoint_bytes']+=st.st_size
     rows.append(dict(path=str(path.relative_to(a.project)),bytes=st.st_size,allocated_bytes=st.st_blocks*512,hardlinks=st.st_nlink))
 result=dict(status='read_only_inventory_completed',shared_free_bytes=shutil.disk_usage(a.project).free,
  shared_guard_bytes=100*(1<<30),stages=stages,checkpoint_files=rows,errors=errors,wall_s=time.monotonic()-start,
  candidates='Completed immutable checkpoint/replay packages may be copied to verified archival storage after permission, hashes and compatibility-reference review. No item automatically deletable. Active runtimes, shared assets, original checkpoints and errata protected.',
  modified_or_deleted_assets=0)
 a.output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(stages=len(stages),checkpoint_files=len(rows),shared_free_bytes=result['shared_free_bytes'],wall_s=result['wall_s'])))
if __name__=='__main__':main()
