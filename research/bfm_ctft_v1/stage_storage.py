"""Read-only project-local output inventory; never deletes or follows links."""
import os
from pathlib import Path
import time


def footprint(stage):
    stage=Path(stage)
    begun=time.monotonic()
    logical=unique=files=0
    inodes=set()
    groups={}
    links=[]
    for root, dirs, names in os.walk(stage, followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink()]
        for name in names:
            p=Path(root)/name
            if p.is_symlink():
                links.append(str(p.relative_to(stage)))
                continue
            st=p.stat()
            rel=p.relative_to(stage)
            group='/'.join(rel.parts[:2]) if len(rel.parts)>1 else rel.parts[0]
            groups[group]=groups.get(group,0)+st.st_size
            logical+=st.st_size
            files+=1
            inode=(st.st_dev,st.st_ino)
            if inode not in inodes:
                unique+=st.st_size
                inodes.add(inode)
    return dict(logical_bytes=logical, unique_inode_logical_bytes=unique,
                files=files, groups=groups, symlinks_not_followed=links,
                wall_s=time.monotonic()-begun, actual_allocated_CPFS_bytes='UNKNOWN',
                no_deletion=True)
