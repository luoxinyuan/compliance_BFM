import hashlib,json,time
from pathlib import Path
s=Path.cwd()
def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
files={};size=0
for p in sorted(s.rglob('*')):
    if not p.is_file() or p.name=='ledger69.json' or 'ipc' in p.parts:continue
    files[str(p.relative_to(s))]=sha(p);size+=p.stat().st_size
d0=read(s/'audit/identity_and_projection.json');g=read(s/'gradient.json')
phys=read(s/'d2/cost.json')
post=sum(float((s/n).read_text()) for n in ['pairs.wall','video.wall','verification.wall'])
out=dict(D0_archive_body_s=d0['seconds'],D0_grad_body_s=g['seconds'],D0_inclusive_upper_s=600,D1_physics_including_cold_from_launch_s=phys['seconds'],D1_cold_import_and_scheduling_io_allowance_s=120,D1_inclusive_upper_s=phys['seconds']+120,postprocess_subjob_s=post,postprocess_plus_packing_transfer_and_publication_upper_s=max(600,post+300),total_job_upper_s=600+phys['seconds']+120+max(600,post+300),raw_physics_ticks=phys['ticks'],new_branches=phys['branches'],reused_A=24,updates=0,raw_output_bytes_at_inventory=size,source_and_asset_sha256=files)
assert out['total_job_upper_s']<21600 and phys['ticks']<=400000 and phys['branches']<=72 and size+256*2**20<32*2**30
(s/'ledger69.json').write_text(json.dumps(out,indent=2))
