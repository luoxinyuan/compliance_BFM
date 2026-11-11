"""Read selected upstream source/configs and execute two reviewed tensor-only helpers.
No external model construction, downloads, package install or simulation.
"""
import argparse,json,time,importlib.util,sys,platform
from pathlib import Path
from .provenance import file_sha

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic()
    import torch,mujoco
    from .backends import pbfm_velocity_command
    root=a.sources;original=json.loads(a.original.read_text());pd=original['action_contract'];pdir=root/'P-BFM';idir=root/'I-BFM-Code'
    pc=json.loads((pdir/'interactive/pbfm/config.json').read_text());ic=json.loads((idir/'checkpoints/ibfm-166m/config.json').read_text());init=json.loads((idir/'checkpoints/ibfm-166m/model/init_kwargs.json').read_text())
    native=module(pdir/'pbfm/agents/velocity_context.py','s0_upstream_velocity')
    sphere=module(idir/'ibfm/humanoidverse/agents/spherical.py','s0_upstream_sphere')
    generator=torch.Generator().manual_seed(6100);n=0
    for i in range(32):
        v=torch.randn(1,2,generator=generator);h=torch.randn(1,2,generator=generator);valid=torch.tensor([[i%2]],dtype=torch.bool)
        assert torch.equal(native.velocity_observation(v,h,valid),pbfm_velocity_command(v,h,valid));n+=1
    q=torch.ones(1,256);assert torch.equal(sphere.sphere_log_map(q,q),torch.zeros_like(q))
    assert abs(float(sphere.sphere_log_map(q,-q).norm())-16*torch.pi)<1e-5
    assert pd['dof_names']==pc['jointNames']==ic['env']['robot_training']['robot']['control_joint_names']
    diffs=[]
    for j,name in enumerate(pd['dof_names']):
        stem=name.removeprefix('left_').removeprefix('right_').removesuffix('_joint')
        origraw=pd['q_PD_scale'][0][j];newraw=pc['actionScale'][j]*pc['actionMultiplier']
        oldkp=pd['control']['stiffness'][stem];oldkd=pd['control']['damping'][stem]
        ikp=ic['env']['robot_training']['stiffness'][stem];ikd=ic['env']['robot_training']['damping'][stem]
        diffs.append(dict(joint=name,original_effective_raw_scale=origraw,pbfm_effective_raw_scale=newraw,original_kp=oldkp,pbfm_kp=pc['kp'][j],ibfm_kp=ikp,original_kd=oldkd,pbfm_kd=pc['kd'][j],ibfm_kd=ikd))
    refs=json.loads((root/'provenance.json').read_text());manifest=[]
    for repo,ref in [('P-BFM',refs['repositories'][1]['ref']),('I-BFM-Code',refs['repositories'][3]['ref'])]:
        for f in sorted((root/repo).rglob('*')):
            if not f.is_file() or '__pycache__' in str(f):continue
            rel=str(f.relative_to(root/repo));r=refs['repositories'][2]['ref'] if rel.startswith('interactive/') else ref
            manifest.append(dict(repo='Roboparty/'+repo,ref=r,path=rel,sha256=file_sha(f),bytes=f.stat().st_size,url=f'https://github.com/Roboparty/{repo}/blob/{r}/{rel}'))
    result=dict(status='passed',joint_order_exact=True,PD_differences=diffs,pbfm_helper_exact_cases=n,ibfm_spherical_helper_tests=2,ibfm_observation_dimensions={k:v['shape'] for k,v in init['obs_space']['spaces'].items()},source_manifest=manifest,installed=dict(python=platform.python_version(),torch=torch.__version__,mujoco=mujoco.__version__,onnxruntime_available=importlib.util.find_spec('onnxruntime') is not None),external_model_forwards=0,physics_steps=0,weights_downloaded=0,job_s=time.monotonic()-start)
    a.output.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
