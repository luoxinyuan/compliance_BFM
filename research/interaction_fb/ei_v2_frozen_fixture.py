"""Full frozen qualification states with one immutable shared fresh-model header.
No raw state, Adam, RNG, solver or actual command history is omitted.
"""
import copy,json
from pathlib import Path
from .journal import dump,read
from .checkpoint import sha
from .ei_v2_checkpoint import OWN,LISTS
from diagnostic_frozen import portable
from f1_mjdata_contract import metadata
from dual_track_task import global_rng

def model_header(folder,agent,runtime):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True);path=folder/'frozen-model.ptz';assert not path.exists()
    m=dump(path,dict(agent=agent.state(),runtime=runtime,identity='fresh_frozen_qualification_initial_model_never_formal_trained'))
    return dict(path=str(path),**m)

def save(path,agent,env,header,runtime,identity):
    path=Path(path);assert not path.exists()
    own={k:copy.deepcopy(getattr(env,k)) for k in OWN}
    data=dict(format='EI-v2-frozen-source-v1',agent_reference=header,own=own,physical=portable(env.task),mj_metadata=metadata(env.e.sim.model,env.e.sim.data),rng=global_rng(),task_rng=copy.deepcopy(env.task.rng.bit_generator.state),physics_rng=copy.deepcopy(env.task.physics_rng),source_sha=env.task.prefix_data['audit']['sha256'],proxy_previous_z=env.proxy.previous_z,proxy_actions=env.proxy.actions,block='frozen-source',runtime=runtime,actual_raw_target=getattr(env,'raw_target',None),stochastic=env.proxy.stochastic,identity=identity)
    for k in LISTS:data[k]=copy.deepcopy(getattr(env,k))
    m=dump(path,data);path.with_suffix('.manifest.json').write_text(json.dumps(dict(format=data['format'],file=path.name,**m),indent=2)+'\n')
    return dict(path=str(path),**m)

def load(path):
    path=Path(path);m=json.loads(path.with_suffix('.manifest.json').read_text());assert m['format']=='EI-v2-frozen-source-v1' and m['file']==path.name
    s=read(path,m);ref=s['agent_reference'];rp=Path(ref['path']);stage=Path(__file__).resolve().parents[2].parent
    rp.resolve().relative_to(stage.resolve())
    agent=read(rp,ref);s['agent']=agent['agent'];return s
