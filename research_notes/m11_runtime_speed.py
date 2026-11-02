"""Resolve static OmegaConf interpolation once; no BFM/PD algorithm replacement."""
import hashlib,json
from omegaconf import OmegaConf

def install(env):
 cfg=env.base.config
 before=OmegaConf.to_container(cfg,resolve=True)
 unresolved=OmegaConf.to_yaml(cfg,resolve=False)
 OmegaConf.resolve(cfg)
 after=OmegaConf.to_container(cfg,resolve=True)
 assert before==after,'Resolved deployment configuration changed'
 return dict(mode='resolved_configuration',interpolations_before=unresolved.count('${'),interpolations_after=OmegaConf.to_yaml(cfg,resolve=False).count('${'),resolved_sha256=hashlib.sha256(json.dumps(after,sort_keys=True,default=str).encode()).hexdigest(),bfm_weights_updated=False,pd_algorithm_changed=False)
