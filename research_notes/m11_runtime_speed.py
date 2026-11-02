"""Resolve available static OmegaConf nodes once; retain unused unresolved nodes."""
import hashlib,json
from omegaconf import OmegaConf,DictConfig,ListConfig
from omegaconf.errors import InterpolationResolutionError,MissingMandatoryValue

def install(env):
 cfg=env.base.config;resolved=[];skipped=[];original=OmegaConf.to_yaml(cfg,resolve=False)
 def plain(v):return OmegaConf.to_container(v,resolve=True) if OmegaConf.is_config(v) else v
 def visit(node,path=''):
  for key in list(node.keys()) if isinstance(node,DictConfig) else range(len(node)):
   name=path+'.'+str(key)
   try:
    if OmegaConf.is_interpolation(node,key):
     value=plain(node[key]);node[key]=value;assert plain(node[key])==value;resolved.append(dict(path=name,value=value))
    value=node[key]
   except (InterpolationResolutionError,MissingMandatoryValue) as exc:
    skipped.append(dict(path=name,error=type(exc).__name__));continue
   if isinstance(value,(DictConfig,ListConfig)):visit(value,name)
 visit(cfg)
 return dict(mode='resolved_available_configuration',resolved_nodes=resolved,skipped_unused_nodes=skipped,interpolations_before=original.count('${'),interpolations_after=OmegaConf.to_yaml(cfg,resolve=False).count('${'),resolved_manifest_sha256=hashlib.sha256(json.dumps(resolved,sort_keys=True,default=str).encode()).hexdigest(),bfm_weights_updated=False,pd_algorithm_changed=False)
