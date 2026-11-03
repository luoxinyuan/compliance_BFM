import torch

def expand(net,ck):
 state=ck['state_dict'];new=net.state_dict()
 for k,v in state.items():
  if v.shape==new[k].shape:new[k].copy_(v)
  elif k in ['center','scale']:new[k][:len(v)].copy_(v)
  elif k in ['actor.0.weight','critic.0.weight']:
   new[k].zero_();new[k][:,:v.shape[1]].copy_(v)
  else:raise ValueError((k,v.shape,new[k].shape))
 net.load_state_dict(new)
