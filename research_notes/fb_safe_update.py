# Copyright (c) Meta Platforms, Inc. and affiliates.
# Derived portions follow the original CC BY-NC 4.0 LICENSE.
"""Joint actor loss adds only native no-contact actual-qPD preservation."""
import types
import torch
from torch.amp import autocast
from fb_safe_action import KEY
from dual_track_fb_extension import CONTACT_KEY


def install_retention(agent,original):
    update=agent.update
    def with_expert(self,buffers,step):
        class Record:
            def __init__(self,buffer):self.buffer=buffer
            def sample(self,n):
                batch=self.buffer.sample(n);agent._native_retention_batch=batch;return batch
        return update(dict(buffers,expert_slicer=Record(buffers['expert_slicer'])),step)
    agent.update=types.MethodType(with_expert,agent)
    def actor(self,obs,action,z,clip_grad_norm):
        with autocast(device_type=self.device,dtype=self._model.amp_dtype,enabled=self.cfg.model.amp):
            dist=self._model._actor(obs,z,self._model.cfg.actor_std)
            executed=dist.sample(clip=self.cfg.train.stddev_clip)
            qd=self._model._critic(obs,z,executed)
            _,_,Qd=self.get_targets_uncertainty(qd,self.cfg.train.actor_pessimism_penalty)
            qa=self._model._aux_critic(obs,z,executed)
            _,_,Qa=self.get_targets_uncertainty(qa,self.cfg.train.actor_pessimism_penalty)
            fs=self._model._forward_map(obs,z,executed);qf=(fs*z).sum(-1)
            _,_,Qf=self.get_targets_uncertainty(qf,self.cfg.train.actor_pessimism_penalty)
            weight=Qf.abs().mean().detach() if self.cfg.train.scale_reg else 1.
            task_loss=-Qd.mean()*self.cfg.train.reg_coeff*weight-Qa.mean()*self.cfg.train.reg_coeff_aux*weight-Qf.mean()
            # Actual real native source observations; no unsafe combination labels.
            expert={k:v.to(self.device) for k,v in self._native_retention_batch['observation'].items()}
            contact=expert[CONTACT_KEY]
            mask=(contact[:,21]==0)&(contact[:,6:9].abs().sum(-1)==0)
            # The protected function is the fully handed-over controller on
            # reliable native states, not a zero-gradient alpha=0 source copy.
            expert[KEY]=torch.ones(len(contact),1,device=self.device)
            with torch.no_grad():
                prompt=self._native_retention_batch['native_z'].to(self.device)
                nominal=self._native_retention_batch['native_action'].to(self.device).clamp(-1,1)
                transform=self._safe_action_transform
                nominal_qpd=transform.offset+nominal*transform.scale
                ankle=nominal_qpd[:,transform.indices]
                mask=mask&((ankle>=transform.lower)&(ankle<=transform.upper)).all(-1)
            normalized=self._model._normalize(expert)
            actual=self._model._actor(normalized,prompt,self._model.cfg.actor_std).mean
            # Same original PD scale; no change to original PD values.
            delta=(actual-nominal.clamp(-1,1))*self._safe_qpd_scale
            keep=delta[mask].square().mean()/.01 if bool(mask.any()) else task_loss.new_zeros(())
            loss=task_loss+.1*keep
        self.actor_optimizer.zero_grad(set_to_none=True)
        from fb_ordered_grad import ordered_component_gradients
        parameters=tuple(self._model._actor.parameters())
        components=[task_loss,.1*keep] if bool(mask.any()) else [task_loss]
        for parameter,gradient in zip(parameters,ordered_component_gradients(components,parameters)):
            parameter.grad=gradient
        if clip_grad_norm is not None:torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(),clip_grad_norm)
        self.actor_optimizer.step()
        return dict(actor_loss=loss.detach(),Q_discriminator=Qd.mean().detach(),Q_aux=Qa.mean().detach(),Q_fb=Qf.mean().detach(),
            native_no_contact_qPD_keep=keep.detach(),native_keep_valid_fraction=mask.float().mean().detach())
    agent.update_actor=types.MethodType(actor,agent)
