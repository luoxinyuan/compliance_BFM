"""Rebuild the versioned executed-action law after an atomic P0-B readback."""
import torch
from fb_safe_action import SafeAction,install,bind_task
from fb_safe_update import install_retention
from fb_ordered_grad import install as ordered


def bind_records(col,transform):
    original_take=col.proxy.take_step;original_act=col.proxy.act
    def act(obs,z,mean=True):
        action=original_act(obs,z,mean)
        if not col.proxy.source_mode:
            record=col.proxy.pending
            if getattr(col.proxy.policy_model,'_safe_action_installed',False):
                record['raw_actor_proposal']=transform.last['raw_proposal'].cpu()
                record['actual_q_PD']=col.task.qof(action).detach().cpu()
                record['safety_transform_active']=transform.last['active'].cpu()
            else:
                record['raw_actor_proposal']=action.detach().cpu()
                record['actual_q_PD']=col.task.qof(action).detach().cpu()
                record['safety_transform_active']=torch.zeros_like(action,dtype=torch.bool).cpu()
        return action
    def take(raw,native_limit=5):
        rows,done,stats=original_take(raw,native_limit)
        for index,(row,record) in enumerate(zip(rows,col.proxy.records)):
            for key in ('raw_actor_proposal','actual_q_PD','safety_transform_active'):
                row[key]=record[key]
            control=col.task.controls[-len(rows)+index]
            assert torch.equal(row['actual_q_PD'][0],torch.as_tensor(control['q_PD'],dtype=torch.float32)), 'Recorded actual q_PD differs from executor'
        return rows,done,stats
    col.proxy.act=act;col.proxy.take_step=take


def restore(path,e,m,contract):
    import json
    from diagnostic_fb_resume import restore as full_restore
    metadata=json.loads((path/'manifest.json').read_text())
    safety=metadata['action_contract']['safe_execution_v1']
    assert safety['version']=='FB_SAFE_ANKLE_EXECUTION_V1' and safety['ankle_target_inset_rad']==.05 and safety['handoff_s']==.35
    contract['safe_execution_v1']=safety
    agent,col,progress,meta=full_restore(path,e,m,contract)
    transform=SafeAction(contract,safety['joint_ranges'],'cuda')
    ordered(agent,record_gradients=False);install(agent._model,m,transform);bind_task(col.task)
    agent._safe_qpd_scale=transform.scale;agent._safe_action_transform=transform
    install_retention(agent,m);bind_records(col,transform)
    return agent,col,progress,meta
