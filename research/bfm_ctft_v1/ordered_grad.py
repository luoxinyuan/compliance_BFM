"""AOT-safe lifetime for the already approved ordered discriminator gradients.

All expert/train/GP branches and FP32 accumulation order are unchanged. AOT may
share a forward tape between expert and train outputs, even when their eager
graphs appear disjoint. Keep that tape alive until the last component gradient.
"""
import inspect
import textwrap
import types
import torch
import fb_ordered_grad as original


def ordered_component_gradients(losses, parameters):
    params=tuple(parameters);sums=[None]*len(params)
    for component,loss in enumerate(losses):
        gradients=torch.autograd.grad(loss,params,allow_unused=True,
                                      retain_graph=component+1<len(losses))
        for index,gradient in enumerate(gradients):
            if gradient is None:continue
            value=gradient.detach()
            if sums[index] is None:sums[index]=value.clone()
            else:sums[index].add_(value)
    return sums


namespace=dict(original.update_discriminator.__globals__)
namespace['ordered_component_gradients']=ordered_component_gradients
exec(compile(textwrap.dedent(inspect.getsource(original.update_discriminator)),
             '<ctft-ordered-grad-tape-lifetime>','exec'),namespace)
update_discriminator=namespace['update_discriminator']


def install(agent,record_gradients=False):
    assert record_gradients is False
    assert not getattr(agent,'_ordered_discriminator_installed',False)
    agent.update_discriminator=types.MethodType(update_discriminator,agent)
    agent._ordered_discriminator_installed=True
    agent._repair_events=[]
    return agent
