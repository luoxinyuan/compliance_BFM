"""Use unchanged v1.1 physical protection during Q-only actual collection."""
from pathlib import Path
from .env import Environment
from .qualification_v11 import install_gate,same_state_reference
from .gate_v11 import metrics,physical_failures,actual_release_index

class GuardedEnvironment(Environment):
    def reset(self,i=0):
        super().reset(i);self.guard_reference=None;install_gate(self)
    def collect_call(self):
        t=self.task
        if not t.done and self.guard_reference is None and any(p['active'] for p in t.physics) and float(t.e.tether.time)>=t.spec['release']-1e-8:
            dest=Path(self.guard_output)/('recovery-reference-'+str(len(self.episodes)))
            dest.mkdir(exist_ok=False)
            self.guard_reference=same_state_reference(self,self.proxy.agent,dest,reference_collect=lambda: Environment.collect_call(self))
            assert not self.guard_reference['failures'],'Frozen recovery reference failed'
        result=super().collect_call()
        t=self.task;index=actual_release_index(t.physics)
        if index is not None and self.guard_reference is not None:
            fail=physical_failures(metrics(t.physics[index:]),'recovery',self.guard_reference['quality'])
            assert not fail,('Recovery protection failed',fail)
        return result
