"""Actual native source z/action attached to original eight-frame CPR snippets."""
import torch
from dual_track_native_data import NativeSequences

class SafeNativeSequences(NativeSequences):
    def add(self,source,parent_id,rows):
        if any(x['parent_id']==parent_id for x in self.parents[source]):return
        super().add(source,parent_id,rows)
        added=[x for x in self.parents[source] if x['parent_id']==parent_id]
        if not added:return
        starts=torch.linspace(0,len(rows)-8,steps=min(16,len(rows)-7)).long().unique().tolist()
        for block,start in zip(added[0]['blocks'],starts):
            block['native_z']=torch.cat([r['z'] for r in rows[start:start+8]])
            block['native_action']=torch.cat([r['action'] for r in rows[start:start+8]])

    def sample(self,batch_size):
        assert batch_size%8==0
        available=[k for k,v in self.parents.items() if v];blocks=[]
        for _ in range(batch_size//8):
            source=available[int(torch.randint(len(available),()).item())]
            parents=self.parents[source];group=parents[int(torch.randint(len(parents),()).item())]
            blocks.append(group['blocks'][int(torch.randint(len(group['blocks']),()).item())])
        result=dict(observation={k:torch.cat([b['observation'][k] for b in blocks]) for k in blocks[0]['observation']},
            next=dict(observation={k:torch.cat([b['next']['observation'][k] for b in blocks]) for k in blocks[0]['observation']}))
        result.update(native_z=torch.cat([b['native_z'] for b in blocks]),native_action=torch.cat([b['native_action'] for b in blocks]))
        return result
