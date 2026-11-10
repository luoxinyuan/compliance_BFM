"""Real CPU process/Pipe tests: per-worker order, policy identity, error propagation."""
import multiprocessing as mp,unittest
from multiprocessing.connection import wait
from frl_inference_rpc import RemoteChannel,validate_request,preserved_row_actions

def client(conn):
 remote=RemoteChannel(conn,'frozen-test-policy')
 for i in range(12):
  assert remote.call('high' if i%2 else 'bfm',dict(value=i))==dict(value=i*2)
 conn.send({'kind':'done'});conn.close()

class RPCTests(unittest.TestCase):
 def test_four_actual_processes(self):
  ctx=mp.get_context('spawn');children=[];active=[];seq={}
  try:
   for _ in range(4):
    parent,child=ctx.Pipe();p=ctx.Process(target=client,args=(child,));p.start();child.close();children.append(p);active.append(parent);seq[parent]=0
   while active:
    available=wait(active,timeout=10);self.assertTrue(available)
    for conn in available:
     msg=conn.recv()
     if msg['kind']=='done':self.assertEqual(seq[conn],12);active.remove(conn);conn.close();continue
     seq[conn]=validate_request(msg,seq[conn],'frozen-test-policy')
     conn.send(dict(sequence=msg['sequence'],policy_sha=msg['policy_sha'],payload=dict(value=msg['payload']['value']*2)))
   for p in children:p.join(10);self.assertEqual(p.exitcode,0)
  finally:
   for p in children:
    if p.is_alive():p.terminate();p.join(5)
 def test_compiled_output_storage_is_not_reused(self):
  import torch
  shared=torch.zeros((1,3))
  def actor(obs,z,mean=True):
   shared.fill_(float(z));return shared
  actions=preserved_row_actions(actor,[{},{}],[1.,2.])
  actor({},9.)
  self.assertTrue(torch.equal(actions[0],torch.ones((1,3))))
  self.assertTrue(torch.equal(actions[1],torch.full((1,3),2.)))
 def test_contract_rejects_version_and_order(self):
  request=dict(kind='bfm',sequence=1,policy_sha='I')
  self.assertEqual(validate_request(request,0,'I'),1)
  for previous,version in [(1,'I'),(0,'different')]:
   with self.assertRaises(AssertionError):validate_request(request,previous,version)
 def test_remote_rejects_wrong_reply(self):
  class Fake:
   def send(self,x):pass
   def recv(self):return dict(sequence=2,policy_sha='I',payload=0)
  with self.assertRaises(AssertionError):RemoteChannel(Fake(),'I').call('bfm',{})
if __name__=='__main__':unittest.main()
