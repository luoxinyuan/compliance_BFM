"""CPU negative regressions for immutable publication, partial-write isolation and hashes."""
import hashlib,json,os,signal,sys
from pathlib import Path
import diagnostic_fb_resume as safe
from dual_track_fb_online import load_package
from dual_track_fb_extension import EXTENSION_VERSION
from dual_track_high import sha,write

def main(source,out):
 out.mkdir(exist_ok=False);before=sha(source/'manifest.json')
 try:safe.publish(source,None,None,0,'fixture',{},'user_stop');raise RuntimeError('Overwrite was allowed')
 except AssertionError:pass
 assert sha(source/'manifest.json')==before
 fixture=out/'corrupt-fixture';fixture.mkdir();(fixture/'fixture.txt').write_text('tampered')
 write(fixture/'manifest.json',dict(version='DUAL_TRACK_FB_FULL_V1',extension=EXTENSION_VERSION,
  completed_files={'fixture.txt':hashlib.sha256(b'original').hexdigest()}))
 try:load_package(fixture,None,None,None,physical=False);raise RuntimeError('Hash corruption accepted')
 except AssertionError:pass
 # Exercise the actual publisher's failure path before rename. No model is built.
 pointer=out/'latest-complete.json';pointer.write_text('{"previous":"retained"}');old=pointer.read_bytes()
 original=safe.save_package
 def broken(path,*args):
  path.mkdir();(path/'incomplete').write_text('injected write failure');raise OSError('injected pre-publication failure')
 safe.save_package=broken
 try:
  try:safe.publish(out/'never-published',None,None,0,'fixture',{},'rolling');raise RuntimeError('Write failure ignored')
  except OSError:pass
 finally:safe.save_package=original
 assert pointer.read_bytes()==old and not (out/'never-published').exists()
 h1,h2=signal.getsignal(signal.SIGTERM),signal.getsignal(signal.SIGINT)
 try:
  flag=safe.NormalStop(out);os.kill(os.getpid(),signal.SIGTERM);assert flag.pending() and flag.reason=='signal:15'
 finally:signal.signal(signal.SIGTERM,h1);signal.signal(signal.SIGINT,h2)
 # External-stop gate itself must return before any caller could update.
 from types import SimpleNamespace
 called=[];original=safe.publish
 safe.publish=lambda *args,**kwargs:called.append(args)
 try:
  res=safe.honor_external_stop(SimpleNamespace(pending=lambda:True),None,SimpleNamespace(transitions=1037),1,'fixture',out)
  assert res['status']=='stopped_by_request' and len(called)==1 and called[0][5]['collected_within_block']==13
 finally:safe.publish=original
 write(out/'result.json',dict(status='completed',overwrite_rejected=True,original_manifest_unchanged=True,
  hash_corruption_rejected_before_model_load=True,partial_write_not_published=True,previous_pointer_retained=True,
  SIGTERM_records_intent_without_async_saving=True,external_stop_gate_returns_before_updates=True,GPU_used=False,updates=0))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
