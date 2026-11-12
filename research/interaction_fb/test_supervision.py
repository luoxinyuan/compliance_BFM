import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('supervision',Path(__file__).with_name('supervision.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SupervisionTests(unittest.TestCase):
    def test_rename_retry_and_hardlink_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pending=root/'.pending';pending.mkdir()
            (pending/'data').write_bytes(b'x'*321)
            os.link(pending/'data',root/'linked')
            original=m.os.scandir; fired=[]
            def scan(path):
                if Path(path)==pending and not fired:
                    fired.append(True);pending.rename(root/'round-1')
                return original(path)
            with patch.object(m.os,'scandir',scan):
                self.assertEqual(m.tree_bytes([root]),321)
            self.assertTrue(fired)
    def test_persistent_race_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(m.os,'scandir',side_effect=FileNotFoundError('injected')):
                with self.assertRaises(FileNotFoundError):m.tree_bytes([tmp],retries=2)
    def test_monitor_exception_reaps_owned_child(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
        try:
            with self.assertRaises(OSError):
                try:raise OSError('injected monitor IO failure')
                finally:m.stop_owned_group(child,grace=1)
            self.assertIsNotNone(child.poll())
            with self.assertRaises(ChildProcessError):os.waitpid(child.pid,os.WNOHANG)
        finally:m.stop_owned_group(child,grace=1)
    def test_actual_queue_run_reaps_on_scanner_failure(self):
        import ast,json,signal,time
        from types import SimpleNamespace
        source=Path(__file__).with_name('pilot_queue.py').read_text()
        tree=ast.parse(source)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'jobs').mkdir();(root/'research/interaction_fb').mkdir(parents=True)
            (root/'research/interaction_fb/benchmark.py').write_text('import time; time.sleep(60)')
            calls=[]
            def size():
                calls.append(1)
                if len(calls)>1:raise OSError('injected scanner failure')
                return 0
            def write(p,x):p.write_text(json.dumps(x))
            ns=dict(gpu_free=lambda:None,shutil=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=200*2**30)),
                    S=root,R=root,J=root/'jobs',PY=Path(sys.executable),CAP=28800,OUTCAP=8*2**30,
                    spent=lambda:0,size=size,time=time,subprocess=subprocess,env=os.environ.copy(),runtime='test',
                    lock={'approval_reference':'test'},write=write,os=os,signal=signal,
                    stop_owned_group=lambda p,grace:m.stop_owned_group(p,grace=.1))
            exec(compile(ast.Module(body=[run],type_ignores=[]),'actual_queue_run','exec'),ns)
            with self.assertRaisesRegex(OSError,'injected scanner failure'):ns['run']('test',[])
            row=json.loads((root/'jobs/job-test.json').read_text())
            self.assertIsNotNone(row['exit_code']);self.assertGreaterEqual(row['wall_s'],0)
            self.assertIn('output_measurement_error',row)
            with self.assertRaises(ChildProcessError):os.waitpid(row['pid'],os.WNOHANG)

    def test_ignoring_sigterm_is_killed_and_reaped(self):
        child=subprocess.Popen([sys.executable,'-u','-c',
            'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'],
            start_new_session=True,stdout=subprocess.PIPE,text=True)
        self.assertEqual(child.stdout.readline().strip(),'ready')
        try:self.assertEqual(m.stop_owned_group(child,grace=.05),-9)
        finally:
            m.stop_owned_group(child,grace=.05);child.stdout.close()

if __name__=='__main__':unittest.main()
