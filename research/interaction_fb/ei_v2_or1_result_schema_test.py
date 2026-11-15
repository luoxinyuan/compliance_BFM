"""Actual-result envelope compatibility regression; no torch/physics import."""
from pathlib import Path
import ast,json,argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();a.scratch.mkdir(parents=True,exist_ok=False)
 source=Path(__file__).with_name('ei_v2_or1_scratch.py');tree=ast.parse(source.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='result_payload');ns=dict(Path=Path,json=json);exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),'exec'),ns)
 saved=json.loads(a.fixture.read_text());assert 'study'in saved and 'expected'in saved['study'];actual=ns['result_payload'](a.fixture);assert actual==saved['study']
 bare=a.scratch/'bare.json';bare.write_text(json.dumps(actual));assert ns['result_payload'](bare)==actual
 bad=a.scratch/'bad.json';bad.write_text(json.dumps(dict(status='wrong',study=actual)))
 try:ns['result_payload'](bad);raise RuntimeError('conflicting status passed')
 except AssertionError:pass
 r=dict(status='ACTUAL_RESULT_SCHEMA_PASS_NOT_GPU_RECOVERY',wrapped_and_bare_exact=True,conflicting_status_rejected=True,expected_meta_exact=True,no_learning_or_physics=True)
 (a.scratch/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

if __name__=='__main__':main()
