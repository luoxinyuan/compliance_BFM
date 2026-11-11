from pathlib import Path
import re,json,hashlib,subprocess
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/f1-l0-array-audit-20261111';inc=P/'.venv/lib/python3.10/site-packages/mujoco/include';h=inc/'mujoco/mjdata.h'
names=re.findall(r'^\s*(?:mjtNum|int|mjtByte|float|double|char|void)\s*\*\s*(\w+)\s*;',h.read_text(),re.M)
source='#include <stddef.h>\n#include <mujoco/mjdata.h>\nsize_t struct_size(void){return sizeof(mjData);}\nsize_t field_offset(int k){switch(k){\n'+''.join('case %d:return offsetof(mjData,%s);\n'%(i,n) for i,n in enumerate(names))+'default:return (size_t)-1;}}\n'
(S/'prep/pointer_layout.c').write_text(source);subprocess.run(['cc','-shared','-fPIC','-I'+str(inc),str(S/'prep/pointer_layout.c'),'-o',str(S/'prep/pointer_layout.so')],check=True)
(S/'prep/pointer_layout.json').write_text(json.dumps(dict(fields=names,header_sha256=hashlib.sha256(h.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest()),indent=2))
print(len(names))
