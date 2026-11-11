"""Audit-only ctypes layout from installed MuJoCo header; no engine replacement."""
import ctypes,json,hashlib
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[2]
lib=ctypes.CDLL(str(S/'prep/pointer_layout.so'));lib.field_offset.argtypes=[ctypes.c_int];lib.field_offset.restype=ctypes.c_size_t;lib.struct_size.restype=ctypes.c_size_t
layout=json.loads((S/'prep/pointer_layout.json').read_text());OFFSETS={n:int(lib.field_offset(i)) for i,n in enumerate(layout['fields'])}
def slot(d,k):return ctypes.c_void_p.from_address(d._address+OFFSETS[k])
def null_arrays(d):return sorted(k for k in OFFSETS if slot(d,k).value is None and isinstance(getattr(d,k),np.ndarray))
def struct_without(d,names):
 b=bytearray(ctypes.string_at(d._address,lib.struct_size()))
 for k in names:b[OFFSETS[k]:OFFSETS[k]+ctypes.sizeof(ctypes.c_void_p)]=bytes(ctypes.sizeof(ctypes.c_void_p))
 return hashlib.sha256(b).hexdigest()
def canonical_before(x,nulls):
 y=dict(x);y['mjdata_arrays']={k:v for k,v in x['mjdata_arrays'].items() if k not in nulls};return y
