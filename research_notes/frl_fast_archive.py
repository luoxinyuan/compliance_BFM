"""Process-local lossless gzip/JSON output acceleration; never touches tensors or RNG."""
import gzip,io,json
from contextlib import contextmanager

def dump_block(obj, fp, *args, **kwargs):
    """Equivalent uncompressed JSON; avoid many Python calls into gzip.write."""
    fp.write(json.dumps(obj, *args, **kwargs))

@contextmanager
def installed(level=1):
    if level not in (1,9):raise ValueError('Only independently benchmarked levels 1 or 9')
    original_open,original_dump=gzip.open,json.dump
    def open_gzip(filename,mode='rb',compresslevel=9,encoding=None,errors=None,newline=None):
        selected=level if any(k in mode for k in ('w','a','x')) else compresslevel
        return original_open(filename,mode,selected,encoding,errors,newline)
    def dump(obj,fp,*args,**kwargs):
        if isinstance(fp,io.TextIOWrapper) and isinstance(fp.buffer,gzip.GzipFile):
            return dump_block(obj,fp,*args,**kwargs)
        return original_dump(obj,fp,*args,**kwargs)
    gzip.open,json.dump=open_gzip,dump
    try:yield
    finally:gzip.open,json.dump=original_open,original_dump
