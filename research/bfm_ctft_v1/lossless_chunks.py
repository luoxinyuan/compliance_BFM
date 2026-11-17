"""Stage-local lossless archive codec using the already installed libzstd.

No dependency installation, tensor casting or state omission. Raw serialized
bytes are chunked and independently compressed; a full raw SHA256 is checked
after decoding. Original checkpoint files are never deleted by this tool.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import struct
import time

MAGIC=b'CTFTZ001'
CHUNK=8<<20
lib=ctypes.CDLL(ctypes.util.find_library('zstd') or 'libzstd.so.1')
lib.ZSTD_compressBound.argtypes=[ctypes.c_size_t]
lib.ZSTD_compressBound.restype=ctypes.c_size_t
lib.ZSTD_compress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
lib.ZSTD_compress.restype=ctypes.c_size_t
lib.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
lib.ZSTD_decompress.restype=ctypes.c_size_t
lib.ZSTD_isError.argtypes=[ctypes.c_size_t]
lib.ZSTD_isError.restype=ctypes.c_uint
lib.ZSTD_getErrorName.argtypes=[ctypes.c_size_t]
lib.ZSTD_getErrorName.restype=ctypes.c_char_p


def checked(n):
    if lib.ZSTD_isError(n):raise RuntimeError(lib.ZSTD_getErrorName(n).decode())
    return n


def encode(raw):
    out=ctypes.create_string_buffer(lib.ZSTD_compressBound(len(raw)))
    n=checked(lib.ZSTD_compress(out,len(out),raw,len(raw),1))
    return len(raw),out.raw[:n]


def decode(blob,n):
    assert 0<n<=CHUNK
    out=ctypes.create_string_buffer(n)
    actual=checked(lib.ZSTD_decompress(out,n,blob,len(blob)))
    assert actual==n
    return out.raw


def decoded_chunks(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        assert f.read(8)==MAGIC
        while True:
            header=f.read(16)
            assert len(header)==16
            raw_size,packed_size=struct.unpack('<QQ',header)
            if raw_size==packed_size==0:break
            assert 0<raw_size<=CHUNK and packed_size<=lib.ZSTD_compressBound(raw_size)
            blob=f.read(packed_size);assert len(blob)==packed_size
            raw=decode(blob,raw_size);digest.update(raw)
            yield raw
        assert f.read(32)==digest.digest(),'Lossless raw checksum mismatch'
        assert f.read(1)==b''


def pack(source,destination,threads=4):
    assert not destination.exists()
    pending=destination.with_name(destination.name+'.pending')
    assert not pending.exists()
    digest=hashlib.sha256();raw_count=0;count=0;start=time.monotonic()
    with source.open('rb') as src,pending.open('xb') as out,ThreadPoolExecutor(max_workers=threads) as pool:
        out.write(MAGIC)
        queue=[]
        while True:
            raw=src.read(CHUNK)
            if raw:
                digest.update(raw);raw_count+=len(raw);count+=1
                queue.append(pool.submit(encode,raw))
            if len(queue)>=2*threads or (not raw and queue):
                n,blob=queue.pop(0).result()
                out.write(struct.pack('<QQ',n,len(blob)));out.write(blob)
            if not raw and not queue:break
        out.write(struct.pack('<QQ',0,0));out.write(digest.digest());out.flush();os.fsync(out.fileno())
    compressed_s=time.monotonic()-start
    os.rename(pending,destination)
    start=time.monotonic();actual=hashlib.sha256();read=0
    for raw in decoded_chunks(destination):actual.update(raw);read+=len(raw)
    assert actual.digest()==digest.digest() and read==raw_count
    return dict(format='CTFTZ001',raw_bytes=raw_count,raw_sha256=digest.hexdigest(),
        archive_bytes=destination.stat().st_size,compression_s=compressed_s,decode_hash_verify_s=time.monotonic()-start,
        chunks=count,threads=threads,level=1,lossless_byte_identity=True,
        no_original_deleted=True,no_model_or_optimizer_updates=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    result=pack(a.source,a.output/(a.source.name+'.ctftz'))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
