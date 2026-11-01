"""Propagate the existing Linux abstract-socket loader fix into spawned children."""
import os,sys
if sys.platform=='linux':
    import multiprocessing.connection as connection
    if not getattr(connection.arbitrary_address,'_m11_abstract',False):
        original=connection.arbitrary_address
        def arbitrary_address(family):
            return b'\0bfm-m11-'+os.urandom(12).hex().encode() if family=='AF_UNIX' else original(family)
        arbitrary_address._m11_abstract=True
        connection.arbitrary_address=arbitrary_address
