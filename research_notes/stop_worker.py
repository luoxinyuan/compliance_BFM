"""Original process-isolated sampler with physical stop task distribution."""
def launch(connection,gpu,index,output):
 from stop_env import install
 install()
 from clean_worker import launch as original
 return original(connection,gpu,index,output)
