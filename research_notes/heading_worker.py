def launch(connection,gpu,index,output):
 from heading_env import install
 install()
 from clean_worker import launch as original
 return original(connection,gpu,index,output)
