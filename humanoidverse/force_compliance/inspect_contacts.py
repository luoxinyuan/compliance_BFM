"""Render the configured contact catalog and verify local points against robot meshes."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from .contacts import CONTACTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/force_compliance/contact_catalog'))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).parents[1]/'data/robots/g1/scene_29dof_freebase_mujoco.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    points=[]
    for index, contact in enumerate(CONTACTS):
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,contact['body'])
        point=data.xpos[body]+data.xmat[body].reshape(3,3)@np.array(contact['point'])
        vertices=[]
        bodies=[body]+[i for i in range(model.nbody) if model.body_parentid[i]==body and model.body_jntnum[i]==0]
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] not in bodies or model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh=model.geom_dataid[geom]
            verts=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
            vertices.append(verts@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom])
        distance=float(np.linalg.norm(np.concatenate(vertices)-point,axis=1).min())
        if distance>.001:
            raise ValueError(f"Contact {contact['name']} is not within 1mm of the mesh")
        points.append(dict(index=index,**contact,world_point=point.tolist(),nearest_vertex_distance=distance))
    (args.output/'contacts.json').write_text(json.dumps(points,indent=2))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    renderer=mujoco.Renderer(model,height=720,width=960)
    try:
        for azimuth in (140,40):
            camera=mujoco.MjvCamera()
            camera.lookat[:]=[0,0,.75]
            camera.distance=2.5;camera.azimuth=azimuth;camera.elevation=-10
            renderer.update_scene(data,camera=camera)
            for contact in points:
                geom=renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012]*3),
                                   np.array(contact['world_point']),np.eye(3).ravel(),np.array([1,.15,.05,1],np.float32))
                geom.label=str(contact['index'])
                renderer.scene.ngeom+=1
            Image.fromarray(renderer.render()).save(args.output/f'contacts_{azimuth}.png')
    finally:
        renderer.close()
    print(json.dumps(points,indent=2))


if __name__=='__main__':
    main()
