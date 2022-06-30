import open3d as o3d
import numpy as np

def pcd_from_obj(obj_path,pts_keep=2000):
  # get obj mesh
  textured_mesh = o3d.io.read_triangle_mesh(obj_path)

  textured_mesh.compute_vertex_normals()
 # o3d.visualization.draw_geometries([textured_mesh])

  # vertex
  pcobj = o3d.geometry.PointCloud()
  pcobj.points = o3d.utility.Vector3dVector(textured_mesh.vertices)

  scale = int(len(pcobj.points) / pts_keep)
  pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcobj, scale) # keep 1/10 points

  o3d.visualization.draw_geometries([pcd_new])

  textured_pc = np.array(pcd_new.points,dtype=np.float32)
  print('points kept:',textured_pc.shape)
  return textured_pc

if __name__ == '__main__':
    obj_path = 'object.obj'#'/data/datasets/graspnet/models/000/textured.obj'
    pcd_from_obj(obj_path)
