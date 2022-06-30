import torch
import time
import pytorch3d.ops.points_normals as pn

a = torch.rand(1000000,15,3)

time1 = time.time()

pn.estimate_pointcloud_normals(a,14)

time2 = time.time()
'''
for i in a:
    pn.estimate_pointcloud_normals(i.unsqueeze(0),14)
time3 = time.time()
'''
print(time2-time1) # 2.3
print(time3-time2) # 25
