import torch
import numpy as np
import time
from get_points import pcd_from_obj

finger_length = 0.06
finger_width = 0.02
approach_dist = finger_length


def to_grasp17(T, R, ):
  '''
  input:(3600,n_pts)
    n_pts,3
    depth
    width
  output:
    n_pts,interval,17  (score, width, height, dep, rot, cent, objid,)
  '''
  npts = T.shape[0]
  grasp_score = 0

def collision_detection(targets, height, depth  ):
  return

'''
npm = 0
npi = -1
for i in range(88):
  labels = np.load('/data/datasets/graspnet/grasp_label/{}_labels.npz'.format(str(i).zfill(3)))


  points = labels['points'] # object grasp points
  print(points.shape)
  if points.shape[0] > npm:
    npm=points.shape[0]
    npi=i
print(npi)
input(npm)
'''
obj_path = 'object.obj'
i = 0
'''
labels = np.load('/data/datasets/graspnet/grasp_label/{}_labels.npz'.format(str(i).zfill(3)))
points = labels['points']
'''
points =(pcd_from_obj(obj_path))
scene_points = points

R = np.load('batch_view.npy') # 3600,3,3
T = scene_points
scene_points = torch.tensor(scene_points).cuda()
T = scene_points


t0 = time.time()
scene_points = scene_points[np.newaxis,:,:] - T[:,np.newaxis,:]

#gripper config
heights = 0.02
finger_width = 0.02
finger_length = 0.06
#
voxel_size = 0.005
collision_thresh = 1e-4


Widths = [0.02]#[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
Depths = [0.04]#  [0.01, 0.02, 0.03, 0.04]

for widths in Widths:
 for depths in Depths:
  collision_masks = []

  interval = 10 
  for i in range(int(3600/interval+1e-3)):
      R_batch = torch.tensor(R[i*interval:(i+1)*interval,:,:]).unsqueeze(1).repeat(1,points.shape[0],1,1).view(-1,3,3).to(scene_points.device)
      scene_points_batch = scene_points.unsqueeze(0).repeat(interval,1,1,1).view(-1,scene_points.shape[0],3)
    
      targets = torch.bmm(scene_points_batch, R_batch) # n_p * interval, scene_points, 3
#  input(targets.shape)
  ## collision detection
        # height mask
      mask1 = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
          # left finger mask
      mask2 = ((targets[:,:,0] > depths - finger_length) & (targets[:,:,0] < depths))
      mask3 = (targets[:,:,1] > -(widths/2 + finger_width))
      mask4 = (targets[:,:,1] < -widths/2)
      # right finger mask
      mask5 = (targets[:,:,1] < (widths/2 + finger_width))
      mask6 = (targets[:,:,1] > widths/2)
      # bottom mask
      mask7 = ((targets[:,:,0] <= depths - finger_length)\
              & (targets[:,:,0] > depths - finger_length - finger_width))
      # shifting mask
      mask8 = ((targets[:,:,0] <= depths - finger_length - finger_width)\
              & (targets[:,:,0] > depths - finger_length - finger_width - approach_dist))
 
    # get collision mask of each point
      left_mask = (mask1 & mask2 & mask3 & mask4)
      right_mask = (mask1 & mask2 & mask5 & mask6)
      bottom_mask = (mask1 & mask3 & mask5 & mask7)
      shifting_mask = (mask1 & mask3 & mask5 & mask8)
      global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)
     # calculate equivalant volume of each part
      left_right_volume = (heights * finger_length * finger_width / (voxel_size**3))#.reshape(-1)
      bottom_volume = (heights * (widths+2*finger_width) * finger_width / (voxel_size**3))#.reshape(-1)
      shifting_volume = (heights * (widths+2*finger_width) * approach_dist / (voxel_size**3))#.reshape(-1)
      volume = left_right_volume*2 + bottom_volume + shifting_volume
     # get collision iou of each part
      global_iou = global_mask.sum(axis=1) / (volume+1e-6)
     # get collison mask
      collision_mask = (global_iou > collision_thresh)
  #    print(collision_mask.shape)
      collision_masks.append(collision_mask.cpu().reshape(interval,-1))
  collision_masks = torch.cat(collision_masks)
#  print(time.time()-t0)
  print('no collision:',(collision_masks==False).sum())
print(time.time()-t0)
print(collision_masks.shape)
input((collision_masks==True).sum())
#targets = np.matmul
#targets = torch.bmm
