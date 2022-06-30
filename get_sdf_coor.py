import torch
import numpy as np
import pytorch3d.ops.points_normals as pn

def read_sdf(path):
    my_file = open(path, 'r')
    nx, ny, nz = [int(i) for i in my_file.readline().split()]     #dimension of each axis should all be equal for LSH
    ox, oy, oz = [float(i) for i in my_file.readline().split()]   #shape origin
    dims = np.array([nx, ny, nz])
    origin = np.array([ox, oy, oz])
    resolution = float(my_file.readline()) # resolution of the grid cells in original mesh coords
    sdf_data = np.zeros(dims)
    # loop through file, getting each value
    count = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                sdf_data[i][j][k] = float(my_file.readline())
                count += 1
    my_file.close()
    return (sdf_data, origin, resolution)


def get_sdf(sdf_data, coor):
    '''
    input:
      sdf_data: 100,100,100;
      coor: n,3 float, 3d coordinates
    output:
      sdf_val in coor: n,1
    '''
    assert len(coor.shape) == 2, 'coor must be 2d'
    n = coor.shape[0]  # n point 
    xs = [0,0,0,0,1,1,1,1]
    ys = [0,0,1,1,0,0,1,1]
    zs = [0,1,0,1,0,1,0,1]
    # 八个角坐标
    coor_floor = coor.int()  # n,3
    coors = coor_floor.clone()
    coors = coors.unsqueeze(0).repeat(8,1,1) # 8,n,3

    sd = torch.zeros(n).to(coor.device)

#    input(coors.shape)  #8, num_samples * n_pt, 3
    if len(coor.shape) == 2:
      print('matrix')
      for i in range(8):
        coors[i][:,0] += xs[i]
        coors[i][:,1] += ys[i]
        coors[i][:,2] += ys[i]
      #coor_ceil = coor_floor + 1
        dim = sdf_data.shape[0] # dim of sdf, coor belong to [0,dim)
        ptmp = coors[i].long() # n,3
        
        out_mask = ((ptmp < 0) | (ptmp >= dim)).any(1)
        ptmp_clamp = torch.clamp(ptmp,0,dim-1) # avoid illegal access

        v = sdf_data[ptmp_clamp[:,0], ptmp_clamp[:,1], ptmp_clamp[:,2]] # n,
        v[out_mask] = 0 # set outliner value to 0
#        print('v:',v)
#        print(torch.prod(1-torch.abs(coor-ptmp),1))
        sd += (torch.prod(1-torch.abs(coor - ptmp),1) * v)

#    print(sd.sum())
    # 加权八个角坐标上的sdf值
    
    return sd



def create_line_of_action_batch(pts, axis, width, num_samples, sdf_scale):
    '''
    Creates a straight line of action, or list of grid points, from a given point and direction in world or grid coords
    input:
      pts: n,3
      axis: direction, n,3
      width: float, grasp width
      num_samples: int, number of samples along the close line
      sdf_scale: float, sdf specific 
    '''
    n_pts = pts.shape[0]
    num_samples = max(num_samples, 3)  # always at least 3 samples
    intervals = torch.cat([pts + t * axis for t in np.linspace(0, float(width)/2, num=num_samples)]).view(num_samples, n_pts, 3)  # num_samples, pts, 3
#    print(intervals)
#    input(intervals.shape)

    # 简单修改sdf.transform_pt_obj_to_grid即可，返回
    return intervals * sdf_scale  



def estimate_normal(contact_pts, interval, thresh):
    '''
    input:
        contact_pts: points to be estimate normal,
        interval: neighbor range,
        thresh: float, only consider the points within thresh
    return:
      normals: normal vectors in contact points
    '''
    # generate neighbors with interval, always make the contact_pts first, i.e. contact_pts=neighbors[0,:]
    neighbors = torch.rand(27,contact_pts.shape).to(contact_pts.device)

    # get sdf values 
    sdf_neighbor = get_sdf(neighbors.view(-1,3)).view(27,-1).permute(1,0) # n,27
    sdf_neighbor = torch.sort(torch.abs(sdf_neighbor),1)[0] # sort ascend
    sdf_mask = (torch.abs(sdf_neighbor) < thresh) # n,27 
    
    npt_keep = sdf_mask.sum(1).min()
    

    normal = pn.estimate_pointcloud_normals(neighbot_pts, npt_keep)[0]   # (pts,3)
    return normal



def close_finger():
    # first get end_points
    # g1_world, g2_world = self.endpoints

    # left_pt = g1_world # shape (n, 3)
    return 
    
if __name__== '__main__':
  pts = torch.ones(100,3) * 0.01
  npts = pts.shape[0]
  axis = torch.rand(100,3)#torch.tensor([1,0,0]).view(1,3).repeat(10,1)
  wid = 0.1
  num_samples = 1000
  sdf_reso = 0.0017
  loas = create_line_of_action_batch(pts, axis, wid, num_samples, 1.0/sdf_reso) # num_samples, pts, 3  
  
  #tmp sdf_data
  sdf_data = torch.ones(100,100,100)
  sdfs = get_sdf(sdf_data, loas.view(-1,3)).view(num_samples,-1)  # num_samples, pts, 
#  input(sdfs)

  # 待确认阈值
  contact_thresh = 0.01
  sdfs_thresh_mask = (sdfs > contact_thresh).all(0) # num_samples, npts -> npts,

  # 求 sdfs>0且最小的 index，通过loas取出对应点(sdf>0 or <0?)
  sdfs_neg_mask = (sdfs < 0) # num_samples, npts 
  sdfs_copy = sdfs.clone()
  sdfs_copy[sdfs_neg_mask] -= 100 # avoid sdf<0
  index_closest = torch.argmax(torch.abs(sdfs_copy), 0) # npts
  index_npts = torch.arange(npts)
  contact_points = loas[index_closest,index_npts,:]  # npts, 3 

#  contact_points[sdfs_thresh_mask,:] = -1  # set the grasp without contact points,(assume always have contact points) 
#  input(contact_points.shape)


    
  # 找3d邻域的点，满足sdfs < contact_thresh, (pts, num_neighbor=15, 3),保证第一个点是要求法向的点
  interval = 1.5
  neighbor_pts = find_neighbors_3d(contact_points, interval)
  normal = pn.estimate_pointcloud_normals(neighbot_pts, 14)[0]   # (pts,3)
  
  
  
  

