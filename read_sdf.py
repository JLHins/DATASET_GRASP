import numpy as np


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


if __name__ == '__main__':
   sdf_data, origin, resolution = read_sdf('object.sdf')
   print(sdf_data.shape,origin,resolution)
