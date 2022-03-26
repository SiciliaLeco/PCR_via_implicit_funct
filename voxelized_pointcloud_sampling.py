from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
import os
import traceback

# kdtree, grid_points, cfg = None, None, None
def voxelized_pointcloud_sampling(path):
    try:
        input_res = 256
        num_points = 10000

        out_path = os.path.dirname(path)

        out_file = out_path + '/voxelized_point_cloud_{}res_{}points.npz'.format(input_res, num_points)
        input_file = path
        if os.path.exists(out_file):
            print(f'Exists: {out_file}')
            return

        mesh = trimesh.load(input_file)
        point_cloud = mesh.sample(num_points)

        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)

        np.savez(out_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies, bb_min = -0.5, bb_max = 0.5, res = 256)
        print('Finished: {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

def init():
    global kdtree, grid_points
    grid_points = create_grid_points_from_bounds(minimun=-0.5, maximum=0.5, res=256)
    kdtree = KDTree(grid_points)

def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list
