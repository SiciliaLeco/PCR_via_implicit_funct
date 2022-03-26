import shutil

from boundary_sampling import boundary_sampling
from convert_to_scaled_off import to_off
import voxelized_pointcloud_sampling as vps
from glob import glob
import os
import configs.config_loader as cfg_loader

cfg = cfg_loader.get_config()

sample_std_dev = [0.08]

def get_file_name(file_dir, ext=".off"):
    List = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".off":
                List.append(os.path.join(root, file))
    return List

# 由于ModelNet40数据集中都是off文件，所以不需要再从obj转off的过程，因此也不需要做to_off的部分
# 由于ModelNet40数据集中已经分好了train和test部分，因此这里不需要再做

'''
计划的存储格式：
---airplane
----|-train 
----|-|-1
----|---|- airplane_0001.off
----|---|- boundary_{}_samples.npz
----|---|- voxelized_point_cloud_{}res_{}points.npz
...............
----|-test
...............
'''

print('Start distance field sampling.')

root = os.getcwd()
img_list = sorted(get_file_name(os.path.join(root, "dataset/train")))
vps.init()

for sigma in sample_std_dev:
    print('Start distance field sampling with sigma: {}.'.format(sigma))
    #multiprocess(partial(boundary_sampling, sigma = sigma))
    # this process is multi-processed for each path: IGL parallelizes the distance field computation of multiple points.

    for path in img_list:
        index = path.split("/")[-1][-8:-4]
        print(index)
        new_dir = root + "/dataset/train/" + index
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        print(new_dir)
        print(new_dir + "/" + path.split("/")[-1])
        if os.path.exists(path):
            shutil.copy(path, new_dir + "/" + path.split("/")[-1])
        off_file_name = new_dir + "/" + path.split("/")[-1]
        boundary_sampling(off_file_name, sigma)

for path in img_list:
    index = path.split("/")[-1][-8:-4]
    print(index)
    new_dir = root + "/dataset/train/" + index
    off_file_name = new_dir + "/" + path.split("/")[-1]
    vps.voxelized_pointcloud_sampling(off_file_name)

