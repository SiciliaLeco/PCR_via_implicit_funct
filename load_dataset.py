import numpy
import torch.utils.data
import os
import glob
import copy
import six
import numpy as np
import torch
import torch.utils.data
import torchvision

import se_math.se3 as se3
import se_math.so3 as so3
import se_math.mesh as mesh
import se_math.transforms as transforms

def get_categories(categoryfile):
    cinfo = None
    if categoryfile:
        categories = [line.rstrip('\n') for line in open(categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
    return cinfo

def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the indexes from given class names
def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []

    # loop all the folderName (class name) to find the class in class_to_idx
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        # check if it is the class we want
        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue
        # to find the all point cloud paths in the class folder
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
    return samples

class PointCloudDataset(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """

    def __init__(self, rootdir, pattern, fileloader, transform=None, classinfo=None):
        super().__init__()

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        # find all the class names
        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        # get all the 3D point cloud paths for the class of class_to_idx
        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        :param index:
        :return:
        """
        path, target = self.samples[index]
        sample = self.fileloader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def split(self, rate):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = rate * len(dataset),
            len(dataset2) = (1-rate) * len(dataset)
        """
        orig_size = len(self)
        select = np.zeros(orig_size, dtype=int)
        csize = np.zeros(len(self.classes), dtype=int)

        for i in range(orig_size):
            _, target = self.samples[i]
            csize[target] += 1
        dsize = (csize * rate).astype(int)
        for i in range(orig_size):
            _, target = self.samples[i]
            if dsize[target] > 0:
                select[i] = 1
                dsize[target] -= 1

        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        samples1 = list(map(lambda i: dataset1.samples[i], np.where(select == 1)[0]))
        samples2 = list(map(lambda i: dataset2.samples[i], np.where(select == 0)[0]))

        dataset1.samples = samples1
        dataset2.samples = samples2
        return dataset1, dataset2


class ModelNet(PointCloudDataset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """

    def __init__(self, dataset_path, train=1, transform=None, classinfo=None, is_uniform_sampling=False):
        # if you would like to uniformly sampled points from mesh, use this function below
        if is_uniform_sampling:
            loader = mesh.offread_uniformed # used uniformly sampled points.
        else:
            loader = mesh.offread # use the original vertex in the mesh file
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class TransformedDataset(torch.utils.data.Dataset):

    def get_size_of_dir(self):
        path_file_number=glob.glob(pathname=os.path.join(self.path, '[0-9]*'))
        return len(path_file_number)

    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None, train="train"):
        self.dataset = dataset
        self.rigid_transform = rigid_transform # torchvision里面对输入数据预处理的transform部分
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

        # ndf params
        self.ndf_info_path = os.path.join("/data/xlli/code/rendf/dataset", train)
        self.res = 256
        self.num_sample_points = 50000
        self.pointcloud_samples = 10000
        self.sample_sigmas = [0.08]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        index = "000" + str(index + 1)
        ndf_path = os.path.join(self.ndf_info_path, index[-4:])
        input_path = ndf_path
        samples_path = ndf_path

        voxel_path = input_path + '/voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)
        occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
        input = np.reshape(occupancies, (self.res,) * 3)

        points = []
        coords = []
        df = []


        for i in range(len(self.sample_sigmas)):
            boundary_samples_path = samples_path + '/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_df = boundary_samples_npz['df']
            # subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points)
            coords.extend(boundary_sample_coords)
            df.extend(boundary_sample_df)
        # return {'grid_coords':np.array(coords, dtype=np.float32),'df': np.array(df, dtype=np.float32),'points':np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}

        # coords = np.array(coords, dtype=np.float32)
        # df = np.array(df, dtype=np.float32)
        # points = np.array(points, dtype=np.float32) # 猜测我这种写法里面points和p0应该是一样的内容
        # inputs = np.array(input, dtype=np.float32)

        points = torch.Tensor(points)
        inputs = torch.Tensor(input)
        df = torch.Tensor(df)
        coords = torch.Tensor(coords)
        if self.source_modifier is not None:
            p_ = self.source_modifier(points)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(points)
        igt = self.rigid_transform.igt

        if self.template_modifier is not None:
            p0 = self.template_modifier(points)
        else:
            p0 = points

        # p0: template, p1: source, igt: transform matrix from p1 to p0
        # return {"p0":p0, "p1":p1, "igt":igt, "coords":coords, "df":df, "inputs":inputs} # return type: tuple
        return p0, p1, igt, coords, df, inputs


# sample = trainset[0]
# print("p0 shape:", sample["p0"].shape) # ([100000, 3])
# print("p1 shape:", sample["p1"].shape) # ([100000, 3])
# print("igt shape:", sample["igt"].shape) # [4, 4]
# print("coords shape:", sample["coords"].shape) # ([100000, 3])
# print("df shape:", sample["df"].shape) # ([100000])
# print("input shape:", sample["inputs"].shape) # ([256, 256, 256])
#



