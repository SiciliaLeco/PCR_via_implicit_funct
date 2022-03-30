import models.local_model as model
import torch
import configs.config_loader as cfg_loader
import torch.optim as optim
import torchvision
from load_dataset import get_categories, ModelNet, TransformedDataset
import se_math.transforms as transforms

cfg = cfg_loader.get_config()

def get_dataloader():
    cinfo = get_categories("/data/xlli/code/fmr/data/categories/modelnet40_test.txt")
    # cinfo = get_categories("data/categories/modelnet40_half1.txt")
    transform = torchvision.transforms.Compose([ \
        transforms.Mesh2Points(), \
        transforms.OnUnitCube(), \
        # transforms.Resampler(10000), \
    ])

    dataset_path = "/data/xlli/code/fmr/data/ModelNet40"
    # dataset_path = "data/ModelNet40"
    net = ModelNet(dataset_path, train=1, transform=transform, classinfo=cinfo, is_uniform_sampling=False)
    trainset = TransformedDataset(net, transforms.RandomTransformSE3(0.8, True))
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=10, shuffle=True, num_workers=0)
    return trainloader
