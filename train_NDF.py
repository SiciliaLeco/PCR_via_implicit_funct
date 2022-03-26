import models.local_model as model
import torch
import configs.config_loader as cfg_loader
import torch.optim as optim
import torchvision
from load_dataset import get_categories, ModelNet, TransformedDataset
import se_math.transforms as transforms

cfg = cfg_loader.get_config()
enc = model.Encoder()
dec = model.Decoder()
network = model.Seq2Seq(enc, dec, 'cpu')


def compute_loss(batch, device, net):
    p = batch.get('grid_coords').to(device)
    df_gt = batch.get('df').to(device)  # (Batch,num_points)
    inputs = batch.get('inputs').to(device)

    df_pred = net(p, inputs)  # (Batch,num_points)

    loss_i = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred, max=0.1), torch.clamp(df_gt,
                                                                                          max=0.1))  # out = (B,num_points) by componentwise comparing vecots of size num_samples:
    loss = loss_i.sum(
        -1).mean()  # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

    return loss


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
optimizer = optim.Adam(net.parameters(), lr= 0.001)
criterion = torch.nn.L1Loss()
epochs = 1

for epoch in range(epochs):
    for i, data in enumerate(trainloader):
        p0, p1, igt, coords, df, inputs = data
        print(p0)
        # df_pred = network(p, inputs)  # (Batch,num_points)
        #
        # loss_i = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred, max=0.1),torch.clamp(df_gt, max=0.1))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # loss = loss_i.sum(-1).mean()# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # loss.backward()
        # optimizer.step()

# torch.save(network, '\model.pth')