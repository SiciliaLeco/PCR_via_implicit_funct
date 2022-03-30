import models.local_model as model
import voxelized_data_shapenet as voxelized_data
import torch
import configs.config_loader as cfg_loader
import torch.optim as optim

cfg = cfg_loader.get_config()
enc = model.Encoder()
dec = model.Decoder()
net = model.Seq2Seq(enc, dec, 'cpu')

def compute_loss(batch, device, net):

    p = batch.get('grid_coords').to(device)
    df_gt = batch.get('df').to(device) #(Batch,num_points)
    inputs = batch.get('inputs').to(device)


    df_pred = net(p,inputs) #(Batch,num_points)

    loss_i = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred, max=0.1),torch.clamp(df_gt, max=0.1))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
    loss = loss_i.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

    return loss

train_dataset = voxelized_data.VoxelizedDataset('train',
                                                res=cfg.input_res,
                                                pointcloud_samples=cfg.num_points,
                                                data_path=cfg.data_dir,
                                                split_file=cfg.split_file,
                                                batch_size=1,
                                                num_sample_points=cfg.num_sample_points_training,
                                                num_workers=0,
                                                sample_distribution=cfg.sample_ratio,
                                                sample_sigmas=cfg.sample_std_dev)

test_dataset = voxelized_data.VoxelizedDataset('test',
                                              res=cfg.input_res,
                                              pointcloud_samples=cfg.num_points,
                                              data_path=cfg.data_dir,
                                              split_file=cfg.split_file,
                                              batch_size=1,
                                              num_sample_points=cfg.num_sample_points_training,
                                              num_workers=0,
                                              sample_distribution=cfg.sample_ratio,
                                              sample_sigmas=cfg.sample_std_dev)

train_dataloader = train_dataset.get_loader()
test_dataloader  = test_dataset.get_loader()

optimizer = optim.Adam(net.parameters(), lr= 0.001)
criterion = torch.nn.L1Loss()

for epoch in range(30):
    print(epoch)
    sum_loss = 0
    net.train()
    for batch in train_dataloader:
        print(1)
        optimizer.zero_grad()
        p = batch.get('grid_coords')
        df_gt = batch.get('df')#(Batch,num_points)
        inputs = batch.get('inputs')

        df_pred = net(p, inputs) #(Batch,num_points)

        loss_i = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred, max=0.1),torch.clamp(df_gt, max=0.1))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        loss = loss_i.sum(-1).mean()# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        loss.backward()
        optimizer.step()

        sum_loss += loss
    print(sum_loss)