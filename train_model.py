import torch
import numpy as np
from random import sample
from cal_registration import SolveRegistration
from load_dataset import TransformedDataset
from dataloader import get_dataloader

class FMRTrain:
    def __init__(self, dim_k, num_points, train_type):
        self.dim_k = dim_k
        self.num_points = num_points # 以上两个变量未被用到
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = train_type  # 0: unsupervised, 1: semi-supervised see. self.compute_loss()

    def create_model(self):
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(isTest=False)
        return fmr_solver

    def compute_loss(self, solver, data, device):
        p0, p1, igt, _, ndf, _ = data # TODO: fix input type
        p0 = p0.to(device)  # template
        p1 = p1.to(device)  # source
        igt = igt.to(device)  # igt: p0 -> p1
        # estimate_T(self, p0, p1, ndf, maxiter=5, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        r = solver.estimate_T(p0, p1, ndf, self.max_iter)
        loss_r = solver.rsq(r)
        est_g = solver.g
        loss_g = solver.comp(est_g, igt)

        # semi-supervised learning, set max_iter>0
        if self._loss_type == 0:
            loss = loss_r
        elif self._loss_type == 1:
            loss = loss_r + loss_g
        elif self._loss_type == 2:
            loss = loss_r + loss_g
        else:
            loss = loss_g
        return loss

    def train(self, model, trainloader, optimizer, device):
        model.train()
        total_loss = 0
        count = 0
        for i, data in enumerate(trainloader):
            loss = self.compute_loss(model, data, device)
            loss = loss.requires_grad_()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            count += 1
        ave_loss = float(total_loss) / count
        print(ave_loss)
        return ave_loss

    #TODO: write validate and test part code
    # def validate(self, model, testloader, device):
    #     model.eval()
    #     vloss = 0.0
    #     count = 0
    #     with torch.no_grad():
    #         for i, data in enumerate(testloader):
    #             loss_net = self.compute_loss(model, data, device)
    #             vloss += loss_net.item()
    #             count += 1
    #
    #     ave_vloss = float(vloss) / count
    #     return ave_vloss

train_helper = FMRTrain(dim_k=1024, num_points=2048, train_type=1)
model = train_helper.create_model()
dataloader = get_dataloader()

learnable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(learnable_params)


for epoch in range(1):
    print(epoch)
    # running_loss = train_helper.train(model, dataloader, optimizer, "cpu")
    # print("epoch {}, los={}".format(epoch, running_loss))

