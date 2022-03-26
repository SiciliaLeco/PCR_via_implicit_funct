from models.local_model import Encoder, Decoder
import torch
import numpy as np
from random import sample

import se_math.se3 as se3
import se_math.invmat as invmat


# B: batch_size N: num_points
class SolveRegistration(torch.nn.Module):
    def __init__(self, model, isTest=False):
        super().__init__()
        self.ndf_model = model
        # functions for registration calculate
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp  # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        # initialization for dt: [w1, w2, w3, v1, v2, v3], 3 rotation angles and 3 translation
        delta = 1.0e-2  # step size for approx. Jacobian (default: 1.0e-2)
        dt_initial = torch.autograd.Variable(torch.Tensor([delta, delta, delta, delta, delta, delta]))
        self.dt = torch.nn.Parameter(dt_initial.view(1, 6), requires_grad=True)

        # results
        self.last_err = None
        self.g_series = None  # for debug purpose
        self.prev_r = None
        self.g = None  # estimated transformation T
        self.isTest = isTest # whether it is testing

        self.batch_size = 10

    def update_T(self, g, dx):
        dg = self.exp(dx)
        return dg.matmul(g)

    def calculate_dist(self, ndf, p1):
        """
        :param ndf [B x N]
        :param p1 [B x N x 3]
        return: distance ndf - p1 [B x N]
        """
        pj = 1/3 * p1.sum(dim=2) # B x N
        distance = pj - ndf
        return distance # B x N

    def approx_Jac(self, r, dt):
        r_ = r.unsqueeze(-1)
        dt_ = dt.unsqueeze(-2)
        return r_ / dt_ # [B x N x 6]


    def IC_algo(self, g0, ndf, p0, p1, maxiter, xtol, isTest = False):
        dist = self.calculate_dist(ndf, p1)
        dt = self.dt.to(p0).expand(self.batch_size, 6)  # convert to the type of p0. [B, 6] #预备delta t，也就是变换量
        J = self.approx_Jac(dist, dt) # [B x N x 6]
        Jt = J.transpose(1, 2) # [B x 6 x N]
        H = Jt.bmm(J) # [B x 6 x 6]
        B = self.inverse(H)
        pinv = B.bmm(Jt) # [B x 6 x N]

        itr = 0
        r = None
        for iter in range(maxiter):
            


    def estimate_T(self, p0, p1, coords, inputs, maxiter=5, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        # TODO: 将df的来源改为ndf的训练结果
        """
        :param maxiter: maximum iteration
        :param xtol: a threshold for early stop of transformation estimation
        :param p0_zero_mean: True: normanize p0 before IC algorithm
        :param p1_zero_mean: True: normanize p1 before IC algorithm
        """
        isTest = False
        g0 = torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()
        ndf = self.ndf_model(coords, inputs)
        self.IC_algo(g0, ndf, p1, maxiter, xtol, isTest)

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, I, reduction='mean') * 16



