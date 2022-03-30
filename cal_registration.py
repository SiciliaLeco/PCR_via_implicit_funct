from models.local_model import Encoder, Decoder
import torch
import numpy as np
from random import sample

import se_math.se3 as se3
import se_math.invmat as invmat


# B: batch_size N: num_points
class SolveRegistration(torch.nn.Module):
    def __init__(self, mode=None, isTest=False):
        super().__init__()
        # self.ndf_model = model
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
        self.isTest = isTest  # whether it is testing

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
        pj = 1 / 3 * p1.sum(dim=2)  # B x N
        distance = pj - ndf
        return distance  # B x N

    def approx_Jac(self, r, dt):
        r_ = r.unsqueeze(-1)
        dt_ = dt.unsqueeze(-2)
        return r_ / dt_  # [B x N x 6]

    def IC_algo(self, g0, ndf, p0, p1, maxiter, xtol, isTest=False):
        # TODO: testing part
        batch_size = p0.size(0)
        self.last_err = None
        g = g0
        self.g_series = torch.zeros(maxiter + 1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        # task 2
        dist = self.calculate_dist(ndf, p1)
        dt = self.dt.to(p0).expand(self.batch_size, 6)  # convert to the type of p0. [B, 6] #预备delta t，也就是变换量
        J = self.approx_Jac(dist, dt)  # [B x N x 6]
        try:
            Jt = J.transpose(1, 2)  # [B, 6, K]
            H = Jt.bmm(J)  # [B, 6, 6]
            B = self.inverse(H)
            pinv = B.bmm(Jt)  # [B, 6, K]
        except RuntimeError as err:
            # singular...?
            self.last_err = err
            r = dist
            return r, g

        itr = 0
        r = None
        for iter in range(maxiter):
            p = self.transform(g.unsqueeze(1), p1)  # B x N x 3
            dist = self.calculate_dist(ndf, p)  # B x N
            dx = -pinv.bmm(dist.unsqueeze(-1)).view(batch_size, 6)
            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0  # no update
                break
            g = self.update_T(g, dx)
            self.g_series[itr + 1] = g.clone()
            self.prev_r = r

        return r, g

    def estimate_T(self, p0, p1, ndf, maxiter=5, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        # TODO: 将df的来源改为ndf的训练结果
        """
        :param maxiter: maximum iteration
        :param xtol: a threshold for early stop of transformation estimation
        :param p0_zero_mean: True: normanize p0 before IC algorithm
        :param p1_zero_mean: True: normanize p1 before IC algorithm
        """
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0)  # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1)  # [B, 4, 4]
        # normalization
        if p0_zero_mean:
            p0_m = p0.mean(dim=1)  # [B, N, 3] -> [B, 3]
            a0 = a0.clone()
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)  # [B, N, 3]
        else:
            q0 = p0
        if p1_zero_mean:
            p1_m = p1.mean(dim=1)  # [B, N, 3] -> [B, 3]
            a1 = a1.clone()
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        isTest = False
        g0 = torch.eye(4).to(q0).view(1, 4, 4).expand(q0.size(0), 4, 4).contiguous()  # B x 4 x 4
        # ndf = self.ndf_model(coords, inputs)
        r, g = self.IC_algo(g0, ndf, q0, q1, maxiter, xtol, isTest)
        self.g = g

        # renormalization
        if p0_zero_mean or p1_zero_mean:
            # output' = trans(p0_m) * output * trans(-p1_m)
            #        = [I, p0_m;] * [R, t;] * [I, -p1_m;]
            #          [0, 1    ]   [0, 1 ]   [0,  1    ]
            est_g = self.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            self.g = est_g

            est_gs = self.g_series  # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            self.g_series = est_gs

        return r

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, I, reduction='mean') * 16

    @staticmethod
    def rsq(r):
        # |r| should be 0
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, reduction='sum')
