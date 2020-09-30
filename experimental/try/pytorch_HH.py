# -*- coding: utf-8 -*-

import torch


dt = 0.01


class HH:
    def __init__(self, num, E_Na=50., g_Na=120., E_K=-77.,
                 g_K=36., E_Leak=-54.387, g_Leak=0.03,
                 C=1.0, Vr=-65., Vth=20., I=0.):
        self.num = num
        self.E_Na = E_Na
        self.g_Na = g_Na
        self.E_K = E_K
        self.g_K = g_K
        self.E_Leak = E_Leak
        self.g_Leak = g_Leak
        self.C = C
        self.Vr = Vr
        self.Vth = Vth
        self.I = I

        self.V = torch.ones(num) * Vr
        self.m = torch.zeros(num)
        self.h = torch.zeros(num)
        self.n = torch.zeros(num)

    def update(self, t):
        # m
        alpha = 0.1 * (self.V + 40) / (1 - torch.exp(-(self.V + 40) / 10))
        beta = 4.0 * torch.exp(-(self.V + 65) / 18)
        dmdt = alpha * (1 - self.m) - beta * self.m
        self.m = self.m + dmdt * dt

        # h
        alpha = 0.07 * torch.exp(-(self.V + 65) / 20.)
        beta = 1 / (1 + torch.exp(-(self.V + 35) / 10))
        dhdt = alpha * (1 - self.h) - beta * self.h
        self.h = self.h + dhdt * dt

        # n
        alpha = 0.01 * (self.V + 55) / (1 - torch.exp(-(self.V + 55) / 10))
        beta = 0.125 * torch.exp(-(self.V + 65) / 80)
        dndt = alpha * (1 - self.n) - beta * self.n
        self.n = self.n + dndt * dt

        # V
        INa = self.g_Na * self.m ** 3 * self.h * (self.V - self.E_Na)
        IK = self.g_K * self.n ** 4 * (self.V - self.E_K)
        IL = self.g_Leak * (self.V - self.E_Leak)
        dvdt = (- INa - IK - IL + self.I) / self.C
        self.V += dvdt * dt


hh = HH(2000)

for t in torch.arange(0., 100., dt):
    hh.update(t)
