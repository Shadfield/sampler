import torch
import torch.nn as nn


class FRNorm(nn.Module):
    """
    Filter response normalisation
        input shape: B*
        output shape: B*
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(0.))
        self.tau = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        msn = torch.sum(x ** 2) / x.size(0)
        norm = x * torch.rsqrt(msn + 1e-16)
        norm = (self.alpha * norm) + self.beta
        tlu = torch.max(norm, self.tau)
        return tlu


class DFEncBlock(nn.Module):
    def __init__(self, chn_in, chn_out, bottleneck=False):
        super().__init__()
        self.inconv = nn.Conv2d(chn_in, chn_out, 1)
        self.activ = nn.ELU()
        self.conv1 = nn.Conv2d(chn_in, chn_out, 3, padding=1)
        self.conv2 = nn.Conv2d(chn_out, chn_out, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.bottleneck = bottleneck

    def forward(self, x):
        res = self.inconv(x)
        c = self.activ(self.conv1(x))
        c = self.activ(self.conv2(c))
        return c + res if self.bottleneck else self.pool(c + res)


class DFDecBlock(nn.Module):
    def __init__(self, chn_in, chn_out, activ=nn.ELU()):
        super().__init__()
        self.activ = activ
        self.inconv = nn.Conv2d(chn_in, chn_out, 1)
        self.conv1 = nn.ConvTranspose2d(chn_in, chn_out, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(chn_out, chn_out, 3, padding=1)
        self.norm = FRNorm()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        residual = self.inconv(x)
        c = self.conv1(x)
        c = self.norm(c)
        c = self.activ(self.conv2(c))

        return self.upsample(c + residual)
