import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SubPixelConv(nn.Module):
    def __init__(self, chn_in, scale_factor):
        super().__init__()
        chn_out = chn_in * (scale_factor ** 2)
        self.conv = nn.Conv2d(chn_in, chn_out, kernel_size=3, groups=chn_in, padding=1)
        self.shuf = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        return self.shuf(self.conv(x))


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


class Reconst(nn.Module):
    def __init__(self, chn=3):
        super().__init__()

        # Channels
        enc_chn = [chn, 32, 64, 128, 128, 128]
        dec_in = [256, 256, 192, 128]
        dec_out = [128, 128, 96, 64]

        # Decoder
        self.dec_blocks = nn.ModuleList()
        for i, o in zip(dec_in, dec_out):
            self.dec_blocks.append(DFDecBlock(i, o))
        self.lastconv = nn.ConvTranspose2d(dec_out[-1], 3, 1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        for i, o in zip(enc_chn[:-1], enc_chn[1:]):
            self.enc_blocks.append(DFEncBlock(i, o))
        self.enc_blocks[-1].bottleneck = True

    def forward(self, x):
        insize = x.shape
        skips = []

        # Encoder
        for mod in self.enc_blocks:
            x = mod(x)
            skips.append(x)

        # Decoder
        skips = skips[:-1]
        for mod, s in zip(self.dec_blocks, skips[::-1]):
            x = torch.cat((x, F.interpolate(s, size=x.shape[2:])), dim=1)
            x = mod(x)
        out_img = self.lastconv(x)

        out_img = F.interpolate(out_img, size=insize[2:], mode='bilinear')
        return (out_img.tanh() + 1) * 0.5
