
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.enhance import normalize_min_max
from kornia.filters import sobel
from kornia.morphology import closing
from modules.reconstructors import DFDecBlock, DFEncBlock

Result = namedtuple('Result', ['reconst', 'task', 'hmap', 'smap'])


@torch.jit.script
def binariser(tensor: torch.Tensor, threshold: float):
    """
    Transforms heatmap into a binary mask in a differentiable manner
    """
    forward = tensor.clone()
    forward[tensor >= threshold] = 1
    forward[tensor < threshold] = 0
    return tensor + (forward - tensor).detach()


@torch.jit.script
def distance_to_probability(dist):
    return 1 - torch.exp(-dist / dist.var())


class SamplerBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('droprate', torch.tensor(0.0))

    def forward(self, img):
        sample_map = torch.ones_like(img)[:, 0, None]
        return sample_map, sample_map


class Downsample(SamplerBase):
    def forward(self, img):

        sample_rate = 1 - self.droprate
        num_elem = img[0, 0].numel()
        sample_every = num_elem / (sample_rate * num_elem)

        if self.droprate >= 0.5:
            sample_every = sample_every.round().to(torch.int)
            sample_every = num_elem if sample_every > num_elem else sample_every
            mask = torch.zeros(sample_every, dtype=torch.float, device=img.device)
            mask[-1] = 1
            num_repeats = num_elem // mask.size(0)
            end_pad_size = num_elem - (num_repeats * sample_every)
            mask = mask.repeat(num_repeats)
            mask = torch.cat([mask, torch.zeros(end_pad_size, device=img.device)])
        else:
            mask = torch.ones_like(img[0, 0]).flatten()
            additional_drop = (1 / (sample_every - 1)).round().to(torch.int)
            if additional_drop > 0:
                mask[1::2][::additional_drop] = 0

        mask = mask.reshape(1, 1, *img.shape[2:]).to(torch.float)
        return mask, mask


class Random(SamplerBase):
    def forward(self, img):
        heat_map = torch.rand_like(img)[:, 0, None]
        sample_map = binariser(heat_map, self.droprate)
        return heat_map, sample_map


class SAUCE(SamplerBase):
    def __init__(self):
        super().__init__()
        self.pad = nn.ConstantPad2d((1, 0), 0.)
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(1.))
        self.gamma = nn.Parameter(torch.tensor(0.))

    def forward(self, img):

        # Brightness
        lin = img.flatten(start_dim=2)
        d_inten = torch.abs(lin[..., 1:] - lin[..., :-1])
        d_inten = self.pad(d_inten)
        d_inten = torch.norm(d_inten, p=2, dim=1)
        d_inten = d_inten.reshape(img.size(0), 1, img.size(2), img.size(3))

        # Distance
        delta = torch.zeros_like(d_inten)
        delta[:, :, 0] = 1

        # Combine and convert to probability
        d_p = self.beta * delta
        i_p = self.alpha * d_inten
        dist = F.relu(d_p + i_p + self.gamma)
        probs = distance_to_probability(dist)

        # Normalise to (0,1)
        heatmap = normalize_min_max(probs)
        sample_map = binariser(heatmap, self.droprate)
        return heatmap, sample_map


class DeepSAUCE(SamplerBase):
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
        self.lastconv = nn.ConvTranspose2d(dec_out[-1], 1, 1)

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
        heatmap = self.lastconv(x)

        # Normalise
        heatmap = F.interpolate(heatmap, size=insize[2:], mode='bilinear')
        heatmap = normalize_min_max(heatmap)
        sample_map = binariser(heatmap, self.droprate)
        return heatmap, sample_map


class LevelCross(SamplerBase):
    def __init__(self):
        super().__init__()
        self.pad = nn.ConstantPad2d((1, 0), 0)
        self.scale = 256

    def forward(self, img):

        # Quantise
        quant = (img * self.scale).to(torch.int)

        # Change in intesity
        lin = quant.flatten(start_dim=2)
        d_inten = torch.abs(lin[..., 1:] - lin[..., :-1])
        d_inten = d_inten.sum(dim=1, keepdim=True)
        hmap = self.pad(d_inten).to(torch.float)

        # Reshape and select
        hmap = hmap.reshape(img.size(0), 1, *img.shape[2:])
        heat_map = normalize_min_max(hmap)
        hmap_max = hmap.amax(dim=(-1, -2), keepdim=True)
        sample_map = torch.zeros_like(hmap)
        sample_map[hmap > self.droprate * hmap_max] = 1

        return heat_map, sample_map


class MixedAdaptiveRandom(SamplerBase):
    """from: High-quality Image Restoration from Partial Mixed Adaptive-Random Measurements"""

    def __init__(self):
        super().__init__()
        self.register_buffer('morph_kernel', torch.ones(3, 3))

    def forward(self, img):
        edge = sobel(img).sum(dim=1, keepdim=True)
        hard_edge = binariser(edge, self.droprate)
        morph = closing(hard_edge, self.morph_kernel)
        random = torch.rand_like(morph)
        heat_map = normalize_min_max(edge + morph + random)

        random = binariser(random, self.droprate)
        sample_map = random.to(torch.int) | morph.to(torch.int) | hard_edge.to(torch.int)
        return heat_map, sample_map.to(torch.float)
