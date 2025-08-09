import torch
import torch.nn as nn
import random
from pytorch_wavelets import DWTForward, DWTInverse

class MST(nn.Module):
    def __init__(self, in_channels):
        super(MST, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, E, F):
        # Step 1: Merge E and F along the channel dimension
        merged = torch.cat((E, F), dim=1)

        # Step 2: Apply a 1x1 convolution to reduce the channels back to in_channels
        reduced = self.conv1x1(merged)

        # Step 3: Apply a sigmoid activation function
        G = self.sigmoid(reduced)

        # Step 4: Element-wise multiplication of F and G
        FG = F * G
        EG = E * (1 - G)
        # Step 5: Element-wise addition of E and FG
        output = EG + FG

        return output

def wavelet_DST(A_fea, B_fea, J):#A是Fs,B是Fr
    # wavelet: DWT
    xfm = DWTForward(J=J, wave='haar', mode='zero').cuda()
    A_fea_Yl, A_fea_Yh = xfm(A_fea)
    B_fea_Yl, B_fea_Yh = xfm(B_fea)
    A_LH, A_HL, A_HH = A_fea_Yh[0][:, :, 0, :, :], A_fea_Yh[0][:, :, 1, :, :], A_fea_Yh[0][:, :, 2, :, :]
    B_LH, B_HL, B_HH = B_fea_Yh[0][:, :, 0, :, :], A_fea_Yh[0][:, :, 1, :, :], A_fea_Yh[0][:, :, 2, :, :]
    mst_LH = MST(A_LH.shape[1]).cuda()
    mst_HL = MST(A_HL.shape[1]).cuda()
    mst_HH = MST(A_HH.shape[1]).cuda()
    C_LH = mst_LH(B_LH, A_LH)
    C_HL = mst_HL(B_HL, A_HL)
    C_HH = mst_HH(B_HH, A_HH)

    ifm = DWTInverse(wave='haar', mode='zero').cuda()
    C_fea_Yh = torch.stack([C_LH, C_HL, C_HH], dim=2)
    C_fea_Yh = [C_fea_Yh]
    C_fea = ifm((B_fea_Yl,C_fea_Yh))
    # D = torch.cat((A_fea, C_fea))
    return  C_fea

class SaveMuVar():
    mu, var = None, None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.mu = output.detach().cpu().mean(dim=[2, 3], keepdim=True).squeeze(-1).squeeze(-1).numpy()
        self.var = output.detach().cpu().var(dim=[2, 3], keepdim=True).squeeze(-1).squeeze(-1).numpy()

    def remove(self):
        self.hook.remove()


class EFDMix(nn.Module):
    """EFDMix.

    Reference:
      Zhang et al. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. CVPR 2022.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        x_view = x.view(B, C, -1)
        value_x, index_x = torch.sort(x_view)  # sort inputs
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index)
        new_x = x_view + (x_view_copy - x_view.detach()) * (1-lmda)
        return new_x.view(B, C, W, H)


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True  # Train: True, Test: False

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        return x_normed*sig_mix + mu_mix


class TriD(nn.Module):
    """TriD.
    Reference:
      Chen et al. Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation. MICCAI 2023.
    """
    def __init__(self, p=0.5, eps=1e-6, alpha=0.1):
        """
        Args:
          p (float): probability of using TriD.
          eps (float): scaling parameter to avoid numerical issues.
          alpha (float): parameter of the Beta distribution.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self._activated = True  # Train: True, Test: False
        self.beta = torch.distributions.Beta(alpha, alpha)

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self._activated:
            return x

        if random.random() > self.p:
            return x

        N, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        # Sample mu and var from an uniform distribution, i.e., mu ～ U(0.0, 1.0), var ～ U(0.0, 1.0)
        mu_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)
        var_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)

        lmda = self.beta.sample((N, C, 1, 1))
        bernoulli = torch.bernoulli(lmda).to(x.device)

        mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
        sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
        # c = wavelet_DST(x,x_normed * sig_mix + mu_mix,1)

        return x_normed * sig_mix + mu_mix
