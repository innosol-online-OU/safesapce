import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp, sqrt

class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss.
    
    Used to penalize visual artifacts during adversarial optimization.
    Returns: 1.0 - SSIM (Minimize this to maximize similarity).
    """
    def __init__(self, window_size=11, sigma=1.5, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.sigma = sigma
        self.window = self.create_window(window_size, channel, sigma)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel, sigma):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, img1, img2):
        """
        Compute 1 - SSIM loss.
        Args:
            img1 (torch.Tensor): Image 1 [B, C, H, W], range [0, 1]
            img2 (torch.Tensor): Image 2 [B, C, H, W], range [0, 1]
        """
        # Ensure window is on same device
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        return 1.0 - self._ssim(img1, img2, self.window, self.window_size, self.channel)
