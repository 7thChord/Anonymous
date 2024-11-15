# Helper functions for noise initialization for sampling
import torch


def calc_mean_std(feat, eps=1e-8):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

#AdaIN
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def dct_low_pass_filter(dct_coefficients, percentage=0.3, cutoff_shape="rect"): # 2d [b c f h w]
    """
    Applies a low pass filter to the given DCT coefficients.

    :param dct_coefficients: 2D tensor of DCT coefficients
    :param percentage: percentage of coefficients to keep (between 0 and 1)
    :return: 2D tensor of DCT coefficients after applying the low pass filter
    """
    # Determine the cutoff indices for both dimensions
    h, w = dct_coefficients.shape[-2:]
    cutoff_y = int(h * percentage)
    cutoff_x = int(w * percentage)

    if cutoff_shape=='rect':
        # Create a mask with the same shape as the DCT coefficients
        mask = torch.zeros_like(dct_coefficients)
        # Set the top-left corner of the mask to 1 (the low-frequency area)
        mask[:, :, :, :cutoff_y, :cutoff_x] = 1
    elif cutoff_shape=='tri':
        # Adapted by us. Keep top left corner.
        mask2d = torch.zeros(h, w)
        for y in range(cutoff_y):
            for x in range(cutoff_x):
                r = y/h + x/w
                if r < percentage:
                    mask2d[y, x] = 1
        mask = torch.zeros_like(dct_coefficients)
        mask[:,:,:] = mask2d
    else:
        raise NotImplementedError

    return mask

def exchanged_mixed_dct_freq(noise, base_content, LPF_3d, normalized=False):
    import torch_dct
    # noise dct
    noise_freq = torch_dct.dct_3d(noise, 'ortho')

    # frequency
    HPF_3d = 1 - LPF_3d
    noise_freq_high = noise_freq * HPF_3d

    # base frame dct
    base_content_freq = torch_dct.dct_3d(base_content, 'ortho')

    # base content low frequency
    base_content_freq_low = base_content_freq * LPF_3d

    # mixed frequency
    mixed_freq = base_content_freq_low + noise_freq_high

    # idct
    mixed_freq = torch_dct.idct_3d(mixed_freq, 'ortho')

    return mixed_freq

def analytic_init(scheduler, timestep, shape, generator, device, dtype):
    '''
    Generate a noise signal following [CIL Eq.5].
    timestep: int, M
    shape: target tensor shape (BCFHW)
    '''
    mu_data = 0
    var_data = 0.566

    alphas_cumprod = scheduler.alphas_cumprod
    alpha_M2 = alphas_cumprod[timestep]
    sigma_M2 = (1 - alphas_cumprod[timestep])

    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    var = alpha_M2 * var_data + sigma_M2

    noise = noise * torch.sqrt(var) + mu_data
    return noise