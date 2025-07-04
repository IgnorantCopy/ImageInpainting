import torch
from torch import nn
import numpy as np


def same_padding(image, kernel_size, stride=1):
    _, _, _, width = image.shape
    return ((stride - 1) * width - stride + kernel_size) // 2


def gather_nd(params, indices):
    """
    Gathers values from the input tensor by index.
    :param params: [b, c, h]
    :param indices: [b, n, 2]
    :return: [b, n, c]
    """
    b, c, h = params.shape
    b, n, _ = indices.shape

    out = torch.zeros((b, n, c), dtype=params.dtype, device=params.device)
    for i in range(b):
        for j in range(n):
            out[i][j] = params[indices[i][j][0], :, indices[i][j][1]]
    return out


def sample(image: torch.Tensor, factor_index: torch.Tensor):
    """

    :param image: [b, c, h]
    :param factor_index: [b, n]
    :return:
    """
    b, c, h = image.shape
    b, n = factor_index.shape

    b_index = torch.arange(b, dtype=torch.int32)    # [b]
    b_index = torch.unsqueeze(b_index, dim=-1)      # [b, 1]
    b_index = b_index.repeat(1, n)                  # [b, n]

    factor_index = torch.clamp(factor_index, 0, max=h - 1)              # [b, n]
    append_index = torch.stack([b_index, factor_index], dim=-1)  # [b, n, 2]
    return gather_nd(image, append_index)               # [b, n, c]


def interpolate_first_dim(image, factor):
    """
    interpolate first dimension of image by factor
    :param image: [b, c, h]
    :param factor: [b, n], dtype=float32
    :return:
    """
    b, h, c = image.shape
    b, n = factor.shape

    factor0 = torch.floor(factor)
    factor1 = factor0 + 1

    factor0_value = sample(image, factor0.type(torch.int32))
    factor1_value = sample(image, factor1.type(torch.int32))

    weight0 = factor1 - factor      # [b, n]
    weight1 = factor - factor0
    weight0 = weight0.unsqueeze(-1) # [b, n, 1]
    weight1 = weight1.unsqueeze(-1)

    return weight0 * factor0_value + weight1 * factor1_value


def apply_crf(image: torch.Tensor, crf: torch.Tensor):
    """
    apply camera response function to image
    :param image: [b, c, w, h]
    :param crf: camera response function
    :return: [b, c, w, h]
    """
    batch_size, *shape = image.shape
    batch_size, k = crf.shape
    image = interpolate_first_dim(
        crf.unsqueeze(-1),
        torch.tensor(k - 1, dtype=torch.float32) * image.reshape(batch_size, -1)
    )
    return image.reshape([batch_size] + shape)


def convert_hdr_to_ldr(hdr: torch.Tensor, crf: torch.Tensor, t: torch.Tensor,
                       batch_size: int):
    """

    :param hdr: high dynamic range image
    :param crf: camera response function
    :param t: sensor exposure time
    :param batch_size: batch size
    :return:
    """
    hdr_image = hdr * t.reshape(t.shape[0], 1, 1, 1)

    # Augment Poisson and Gaussian noise
    sigma_s = 0.08 / 6 * torch.empty((hdr_image.shape[0], 3, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0)
    sigma_c = 0.005 * torch.empty((hdr_image.shape[0], 3, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0)
    noise_s = np.random.normal(size=hdr_image.shape) * sigma_s * hdr_image
    noise_c = np.random.normal(size=hdr_image.shape) * sigma_c
    noise = noise_s + noise_c
    hdr_image = nn.ReLU()(hdr_image + noise)

    # Dynamic range clipping
    clipped_image = torch.clamp(hdr_image, 0, 1)
    # Non-linear mapping
    ldr = apply_crf(clipped_image, crf)
    # Quantization
    quantized_hdr = (ldr * 255.0 + 0.5) / 255.0
    quantized_hdr_8bit = quantized_hdr.type(torch.uint8)
    image_list = []
    for i in range(batch_size):
        image = quantized_hdr_8bit[i]
        image =

