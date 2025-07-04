# coding=utf-8
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as spio

import torch
import torch.nn as nn

import utils_image as util


class ToneMapping(nn.Module):
    '''
    Tone Mapping
    '''
    def __init__(self, ToneCurveX, ToneCurveY, delta=1e-6):
        super(ToneMapping, self).__init__()
        self.delta = delta
        xi = np.linspace(0, 1, num=int(1/delta+1), endpoint=True)
        yi = interp1d(ToneCurveX, ToneCurveY, kind='cubic')(xi)
        self.register_buffer('yi', torch.from_numpy(yi).float())

    def forward(self, x):
        x = self.yi[(torch.round(x.clamp_(0, 1) / self.delta)).long()]
        return x.clamp_(0, 1)

ToneCurve_idx = 202
data = spio.loadmat('tonecurves.mat', struct_as_record=False, squeeze_me=True)
ToneCurves = data['ToneCurves']
ToneCurve = ToneCurves[ToneCurve_idx, :]
ToneCurve = np.reshape(ToneCurve, (2, -1), 'F')
ToneCurveX, ToneCurveY = ToneCurve[0, :], ToneCurve[1, :]

ToneMappingLayer = ToneMapping(ToneCurveX, ToneCurveY)


import matplotlib.pyplot as plt
plt.plot(ToneCurveX, ToneCurveY)  # 原始数据点

plt.xlabel('x')
plt.ylabel('y')
#plt.title('Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()


img = util.imread_uint('zurich.png', n_channels=3)
img = util.uint2tensor4(img)

img_tm = ToneMappingLayer.forward(img)

util.imsave(util.tensor2uint(img_tm), 'zurich_tm.png')

util.imshow(np.column_stack((util.tensor2uint(img),util.tensor2uint(img_tm))), title='Tone Mapping: before vs. after')
