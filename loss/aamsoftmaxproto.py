#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.aamsoftmax as aamsoftmax
import loss.angleproto as angleproto

class LossFunction(nn.Module):
    def __init__(self, num_out, num_class, margin=0.3, scale=15, easy_margin=False, init_w=10.0, init_b=-5.0):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.aamsoftmax = aamsoftmax.LossFunction(num_out=num_out, num_class=num_class, margin=margin, scale=scale, easy_margin=easy_margin)
        self.angleproto = angleproto.LossFunction(init_w=init_w, init_b=init_b)
        print('Initialised AAMSoftmaxPrototypicalLoss')

    def forward(self, x, label=None):
        assert x.size()[1] == 2
        nlossS, prec1 = self.aamsoftmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))
        nlossM, _     = self.angleproto(x,None)
        return nlossS+nlossM, prec1
