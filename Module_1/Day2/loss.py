import torch
import torch.nn as nn
import numpy as np
from lib import cmpt_logpn, cmpt_logpm, get_mask



class NceLoss(object):
    """ Computes the NCE loss function:
    inputs:
        - p_theta parms
        - p_n params
        - niu = Tn/Td
    """
    def __init__(self,
                 p_theta,
                 p_n,
                 niu = 1):
        self.lambda_ = p_theta["pmat"]
        self.c_ = p_theta["c"]
        self.mean = p_n["mean"]
        self.cov = p_n["cov"]
        self.niu = niu
        self.mask_prec = get_mask(28)

    def cmpt_gs(self, xt, yt):
        """ Computes h(yt;theta) and h(xt;theta)
        """
        gxt = cmpt_logpm(xt,self.lambda_, self.c_, self.mask_prec) \
            - cmpt_logpn(xt, self.mean, self.cov)
        gyt = cmpt_logpm(yt, self.lambda_, self.c_, self.mask_prec) \
            - cmpt_logpn(yt, self.mean, self.cov)
        # print(gxt)
        # print(gyt)
        # print()
        return gxt, gyt

    def __call__(self, xt, yt):
        """ Computes the loss
        input: img, label
        """
        
        gxt, gyt = self.cmpt_gs(xt, yt)
        loss = (gxt - torch.log(torch.exp(gxt) + self.niu)).mean() + \
            (-gyt - torch.log(1 + self.niu * torch.exp(-gyt))).mean()
        return -loss
    

        
    
