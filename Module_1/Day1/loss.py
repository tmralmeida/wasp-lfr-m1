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
        return gxt, gyt

    def __call__(self, xt, yt):
        """ Computes the loss
        input: img, label
        """
        
        gxt, gyt = self.cmpt_gs(xt, yt)
        loss = (gxt - torch.log(torch.exp(gxt) + self.niu)).mean() + \
            (-gyt - torch.log(1 + self.niu * torch.exp(-gyt))).mean()
        
        return -loss
    
class cNceLoss(object):
    """ Computes the cNCE loss function:
    inputs:
        - precision matrix
    """
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.mask = get_mask(28)
    
    def cmpt_g(self, xt, yt):
        _, neg_log_d = np.linalg.slogdet(self.lambda_.detach().numpy())
        c = 0.5 * 28**2 * np.log(2*np.pi) - 0.5 * neg_log_d
        log_phi_xt = cmpt_logpm(xt,self.lambda_,c, self.mask)
        log_phi_yt = cmpt_logpm(yt,self.lambda_,c, self.mask)
        # Slides:
        st = torch.stack((log_phi_xt, log_phi_yt), dim = 1)
        return log_phi_xt, st
        # return log_phi_yt - log_phi_xt 

    # Paper
    # def __call__(self, xt, yt):
    #     k = len(yt)
    #     loss = 0
    #     for i in range(k):
    #         gxy = self.cmpt_g(xt, yt[i])
    #         loss +=  (-gxy + torch.log(1 + torch.exp(gxy))).mean()
    #     return  loss
    
    # Slides
    def __call__(self, xt, yt):
        k = len(yt)
        loss = 0
        for i in range(k):
            log_phi_xt, st = self.cmpt_g(xt, yt[i])
            loss += (log_phi_xt - torch.logsumexp(st, dim = 1)).mean()
        return  -loss
        
    
