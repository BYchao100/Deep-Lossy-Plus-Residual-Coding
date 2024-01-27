import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticMixtureModel(nn.Module):
    def __init__(self, mean, log_sigma, mixture_weights, autoregression_coefficients=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N,C,H,W = mean.shape
        self.mix_num = C//3
        self.mean = torch.reshape(mean, (N,3,self.mix_num,H,W))  
        
        self.log_sigma = torch.reshape(log_sigma, (N,3,self.mix_num,H,W))
        self.log_sigma = torch.clamp(self.log_sigma, min=-7.)
        
        # log weights
        self.mixture_weights = torch.reshape(mixture_weights, (N,1,self.mix_num,H,W)) + torch.zeros(N,3,self.mix_num,H,W, device=mixture_weights.device)
        
        if autoregression_coefficients is not None:
            self.coeffs = torch.tanh(autoregression_coefficients)
            self.coeffs = torch.reshape(self.coeffs, (N,3,self.mix_num,H,W))
            self.channel_ag = True
        else:
            self.channel_ag = False
    
    def forward(self, input, tau=0):
        N,C,H,W = input.shape
        half = float(0.5 + tau)/255.*2
        
        x = torch.reshape(input, (N,C,1,H,W)) + torch.zeros(N,C,self.mix_num,H,W, device=input.device)
        
        if self.channel_ag:
            m1 = torch.reshape(self.mean[:,0,:,:,:], (N,1,self.mix_num,H,W))
            m2 = self.mean[:,1,:,:,:] + self.coeffs[:,0,:,:,:] * x[:,0,:,:,:]
            m2 = torch.reshape(m2, (N,1,self.mix_num,H,W))
            m3 = self.mean[:,2,:,:,:] + self.coeffs[:,1,:,:,:] * x[:,0,:,:,:] + self.coeffs[:,2,:,:,:] * x[:,1,:,:,:]
            m3 = torch.reshape(m3, (N,1,self.mix_num,H,W))
            self.mean = torch.cat((m1,m2,m3), 1)
            
        centered_x = x - self.mean
        inv_sigma = torch.exp(-self.log_sigma)
        plus_in = inv_sigma * (centered_x + half)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_sigma * (centered_x - half)
        cdf_min = torch.sigmoid(min_in)
        log_one_minus_cdf_min = -F.softplus(min_in) # 255
        log_cdf_plus = plus_in - F.softplus(plus_in) # -255
        cdf_delta = cdf_plus - cdf_min
        
        log_probs = torch.where(x - half < -1.999, log_cdf_plus,
                               torch.where(x + half > 1.999, log_one_minus_cdf_min,
                                          torch.log(torch.clamp(cdf_delta, min=1e-9))))
        
        log_probs = log_probs + self.log_prob_from_logits(self.mixture_weights)
        
        return self.log_sum_exp(log_probs)
    
    
    def log_prob_from_logits(self, x): # normalize
        axis = 2
        m = torch.amax(x, axis, keepdim=True)
        return x - m - torch.log(torch.sum(torch.exp(x-m), axis, keepdim=True))
    
    
    def log_sum_exp(self, x):
        axis = 2
        m = torch.amax(x, axis)
        m2 = torch.amax(x, axis, keepdim=True)
        
        return m + torch.log(torch.sum(torch.exp(x-m2), axis))
        
        
        