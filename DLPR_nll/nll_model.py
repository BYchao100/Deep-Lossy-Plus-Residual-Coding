import torch
import torch.nn as nn

from compressai.layers import GDN, conv3x3, subpel_conv3x3, ResidualBlock
from compressai.models.utils import update_registered_buffers
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel

from logisticmixturemodel import LogisticMixtureModel
from custom_layers import conv1x1, subpel_conv1x1, downsample_conv1x1, SWin_Attention, maskedconv7x7_parallel, Conv2d_cond

import math
import random

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class AnalysisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=2)
        self.gdn = GDN(out_ch)
        self.skip = conv3x3(in_ch, out_ch, stride=2)
        self.rb = ResidualBlock(out_ch, out_ch)
        
    def forward(self, input):
        out = self.conv1(input)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out = out + self.skip(input)

        out = self.rb(out)
        return out
        

class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rb = ResidualBlock(in_ch, out_ch)
        self.conv_up = subpel_conv3x3(out_ch, out_ch, r=2)
        self.igdn = GDN(out_ch, inverse=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.upsample = subpel_conv3x3(out_ch, out_ch, r=2)
                
    def forward(self, input):
        out1 = self.rb(input)
        out = self.conv_up(out1)
        out = self.igdn(out)
        out = self.conv(out)
        out = self.leaky_relu(out)
        
        out = out + self.upsample(out1)
        
        return out
    

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            AnalysisBlock(in_ch, out_ch),
            AnalysisBlock(out_ch, out_ch),
            SWin_Attention(dim=out_ch, num_heads=8, window_size=8),
            AnalysisBlock(out_ch, out_ch),
            conv3x3(out_ch, out_ch, stride=2),
            SWin_Attention(dim=out_ch, num_heads=8, window_size=4),
        )
        
    def forward(self, input):
        out = self.layers(input)
        return out
    
    
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, prior_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            SWin_Attention(dim=in_ch, num_heads=8, window_size=4),
            SynthesisBlock(in_ch, in_ch),
            SynthesisBlock(in_ch, in_ch),
            SWin_Attention(dim=in_ch, num_heads=8, window_size=8),
            SynthesisBlock(in_ch, in_ch)
        )
        self.conv_rec = subpel_conv3x3(in_ch, out_ch, r=2)
        self.conv_prior = subpel_conv3x3(in_ch, prior_ch, r=2)

    def forward(self, input):
        out = self.layers(input)
        rec = self.conv_rec(out)
        prior = self.conv_prior(out)
        return rec, prior
            

class HyperEncoder(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            downsample_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            downsample_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch)
        )
        
    def forward(self, input):
        out = self.layers(input)
        return out
    
    
class HyperDecoder(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers_mu = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

        self.layers_sigma = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )
        
    def forward(self, input):
        mu = self.layers_mu(input)
        sigma = self.layers_sigma(input)
        return mu, sigma


class ResBlock_1x1(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        out = self.layers(input)
        out = out + input

        return out


class ResBlock_1x1_cond(nn.Module):
    def __init__(self, num_ch, cond_num, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_cond1 = Conv2d_cond(cond_num, num_ch, num_ch, kernel_size=1, bias=False)
        self.conv_cond2 = Conv2d_cond(cond_num, num_ch, num_ch, kernel_size=1, bias=False)
        self.conv_cond3 = Conv2d_cond(cond_num, num_ch, num_ch, kernel_size=1, bias=False)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)
        self.leaky_relu3 = nn.LeakyReLU(inplace=True)


    def forward(self, input, cond):
        out = self.leaky_relu1(self.conv_cond1(input, cond))
        out = self.leaky_relu2(self.conv_cond2(out, cond))
        out = self.leaky_relu3(self.conv_cond3(out, cond))

        out = out + input

        return out
    
        
class LossyCompressor(CompressionModel):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(entropy_bottleneck_channels=num_ch, *args, **kwargs)
        self.encoder = Encoder(3, num_ch)
        self.decoder = Decoder(num_ch, 3, 256)
        self.hyperencoder = HyperEncoder(num_ch)
        self.hyperdecoder = HyperDecoder(num_ch)
        
        self.gaussian_conditional = GaussianConditional(None)
    
    def forward(self, input):
        y = self.encoder(input/255.)
        z = self.hyperencoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma_hat, means=mu_hat)
        x_hat, res_prior = self.decoder(y_hat)
        x_hat = x_hat*255.
        
        return {
            "x_hat":x_hat,
            "res_prior":res_prior,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


class ResidualCompressor(nn.Module):
    def __init__(self, num_ch, num_mixtures, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ResBlock = ResBlock_1x1(num_ch)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, 10*num_mixtures),
        )
         
    def forward(self, input):
        out = self.ResBlock(input)
        out = self.layers(out)
        
        return out


class ResidualCompressor_cond(nn.Module):
    def __init__(self, num_ch, num_mixtures, cond_num, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ResBlock_cond = ResBlock_1x1_cond(num_ch, cond_num)
        self.conv_cond1 = Conv2d_cond(cond_num, num_ch, num_ch, kernel_size=1, bias=False)
        self.conv_cond2 = Conv2d_cond(cond_num, num_ch, 10*num_mixtures, kernel_size=1, bias=False)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
         
    def forward(self, input, cond):
        out = self.ResBlock_cond(input, cond)
        out = self.leaky_relu1(self.conv_cond1(out, cond))
        out = self.conv_cond2(out, cond)
        
        return out

        
class NearLosslessCompressor(nn.Module):
    def __init__(self, num_ch, cond_num, mask_type="3P", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossy_compressor = LossyCompressor(num_ch)
        self.mask_conv = maskedconv7x7_parallel(3, 256, mask_type)
        self.residual_compressor = ResidualCompressor(256, 5)
        self.fusion = conv1x1(256 * 2, 256)

        self.residual_compressor_cond = ResidualCompressor_cond(256, 5, cond_num)
        self.fusion_cond = Conv2d_cond(cond_num, 256 * 2, 256, kernel_size=1, bias=False)

        self.cond_num = cond_num


    def forward(self, input):
        lossy_out = self.lossy_compressor(input)
        noise = torch.empty_like(input).uniform_(-0.5, 0.5)

        x_hat = torch.clamp(lossy_out["x_hat"] + noise, min = 0., max = 255.)
        
        res = input - x_hat
        res_n = res/255.*2
        
        context = self.mask_conv(res_n)
        res_prior_context = self.fusion(torch.cat((lossy_out["res_prior"], context), 1))

        lmm_params = self.residual_compressor(res_prior_context)
        mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)

        lmm = LogisticMixtureModel(mu, log_sigma, weights, coeffs)
        
        log_res_likelihoods = lmm(res_n)

        # scalable nll
        tau = random.randint(1, self.cond_num)
        res_q = torch.where(res>=0, (2*tau+1) * torch.floor((res+tau)/(2*tau+1)),
                            (2*tau+1) * torch.ceil((res-tau)/(2*tau+1)))
        context_q = self.mask_conv(res_q/255.*2)
        res_prior_context_bias_cat = torch.cat((lossy_out["res_prior"], context_q), 1).detach()
        
        batch_sz = input.shape[0]
        tau_list = torch.ones(batch_sz, dtype=torch.long, device=input.device) * (tau-1)
        cond = nn.functional.one_hot(tau_list, num_classes=self.cond_num).float()   

        res_prior_context_bias = self.fusion_cond(res_prior_context_bias_cat, cond)
        lmm_params_cond = self.residual_compressor_cond(res_prior_context_bias, cond)
        mu_cond, log_sigma_cond, coeffs_cond, weights_cond = torch.split(lmm_params_cond, 15, dim=1)
        lmm_model_cond = LogisticMixtureModel(mu_cond, log_sigma_cond, weights_cond, coeffs_cond)

        log_res_likelihoods_cond = lmm_model_cond(res_n.detach())

        return {
            "x_hat": lossy_out["x_hat"],
            "res_likelihoods": log_res_likelihoods,
            "res_likelihoods_cond": log_res_likelihoods_cond,
            "img_likelihoods": lossy_out["likelihoods"],
        }


    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.lossy_compressor.gaussian_conditional,
            "lossy_compressor.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def parameters_wo_cond(self):
        parameters = (
            p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and not n.startswith("residual_compressor_cond.") and not n.startswith("fusion_cond.")
        )
    
        return parameters

    def parameters_w_cond(self):
        parameters = (
            p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and (n.startswith("residual_compressor_cond.") or n.startswith("fusion_cond."))
        )
    
        return parameters

    
    
class RateDistortion(nn.Module):
    def __init__(self, lmbda=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N*H*W
        
        out["img_bpp"] = sum(
            (torch.log(likelihoods).sum()/(-math.log(2) *num_pixels))
            for likelihoods in output["img_likelihoods"].values()
        )
        out["res_bpp"] = output["res_likelihoods"].sum()/(-math.log(2) *num_pixels)
        out["res_bpp_cond"] = output["res_likelihoods_cond"].sum()/(-math.log(2) *num_pixels)

        x_hat = torch.clamp(output["x_hat"], min = -0.5, max = 255.5)
        out["mse_loss"] = self.mse(x_hat, target)

        out["loss"] = self.lmbda * out["mse_loss"] + out["img_bpp"] + out["res_bpp"]
        
        return out