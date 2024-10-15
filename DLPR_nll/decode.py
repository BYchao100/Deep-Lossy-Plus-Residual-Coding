import torch
import torch.nn as nn
import torch.nn.functional as F

from nll_model_eval import NearLosslessCompressor
import argparse
from PIL import Image
import numpy as np
import os
import time
import pickle
import torchac

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def coding_order_table7x7(patch_sz=64, mask_type="3P"):
    
    if mask_type not in ("5P", "4P", "3P", "2P", "P"):
        raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)

    if mask_type == "5P":
        for i in range(patch_sz):
            start = 4 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "4P":
        for i in range(patch_sz):
            start = 3 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "3P":
        for i in range(patch_sz):
            start = 2 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)     
    elif mask_type == "2P":
        for i in range(patch_sz):
            start = i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "P":
        for i in range(patch_sz):
            start = 1
            COT[i, :] = torch.arange(start, start + patch_sz)      

    return COT
    
def patch2img(patch, img_sz):
    h, w = img_sz
    patch_sz = patch.shape[2]
    h_num = h//patch_sz - (patch_sz -(h % patch_sz)) // patch_sz + 1
    w_num = w//patch_sz - (patch_sz -(w % patch_sz)) // patch_sz + 1

    patch = patch.permute(0, 2, 3, 1)
    patch_w = torch.chunk(patch, w_num, dim=0)
    patch_h = torch.cat(patch_w, dim=2)

    patch_h = torch.chunk(patch_h, h_num, dim=0)
    img = torch.cat(patch_h, dim=1).squeeze(0)

    return img[-h:, -w:, :]

def decompress(model, code_lossy, code_res, img_shape, res_range, COT, tau=0, mix_num=5):

    norm_scale = 1/255.*2
    half = (0.5 + tau) * norm_scale
    mix_num2 = 2 * mix_num
    bin_sz = 2 * tau + 1
    samples_end = (255 // bin_sz) * bin_sz

    device = next(model.parameters()).device

    res_q_min, res_q_max = res_range
    res_q_max_norm = res_q_max * norm_scale
    res_q_min_norm = res_q_min * norm_scale

    res_q_max_idx = (res_q_max + samples_end) // bin_sz
    res_q_min_idx = (res_q_min + samples_end) // bin_sz
    print("Decode Res range:[{}({}),{}({})]".format(res_q_min, res_q_min_idx, res_q_max, res_q_max_idx))

    samples = torch.arange(res_q_min, res_q_max + 1, step=bin_sz, dtype=torch.float32).to(device)
    samples = samples * norm_scale

    with torch.no_grad():

        time_start = time.time()

        rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"])

        time_end_ls = time.time()

        x_hat = rec_lossy["x_hat"]
        res_prior = rec_lossy["res_prior"]

        res_tmp = torch.zeros_like(x_hat)
        max_step = torch.max(COT)

        j = 0
        for i in range(max_step):

            h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
            ctx = model.mask_conv(res_tmp * norm_scale)[:, :, h_idx, w_idx].unsqueeze(3)
            rp = res_prior[:, :, h_idx, w_idx].unsqueeze(3)

            res_crop = res_tmp[:, :, h_idx, w_idx].unsqueeze(3)

            if tau == 0:
                rp_ctx = model.fusion(torch.cat((rp, ctx), 1))
                lmm_params = model.residual_compressor(rp_ctx)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)
                coeffs = torch.tanh(coeffs)
            elif tau > 0:
                if tau > 5:
                    tau = 5
                rp_ctx = model.fusion_cond.run(torch.cat((rp, ctx), 1), tau-1)
                lmm_params = model.residual_compressor_cond.run(rp_ctx, tau-1)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)
                coeffs = torch.tanh(coeffs)  

            for c in range(3):
                if c==0:
                    mu_c = mu[:, :mix_num, :, :].permute(0, 2, 1, 3)
                elif c==1:
                    mu_c = mu[:, mix_num:mix_num2, :, :] + (res_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, :mix_num, :, :]
                    mu_c = mu_c.permute(0, 2, 1, 3)
                else:
                    mu_c = mu[:, mix_num2:, :, :] + (res_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, mix_num:mix_num2, :, :] + \
                           (res_crop[:, 1:2, :, :] * norm_scale) * coeffs[:, mix_num2:, :, :]
                    mu_c = mu_c.permute(0, 2, 1, 3)

                samples_centered = samples - mu_c
                inv_sigma = torch.exp(-log_sigma[:, c * mix_num:(c + 1) * mix_num, :, :].permute(0, 2, 1, 3))
                plus_in = inv_sigma * (samples_centered + half)
                cdf_plus = torch.sigmoid(plus_in)
                min_in = inv_sigma * (samples_centered - half)
                cdf_min = torch.sigmoid(min_in)
                cdf_delta = cdf_plus - cdf_min
                one_minus_cdf_min = torch.exp(-F.softplus(min_in))  # res_q_max
                cdf_plus = torch.exp(plus_in - F.softplus(plus_in))  # res_q_min

                samples2 = samples - torch.zeros_like(mu_c)
                cdf_delta = torch.where(samples2 - half < res_q_min_norm, cdf_plus,
                                        torch.where(samples2 + half > res_q_max_norm, one_minus_cdf_min,
                                                    cdf_delta))


                weights_c = weights.permute(0, 2, 1, 3)
                m = torch.amax(weights_c, 2, keepdim=True)
                weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                pmf = torch.sum(cdf_delta * weights_c, dim=2)

                pmf = pmf.clamp_(1./64800, 1.)
                pmf = pmf/torch.sum(pmf, dim=2, keepdim=True)
                cdf = torch.cumsum(pmf, dim=2).clamp_(0., 1.)
                cdf = F.pad(cdf, (1, 0))

                symbol_out = torchac.decode_float_cdf(cdf.cpu(), code_res[j], needs_normalization=False)
                res_crop[:, c, :, 0] = symbol_out.float() * bin_sz + res_q_min
                j += 1

            res_tmp[:, :, h_idx, w_idx] = res_crop.squeeze(3)

        time_end_res = time.time()

    print("total runtime:{:.2f}, lossy runtime:{:.2f}, res runtime:{:.2f}".format(time_end_res-time_start, time_end_ls-time_start, time_end_res-time_end_ls))

    x_lossy = patch2img(x_hat, img_shape)
    res = patch2img(res_tmp, img_shape)

    x_rec = (x_lossy + res).clamp_(min=0, max=255)

    return x_rec
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image decompression using neural networks.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input bitstream file path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output image file path')

    args = parser.parse_args()
    ckp_dir = "./ckp_nll"

    # im = Image.open('input.png')
    # I = np.array(im).astype(np.float32)

    device = torch.device('cuda')
    nll_module = NearLosslessCompressor(192, 5).eval().to(device)

    ckp = torch.load(os.path.join(ckp_dir, "ckp.tar"), map_location=device)
    nll_module.load_state_dict(ckp['model_state_dict'])
    nll_module.lossy_compressor.update(force=True)

    # COT
    COT = coding_order_table7x7()

    with open(args.input, 'rb') as f:
        code_lossy, code_res, img_shape, res_range,tau = pickle.load(f)

    I_nll = decompress(nll_module, code_lossy, code_res, img_shape, res_range, COT, tau)
    I_nll = I_nll.cpu().numpy()

    # max_diff = np.max(np.abs(I_nll - I))
    # mse_loss = np.mean((I - I_nll) ** 2)
    # psnr_nll = 10. * np.log10(255. ** 2 / mse_loss)
    # print("max diff nll:{}, psnr nll:{:.4f}".format(max_diff, psnr_nll))

    I_nll = I_nll.astype(np.uint8)
    im_nll = Image.fromarray(I_nll)

    im_nll.save(args.output)



    
