import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from utils.data.datasets import ImageDataset
from utils.data.transform import build_transforms

import os
import numpy as np
import math

from ll_model import LosslessCompressor, RateDistortion


def configure_optimizers(model, lr, aux_lr):

    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_lr,
    )

    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, train_step, tb_writer=None, clip_max_norm=None):
    
    model.train()
    device = next(model.parameters()).device
    
    train_size = 0
    for x in train_dataloader:
        x = x.to(device).contiguous()
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        out = model(x)
        
        out_criterion = criterion(out, x)
        out_criterion["loss"].backward()
        if clip_max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        aux_loss = model.lossy_compressor.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        train_step += 1
        if tb_writer and train_step % 10==1:
            tb_writer.add_scalar('train loss', out_criterion["loss"].item(), train_step)
            tb_writer.add_scalar('train mse', out_criterion["mse_loss"].item(), train_step)
            tb_writer.add_scalar('train img bpp', out_criterion["img_bpp"].item(), train_step)
            tb_writer.add_scalar('train res bpp', out_criterion["res_bpp"].item(), train_step)
            tb_writer.add_scalar('train bpp', out_criterion["img_bpp"].item() + out_criterion["res_bpp"].item(), train_step)
            tb_writer.add_scalar('train aux', model.lossy_compressor.aux_loss().item(), train_step)
            
        train_size += x.shape[0]
        
    print("train sz:{}".format(train_size))
    return train_step
            
        
def eval_epoch(model, criterion, eval_dataloader, epoch, tb_writer=None):
    
    model.eval()
    device = next(model.parameters()).device
    
    loss = 0
    img_bpp = 0
    res_bpp = 0
    mse_loss = 0
    aux_loss = []
    
    if tb_writer:
        save_imgs = True
    else:
        save_imgs = False
    
    
    eval_size = 0
    with torch.no_grad():
        for x in eval_dataloader:
            x = x.to(device).contiguous()
            
            out = model(x)
            out_criterion = criterion(out, x)

            N, _, H, W = x.shape
            
            loss += out_criterion["loss"] * N 
            img_bpp += out_criterion["img_bpp"] * N 
            res_bpp += out_criterion["res_bpp"] * N
            mse_loss += out_criterion["mse_loss"] * N
            aux_loss.append(model.lossy_compressor.aux_loss())
            
            if save_imgs:
                x_rec = out["x_hat"].clamp_(0, 255)
                tb_writer.add_image('input/0', x[0,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('input/1', x[1,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('input/2', x[2,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('output/0', x_rec[0,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('output/1', x_rec[1,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('output/2', x_rec[2,:,:,:].to(torch.uint8), epoch)
                save_imgs = False
            
            eval_size += x.shape[0]
            
        loss = (loss/eval_size).item()
        img_bpp = (img_bpp/eval_size).item()
        res_bpp = (res_bpp/eval_size).item()
        bpp_loss = img_bpp + res_bpp
        mse_loss = (mse_loss/eval_size).item()
        aux_loss = (sum(aux_loss)/len(aux_loss)).item()
        psnr = 10. * np.log10(255.**2/mse_loss)
        if tb_writer:
            tb_writer.add_scalar('eval/eval loss', loss, epoch)
            tb_writer.add_scalar('eval/eval bpp', bpp_loss, epoch)
            tb_writer.add_scalar('eval/eval img bpp', img_bpp, epoch)
            tb_writer.add_scalar('eval/eval res bpp', res_bpp, epoch)
            tb_writer.add_scalar('eval/eval mse', mse_loss, epoch)
            tb_writer.add_scalar('eval/eval psnr', psnr, epoch)
            tb_writer.add_scalar('eval/eval aux', aux_loss, epoch)
            
        print("eval sz:{}".format(eval_size))
    
    return loss, bpp_loss, img_bpp, res_bpp, mse_loss, psnr, aux_loss
        

def train(train_dataloader, eval_dataloader, epochs, ckp_dir, log_dir, resume=False):
    ll_module = LosslessCompressor(192, "3P").cuda()
    rd_criterion = RateDistortion(lmbda=0.0)
    optimizer, aux_optimizer = configure_optimizers(ll_module, lr=1e-4, aux_lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[350, 390, 430, 470, 510, 550, 590], gamma=0.9, verbose=True)

    tb_writer = SummaryWriter(log_dir)

    if resume:
        ckp = torch.load(os.path.join(ckp_dir, "ckp.tar"))
        start_epoch = ckp['epoch'] + 1
        ll_module.load_state_dict(ckp['model_state_dict'])
        optimizer.load_state_dict(ckp['optimizer_state_dict'])
        aux_optimizer.load_state_dict(ckp['aux_optimizer_state_dict'])
        scheduler.load_state_dict(ckp['scheduler'])
        train_step = ckp['step']
    else:
        start_epoch = 0
        train_step = 0

    for epoch in range(start_epoch, epochs):
        # train
        train_step = train_one_epoch(ll_module, rd_criterion, train_dataloader, optimizer, aux_optimizer, train_step, tb_writer, clip_max_norm=1.0)
        # eval
        loss, bpp_loss, img_bpp, res_bpp, mse_loss, psnr, aux_loss = eval_epoch(ll_module, rd_criterion, eval_dataloader, epoch, tb_writer)

        scheduler.step()

        # save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': ll_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'aux_optimizer_state_dict': aux_optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step':train_step,
        }, os.path.join(ckp_dir, "ckp.tar"))

        print("Epoch(Eval):{}, bpp:{}, img bpp:{}, res bpp:{}, mse:{}, psnr:{}".format(epoch, bpp_loss, img_bpp, res_bpp, mse_loss, psnr))

    tb_writer.close()
    
    
if __name__ == "__main__":
    
    ckp_dir = './ckp_ll'
    log_dir = './logs_ll'
    epochs = 600
    resume = False

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    
    transform_train = build_transforms("p64")
    transform_eval = build_transforms("p64_centercrop")
    train_data= ImageDataset("../Datasets/DIV2K_train_p128", transform = transform_train)
    train_dataloader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=True)
    
    eval_data= ImageDataset("../Datasets/DIV2K_valid_p128", transform = transform_eval)
    eval_dataloader = data.DataLoader(eval_data, batch_size=64, shuffle=False, num_workers=8, prefetch_factor=2, pin_memory=True)
    
    train(train_dataloader, eval_dataloader, epochs, ckp_dir, log_dir, resume)
    