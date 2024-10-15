# Deep-Lossy-Plus-Residual-Coding
Deep Lossy Plus Residual (DLPR) coding is the state-of-the-art learning-based lossless and near-lossless image compression algorithm with fast running speed (TPAMI'24, Journal extension of our CVPR'21 work).

## Usage
The code is run with `Python 3.9`, `Pytorch 1.11.0`, `Cudatoolkit 11.3.1`, `Timm 0.5.4`, `Torchac 0.9.3` and `Compressai 1.2.0`.

**Please note**: Inconsistent pytorch and cuda versions, especially higher versions, may cause failure.

### Data preparation
Download and extract `DIV2K_train_HR` and `DIV2K_valid_HR` high-resolution images from [`DIV2K Dataset`](https://data.vision.ee.ethz.ch/cvl/DIV2K/) to `Datasets` folder. 
```
./Datasets/
  DIV2K_train_HR/
      img1.png
      img2.png
      ...
  DIV2K_valid_HR/
      img3.png
      img4.png
      ...
  extract_patches_train.py
  extract_patches_valid.py
```
Run `extract_patches_train.py` and `extract_patches_valid.py` to crop 2K images into $128\times 128$ patches for network training and validation.

### DLPR coding for lossless compression ($\tau=0$)
In `DLPR_ll` folder, we provide the DLPR coding system for lossless compression only, without Scalable Quantized Residual Compressor. 
* Run `train.py` to train the DLPR coding system with $\lambda=0$.

* Run `test.py` to encode and decode test images of **arbitrary sizes**. Please adjust `input_path` to evaluate your own images.

* **Update** `encode.py`: Run `python encode.py -i input.png -o bitstream.bin` to encode `input.png` to `bitstream.bin`.

* **Update** `decode.py`: Run `python decode.py -i bitstream.bin -o rec.png` to decode `bitstream.bin` to `rec.png`.

The trained model `ckp_ll_trained` can be downloaded from [`Baidu Netdisk`](https://pan.baidu.com/s/1SrLK2OWhtFhn1BlobSdTmg) with access code `dlpr`.

### DLPR coding for near-lossless compression ($\tau\ge0$)
In `DLPR_nll` folder, we provide the DLPR coding system for scalable near-lossless compression. 
* Run `train.py` to train the DLPR coding system with $\lambda=0.03$.

* Run `test.py` to encode and decode test images of **arbitrary sizes**. Please adjust `input_path` and `tau` to evaluate your own images. If $\tau=0$, the special case is lossless image compression. However, `DLPR_ll` with $\lambda=0$ enjoys better lossless compression performance.

* **Update** `encode.py`: Run `python encode.py -tau k -i input.png -o bitstream.bin` to encode `input.png` to `bitstream.bin` with $\tau=k$.

* **Update** `decode.py`: Run `python decode.py -i bitstream.bin -o rec.png` to decode `bitstream.bin` to `rec.png`.

The trained model `ckp_nll_trained` can be downloaded from [`Baidu Netdisk`](https://pan.baidu.com/s/1SrLK2OWhtFhn1BlobSdTmg) with access code `dlpr`.

## Citation

```
@ARTICLE{DLPR,
  author={Bai, Yuanchao and Liu, Xianming and Wang, Kai and Ji, Xiangyang and Wu, Xiaolin and Gao, Wen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Deep Lossy Plus Residual Coding for Lossless and Near-lossless Image Compression}, 
  year={2024},
  volume={46},
  number={5},
  pages={3577-3594},
  doi={10.1109/TPAMI.2023.3348486}
}
```
