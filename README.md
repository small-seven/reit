# ReiT
This repository contains code for our CVPR paper ["Random Entangled Tokens for Adversarially Robust Vision Transformer"](https://openaccess.thecvf.com/content/CVPR2024/html/Gong_Random_Entangled_Tokens_for_Adversarially_Robust_Vision_Transformer_CVPR_2024_paper.html).

# Dependencies
We were using PyTorch 1.10.0 for all the experiments. You may want to install other versions of PyTorch according to the cuda version of your computer/server.
The code is run and tested on the Artemis HPC server, 4090 lab-cluster server, and NCI server with multiple GPUs. Running on a single GPU may need adjustments.

# Data and pre-trained models
We used the pre-trained models in the [TIMM package](https://github.com/guigrpa/timm). We used the CIFAR-10, CIFAR-100, ImageNet-1K, and [ImageNette](https://github.com/fastai/imagenette/) datasets to fine-train and evaluate our proposed models and the baselines. 

# Usage
Examples of training and evaluation scripts can be found in `train.py`:
```
python train.py --model=vit_small_patch16_224 --train_mode=pgd --batch_size=256 --epochs=40 --crop=32 --patch=4 --r_std=0.1  --mul_gpus
```
and `test.py`:
```
python test.py --model=vit_small_patch16_224 --batch_size=128 --crop=32 --patch=4 --mul_gpus --r_std=0.1 --test_mode=fgsm --load="" --eval
```

# Reference
If you find our paper/this repo useful for your research, please consider citing our work.
```
@inproceedings{gong2024random,
  title={Random Entangled Tokens for Adversarially Robust Vision Transformer},
  author={Gong, Huihui and Dong, Minjing and Ma, Siqi and Camtepe, Seyit and Nepal, Surya and Xu, Chang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24554--24563},
  year={2024}
}
```
