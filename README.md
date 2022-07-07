# Adversarial Robustness of SOTA Vision Transformers vs. CNNs

A fair comparision of the adversarial robustness of some relatively recent vision transformers (PoolFormer, Swin Transformer, DeiT) and CNNs (ConvNext, ResNet50). We try to algin the following:

* **model size**
* **pretrain dataset**
* **macro and micro architectural design**
* **training recipe**

For full details and discussion, please check [BLOG.md](BLOG.md).

## Adversarial Robustness Comparison on the ImageNet

| **Model**      | **Train Prec** | **Num Para** | **Clean Acc@1** | **PGD-5 Acc@1** | **Relative Drop** |
|----------------|:-------:|:----------:|:---------------:|:---------------:|:-------------------:|
| convnext-tiny  |   bf16  |     29M    |      69.98%     |      49.95%     |         -29%        |
| swin-tiny      |   bf16  |     28M    |      67.24%     |      46.66%     |         -31%        |
| poolformer-s36 |   bf16  |    30.8M   |      66.32%     |      44.80%     |         -32%        |
| deit-small     |   bf16  |     22M    |      66.48%     |      44.60%     |         -33%        |
| deit-small*    |    full   |     22M    |      66.50%     |      43.95%     |         -34%        |
| resnet50-gelu* |    full   |     25M    |      67.38%     |      44.01%     |         -35%        |

model* denotes model results reported in the previous work.

## Installation
```
conda create -n advtrans python=3.8

conda activate advtrans

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install tensorboardx timm wandb
```

## Adversarial Training and Evaluation Cmd
Upcoming.

### Single-node multi-GPU distributed training

### Multi-node multi-GPU distributed training with SLURM

### Distributed mixed precision training with Google Brain's `bf16`


## Acknowledgement
Our implementation is based on the [ConvNext](https://github.com/facebookresearch/ConvNeXt) and [timm](https://github.com/rwightman/pytorch-image-models) repositories.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@unpublished{xu2022AdvTransCNNs,
  author = {Ke Xu and Ram Nevatia},
  title  = {Adversarial Robustness of SOTA Vision Transformers vs. CNNs},
  month  = "May",
  year   = {2022},
}
```
