# MindSpore Twins

## Introduction

This work is used for reproduce Twins based on NPU(Ascend 910)

**Twins** is introduced in [arxiv](https://arxiv.org/abs/2104.13840)

Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully-devised yet simple spatial attention mechanism performs favourably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins- PCPVT and Twins-SVT. Our proposed architectures are highly-efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks including image- level classification as well as dense detection and segmentation.

Twins achieves strong performance on ImageNet classification (81.2 on val)

![framework](/figures/twins_svt_s.png)

## Data preparation

Download and extract [ImageNet](https://image-net.org/).

The directory structure is the standard layout for the MindSpore [`dataset.ImageFolderDataset`](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/dataset/mindspore.dataset.ImageFolderDataset.html?highlight=imagefolderdataset), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
## Training


```
mpirun -n 8 python train.py --config <config path> > train.log 2>&1 &
```

## Evaluation 


```
python eval.py --config <config path>
```


## Acknowledgement

We heavily borrow the code from [Twins](https://github.com/Meituan-AutoML/Twins) and [swin_transformer](https://gitee.com/mindspore/models/tree/master/research/cv/swin_transformer)
We thank the authors for the nicely organized code!
