<br><br><br>
-->

# AdversarialPoseNet-2DMedical
Investigation of adversarial learning for landmark localization in 2D medical images (lower extrimites)


Tensorflow implementation for learning an image-to-image translation **without** input-output pairs.
The method is proposed by [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/) in 
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkssee](https://arxiv.org/pdf/1703.10593.pdf). 
For example in paper:

<img src="imgs/teaser.jpg" width="1000px"/>

<!--
## Applications
### Monet Paintings to Photos
<img src="imgs/painting2photo.jpg" width="1000px"/>

### Collection Style Transfer
<img src="imgs/photo2painting.jpg" width="1000px"/>

### Object Transfiguration
<img src="imgs/objects.jpg" width="1000px"/>

### Season Transfer
<img src="imgs/season.jpg" width="1000px"/>

### Photo Enhancement: iPhone photo to DSLR photo
<img src="imgs/photo_enhancement.jpg" width="1000px"/>

-->

## Update Results
The results of this implementation:

- Horses -> Zebras <br>
<img src="imgs/n02381460_510.jpg" width="200px"/> <img src="imgs/AtoB_n02381460_510.jpg" width="200px"/> <img src="imgs/n02381460_4530.jpg" width="200px"/> <img src="imgs/AtoB_n02381460_4530.jpg" width="200px"/> <img src="imgs/n02381460_4660.jpg" width="200px"/> <img src="imgs/AtoB_n02381460_4660.jpg" width="200px"/> <img src="imgs/n02381460_8980.jpg" width="200px"/> <img src="imgs/AtoB_n02381460_8980.jpg" width="200px"/>

- Zebras -> Horses <br>
<img src="imgs/n02391049_1760.jpg" width="200px"/> <img src="imgs/BtoA_n02391049_1760.jpg" width="200px"/> <img src="imgs/n02391049_3070.jpg" width="200px"/> <img src="imgs/BtoA_n02391049_3070.jpg" width="200px"/> <img src="imgs/n02391049_5100.jpg" width="200px"/> <img src="imgs/BtoA_n02391049_5100.jpg" width="200px"/> <img src="imgs/n02391049_7150.jpg" width="200px"/> <img src="imgs/BtoA_n02391049_7150.jpg" width="200px"/>

You can download the pretrained model from [this url](https://1drv.ms/u/s!AroAdu0uts_gj5tA93GnwyfRpvBIDA)
and extract the rar file to `./checkpoint/`.


## Prerequisites
- tensorflow r1.1
- numpy 1.11.0
- scipy 0.17.0
- pillow 3.3.0

## Getting Started
### Installation
- Install tensorflow from https://github.com/tensorflow/tensorflow
- Clone this repo:
```bash
git clone https://github.com/xhujoy/CycleGAN-tensorflow
cd CycleGAN-tensorflow
```

### Train
- Download a dataset (e.g. zebra and horse images from ImageNet):
```bash
bash ./download_dataset.sh horse2zebra
```
- Train a model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=horse2zebra
```
- Use tensorboard to visualize the training details:
```bash
tensorboard --logdir=./logs
```

### Test
- Finally, test the model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=horse2zebra --phase=test --which_direction=AtoB
```

## Training and Test Details
To train a model,  
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=/path/to/data/ 
```
Models are saved to `./checkpoints/` (can be changed by passing `--checkpoint_dir=your_dir`).  

To test the model,
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=/path/to/data/ --phase=test --which_direction=AtoB/BtoA
```

## Datasets
Download the datasets using the following script:

- `lower leg datasets`: 750 2D X-RAYS of lower legs. 


## Reference
- The torch implementation of CycleGAN, https://github.com/junyanz/CycleGAN
- The tensorflow implementation of pix2pix, https://github.com/yenchenlin/pix2pix-tensorflow