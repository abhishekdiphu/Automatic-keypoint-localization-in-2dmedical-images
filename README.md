
# AdversarialPoseNet-2DMedical
Investigation of adversarial learning for landmark localization in 2D medical images (lower extrimites)


Pytorch implementation of chen et al. "Adversarial PoseNet" for landmark localization on medical data.
The method was  proposed by [Yu Chen, Chunhua Shen, Xiu-Shen Wei, Lingqiao Liu, Jian Yang](https://scholar.google.com/citations?user=IWZubqUAAAAJ&hl=zh-CN) in 
[Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389). 



## Lanmark localization 
<img src="readmeimages/inputs01.png" width="200px"/>



##  Results Visualization
The results of this implementation:

- Adversarial PoseNet<br>
<img src="testresults-1/Adversarial-1/results_116.png" width="500px"/> <img src="testresults-1/Adversarial-1/results_117.png" width="500px"/> <img src="testresults-1/Adversarial-1/results_118.png" width="500px"/> 

- Stack-hour-glass Network(supervised setup)<br>
<img src="testresults-1/baseline-1/results_117.png" width="500px"/> 
<img src="testresults-1/baseline-1/results_118.png" width="500px"/> 
<img src="testresults-1/baseline-1/results_119.png" width="500px"/> 



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


- `lower leg datasets`: The dataset includes around 660 2D medical images. They are grayscale in nature. each lower leg radiograph image in the dataset has been labelled with 6 joint co-ordinate, one each for distinct epiphyses parts ( bone extremities ) femur, ankle , knee for both the legs. The size of each of the image is approximately 7300 x 2800 pixels. Here in this resolution, isotopic range is .143mm = 1 px.

<img src="readmeimages/lowerleg_greyscale.png" width="300px"/><img src="readmeimages/lowerleg_annotated.png" width="300px"/>


## Reference
- The torch implementation of CycleGAN, https://github.com/junyanz/CycleGAN
- The tensorflow implementation of pix2pix, https://github.com/yenchenlin/pix2pix-tensorflow
