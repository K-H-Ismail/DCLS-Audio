# Installation

We provide installation instructions for AudioSet classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n convnext python=3.8 -y
conda activate convnext
```

Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. For example:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Clone this repo and install required packages:
```
git clone https://github.com/facebookresearch/ConvNeXt
pip install timm==0.3.2 tensorboardX six python-hostlist h5py dcls submitit
```

The results in the paper are produced with `torch==1.8.0+cu111 torchvision==0.9.0+cu111 timm==0.6.11`.

## Dataset Preparation

Download the [AudioSet-2M](http://?.org/)  dataset and structure the data as follows:
```
/path/to/audioset/hdf5s/waveforms/
├── balanced_train.h5 
├── eval.h5 
├── full_unbal_bal_train_wav.h5
```


