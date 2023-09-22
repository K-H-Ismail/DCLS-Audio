# [Audio classification with Dilated Convolution with Learnable Spacings](https://arxiv.org/abs/?)

Official PyTorch implementation from the following paper:

[Audio classification with Dilated Convolution with Learnable Spacings](https://arxiv.org/abs/?). \
by Ismail Khalfaoui Hassani, Thomas Pellegrini and Timothée Masquelier.

--- 


## Catalog
- [ ] AudioSet-2M dataset
- [x] AudioSet-2M Training Code  
- [x] Pre-trained models on AudioSet-2M



<!-- ✅ ⬜️  -->

## Results and Pre-trained Models
### AudioSet-2M trained models

| Model @ 128x1001 | Kernel Size / Count | Method        | # Parameters | mAP             | Throughput (sample/s) | model |
|------------------|----------------------|---------------|--------------|-----------------|------------------------| ---------|
| ConvFormer-S18†  | 7x7 / 49             | Depth. Conv.  | 26.8M        | 43.14 ± 0.03    | 513.3                  | [model](https://zenodo.org/record/8370979/files/convformer_audio_s18.pth) |
| ConvFormer-S18†  | 23x23 / 26           | DCLS-Gauss   | 26.8M        | 43.68 ± 0.02    | 396.8                  | [model](https://zenodo.org/record/8370979/files/convformer_dcls_audio_s18.pth) |
| FastVIT-SA24‡    | 7x7 / 49             | Depth. Conv.  | 21.5M        | 43.82 ± 0.05    | 633.6                  | [model](https://zenodo.org/record/8370979/files/fastvit_dcls_audio_sa24.pth) |
| FastVIT-SA24‡    | 23x23 / 26           | DCLS-Gauss   | 21.5M        | 44.4 ± 0.07      | 551.7                  | [model](https://zenodo.org/record/8370979/files/fastvit_dcls_audio_sa24.pth) |
| ConvNeXt-T       | 7x7 / 49             | Depth. Conv.  | 28.6M        | 44.83 ± 0.14    | 591.4                 | [model](https://zenodo.org/record/8370979/files/convnext_dcls_audio_tiny.pth) |
| ConvNeXt-T       | 23x23 / 26           | DCLS-Gauss   | 28.6M        | 45.52 ± 0.05    | 509.4                  | [model](https://zenodo.org/record/8370979/files/convnext_dcls_audio_tiny.pth) |

† Trained using LAMB, ‡ No ImageNet pretraining.



## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation
We give an example evaluation command for an  AudioSet-2M pre-trained FastVit-DCLS-AUDIO-SA24:

Single-GPU
```
python main.py --model fastvit_dcls_audio_sa24 --eval true \
--resume https://zenodo.org/record/8370979/files/fastvit_dcls_audio_sa24.pth \
--drop_path 0.4 \
--data_path /path/to/AudioSet-2M/hdf5s/waveforms/ 
```


Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_dcls_audio_tiny --eval true \
--resume https://zenodo.org/record/8370979/files/fastvit_dcls_audio_sa24.pth \
--drop_path 0.4 \
--data_path /path/to/AudioSet-2M/hdf5s/waveforms/
```

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository was inspired by [ConvNeXt](https://github.com/facebookresearch/ConvNeXt).

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```

```
