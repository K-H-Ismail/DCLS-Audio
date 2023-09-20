# Training

We provide AudioSet-2M training commands here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## Multi-node Training
We use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit) for producing the results and models in the paper. Please install:
```
pip install submitit
```
We will give example commands for both multi-node and single-machine training below.

## AudioSet-2M  Training 
ConvNeXt-DCLS-AUDIO-T training on AudioSet-2M with 4 8-GPU nodes:
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_dcls_audio_tiny --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--data_path /path/to/AudioSet-2M/hdf5s/waveforms/ \
--job_dir /path/to/save_results
```

- You may need to change cluster-specific arguments in `run_with_submitit.py`.
- You can add `--use_amp true` to train in PyTorch's Automatic Mixed Precision (AMP).
- Use `--resume /path_or_url/to/checkpoint.pth` to resume training from a previous checkpoint; use `--auto_resume true` to auto-resume from latest checkpoint in the specified output folder.
- `--batch_size`: batch size per GPU; `--update_freq`: gradient accumulation steps.
- The effective batch size = `--nodes` * `--ngpus` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `4*8*128*1 = 4096`. You can adjust these four arguments together to keep the effective batch size at 4096 and avoid OOM issues, based on the model size, number of nodes and GPU memory.

You can use the following command to run this experiment on a single machine: 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_dcls_audio_tiny --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--data_path /path/to/AudioSet-2M/hdf5s/waveforms/ \
--job_dir /path/to/save_results
```

- Here, the effective batch size = `--nproc_per_node` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `8*128*4 = 4096`. Running on one machine, we increased `update_freq` so that the total batch size is unchanged.

To train other models, `--model` and `--opt` need to be changed. Examples are given below, each with both multi-node and single-machine commands:

<details>
<summary>
ConvFormer-DCLS-AUDIO-S18
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convformer_dcls_audio_s18 --drop_path 0.4 \
--batch_size 64 --lr 4e-3 --update_freq 2 --opt lamb\
--data_path /path/to/AudioSet-2M/hdf5s/waveforms/ \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convformer_dcls_audio_s18 --drop_path 0.4 \
--batch_size 64 --lr 4e-3 --update_freq 2 --opt lamb\
--data_path /path/to/AudioSet-2M/hdf5s/waveforms/ \
--job_dir /path/to/save_results
```
</details>
<details>
<summary>
FastVit-DCLS-AUDIO-SA24
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model fastvit_dcls_audio_sa24 --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--data_path /path/to/AudioSet-2M/hdf5s/waveforms/ \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model fastvit_dcls_audio_sa24 --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--data_path /path/to/AudioSet-2M/hdf5s/waveforms/ \
--job_dir /path/to/save_results
``` 


``` 

</details>

