# Time-series Diffusion model for stroke motion generation

## Argument
```
--data: Training data path
--config: Training config
--output: Sampling output folder
--train: Train or not
--sample: Sample or not
--milestone: Checkpoint milestone
--use_label: Use conditional label or not
--analyze: Analyze with PCA and Plotting or not
--image_path: Plot output folder
--num: Plot number for random sample
--csv: record the performance to csv file
``` 

## Config Params
### Diffusion Model
```
seq_length: sequence length
feature_size: data channel size
timesteps: Diffusion training timestep
sampling_timestep: Diffusion sampling timestep
loss_type: 'l1' or 'l2'
beta_schedule: 'linear' or 'cosine' for timestep schedule
```
### Transformer Model
```
n_layer_enc: number of layer for encoder
n_layer_dec: number of layer for decoder
d_model: hidden model for transformer
n_head: number of head for attention
mlp_hidden_times: transformer MLP output ratio related to d_model
attn_pdrop: Attention dropout probability
resid_pdrop: Attention MLP dropout probability
kernel_size: kernel size for CNN
padding_size: padding size for CNN
```

##### Referrences:
Original author: [https://github.com/Y-debug-sys/Diffusion-TS/tree/main]
```
@article{yuan2024diffusion,
  title={Diffusion-TS: Interpretable Diffusion for General Time Series Generation},
  author={Yuan, Xinyu and Qiao, Yan},
  journal={arXiv preprint arXiv:2403.01742},
  year={2024}
}
```


