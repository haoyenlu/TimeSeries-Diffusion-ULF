import numpy as np
import torch

def build_dataloader(config,args):
    numpy_data = np.load(args.data,allow_pickle=True).item()
    train_data = numpy_data['data'].transpose(0,2,1)
    dataloader = torch.utils.data.DataLoader(train_data,config['dataset']['batch_size'],shuffle=True)
    _, seq_len , feat = train_data.shape
    assert seq_len == config['model']['params']['seq_length'] and config['model']['params']['feature_size'] == feat
    return dataloader
