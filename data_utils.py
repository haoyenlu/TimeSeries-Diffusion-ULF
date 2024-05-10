import numpy as np
import torch

def build_dataloader(config,args):
    numpy_data = np.load(args.data,allow_pickle=True).item()
    if args.use_label:
        dataset = []
        for data,label in zip(numpy_data['data'],numpy_data['label']):
            dataset.append([data.tranpose(0,2,1),label])
    else:
        dataset = numpy_data['data'].transpose(0,2,1)
    dataloader = torch.utils.data.DataLoader(dataset,config['dataset']['batch_size'],shuffle=True)
    _, feat , seq_len = numpy_data['data'].shape
    assert seq_len == config['model']['params']['seq_length'] and config['model']['params']['feature_size'] == feat
    return dataloader
