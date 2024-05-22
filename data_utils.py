import numpy as np
import torch

def build_dataloader(config,args):
    numpy_data = np.load(args.data,allow_pickle=True).item()
    _ , feat , seq_len = numpy_data['data'].shape
    assert seq_len == config['model']['params']['seq_length'] and config['model']['params']['feature_size'] == feat, "Sequence Length or Feature Size doesn't match with config"

    if args.use_label:
        _ , label_dim = numpy_data['label'].shape
        assert label_dim == config['model']['params']['label_dim'], "Label Size doesn't match with config"

        dataset = []
        for data,label in zip(numpy_data['data'],numpy_data['label']):
            dataset.append([data.transpose(1,0),label.astype(float)])
    else:
        dataset = numpy_data['data'].transpose(0,2,1)

    dataloader = torch.utils.data.DataLoader(dataset,config['dataset']['batch_size'],shuffle=True)
    return dataloader
