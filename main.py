import os
import torch
import argparse
import numpy as np

from train_utils import Trainer
from data_utils import build_dataloader
from io_utils import load_yaml_config, seed_everything, instantiate_from_config


def parse_argument():
    parser = argparse.ArgumentParser(description='Pytorch Training Script')
    parser.add_argument('--data',type=str,default=None)
    parser.add_argument('--config',type=str,default=None)
    parser.add_argument('--output',type=str,default='./samples')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--sample',action='store_true')
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--milestone',type=int,default=None)
    parser.add_argument('--use_label',action='store_true')

    args = parser.parse_args()

    return args

def main():
    args = parse_argument()

    seed_everything(args.seed)

    os.makedirs(args.sample,exist_ok=True)

    config = load_yaml_config(args.config)
    
    model = instantiate_from_config(config['model'],configs=config['model']).cuda()


    trainer = Trainer(config=config,args=args,model=model)

    if args.milestone:
        trainer.load(args.milestone)

    if args.train:
        dataloader = build_dataloader(config,args)
        trainer.train(dataloader)

    if args.sample:
        output_name = f'synthesize_{config['model']['backbone']['params']['n_layer_enc']}_{config['model']['backbone']['params']['n_layer_dec']}_{config['model']['backbone']['params']['d_model']}'
        if args.use_label:
            samples,labels = trainer.sample(config)
            output = {'data':samples,'label':labels}
        
        else:
            samples = trainer.sample(config)
            output = {'data':samples}

        np.save(os.path.join(args.sample,f"{output_name}.npy"),output)
            
if __name__ == '__main__':
    main()
