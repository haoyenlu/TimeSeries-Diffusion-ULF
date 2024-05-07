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
    parser.add_argument('--output',type=str,help="Save Checkpoint Directory",default='./ckpt')
    parser.add_argument('--sample',type=str,default='./samples')
    parser.add_argument('--tensorboard',action='store_true')
    parser.add_argument('--mode',type=str)
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--task',type=str,default='T01')
    parser.add_argument('--milestone',type=int,default=10)

    args = parser.parse_args()

    return args

def main():
    args = parse_argument()

    seed_everything(args.seed)

    os.makedirs(args.output,exist_ok=True)
    os.makedirs(args.sample,exist_ok=True)

    config = load_yaml_config(args.config)
    
    model = instantiate_from_config(config['model']).cuda()

    dataloader = build_dataloader(config,args)
    trainer = Trainer(config=config,args=args,model=model,dataloader=dataloader)

    if args.mode == 'train':
        trainer.train()

    elif args.mode == 'sample':
        trainer.load(args.milestone)
        sample = trainer.sample(num=config['dataset']['num_sample'],size_every = config['dataset']['size_every'])
        np.save(os.path.join(args.sample,f"ddpm_{args.task}.npy",sample))

if __name__ == '__main__':
    main()
