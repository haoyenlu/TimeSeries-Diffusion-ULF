import os
import torch
import argparse
import numpy as np
import csv

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
    parser.add_argument('--milestone',type=int,default=None)
    parser.add_argument('--use_label',action='store_true')
    parser.add_argument('--analyze',action="store_true")
    parser.add_argument('--image_path',type=str,default='./images')
    parser.add_argument('--num',type=int,default=5)
    parser.add_argument('--csv',type=str,default=None)

    args = parser.parse_args()

    return args

def main():
    args = parse_argument()

    seed = np.random.randint(0,99999)
    seed_everything(seed)
    
    os.makedirs(args.output,exist_ok=True)
    config = load_yaml_config(args.config)
    model = instantiate_from_config(config['model'],configs=config['model']).cuda()


    trainer = Trainer(config=config,args=args,model=model)

    if args.milestone:
        trainer.load(args.milestone)

    if args.train:
        dataloader = build_dataloader(config,args)
        trainer.train(dataloader)


    if args.sample:
        trainer.sample(config,output_path=args.output)

        if args.analyze:
            trainer.export_to_csv(args.csv)
            trainer.export_analysis(args.data,args.output,args.num,args.image_path)
        


            
if __name__ == '__main__':
    main()
