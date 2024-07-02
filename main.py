import os
import torch
import argparse
import numpy as np

from train_utils import Trainer
from data_utils import build_dataloader
from io_utils import load_yaml_config, seed_everything, instantiate_from_config

from analyze import plot_sample, plot_all_pca


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

    args = parser.parse_args()

    return args

def main():
    args = parse_argument()

    seed = np.random.randint(0,99999)
    seed_everything(seed)
    print(f"Use Seed: {seed}")
    
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
        output_name = f"synthesize_{config['model']['backbone']['params']['n_layer_enc']}_{config['model']['backbone']['params']['n_layer_dec']}_{config['model']['backbone']['params']['d_model']}"
        if args.use_label:
            samples,labels = trainer.sample(config)
            output = {'data':samples.transpose(0,2,1),'label':labels}
        
        else:
            samples = trainer.sample(config)
            output = {'data':samples.transpose(0,2,1)}

        np.save(os.path.join(args.output,f"{output_name}.npy"),output)
        
        if args.analyze:
            real = np.load(args.data,allow_pickle=True).item()
            plot_sample(real['data'],output['data'],n=args.num,output_path=args.image_path)
            plot_all_pca(real['data'],output['data'],output_path=args.image_path)
            

            
if __name__ == '__main__':
    main()
