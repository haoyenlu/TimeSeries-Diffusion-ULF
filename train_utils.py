import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from io_utils import instantiate_from_config

def cycle(dl):
    while True:
        for data,label in dl:
            yield data

class Trainer:
    def __init__(self,config,args,model,dataloader,logger=None):
        super().__init__()
        self.model = model
        self.device = model.betas.device
        self.train_num_epochs = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader)
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder,exist_ok=True)

        start_lr = config['solver'].get('base_lr',1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p:p.requires_grad,self.model.parameters()),lr=start_lr,betas=[0.9,0.96])
        self.ema = EMA(self.model,beta=ema_decay,update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        self.log_frequency = 100

    def save(self,milestone):
        data = {
            'step':self.step,
            'model':self.model.state_dict(),
            'ema':self.ema.state_dict(),
            'opt':self.opt.state_dict()
        }
        torch.save(data,os.path.join(self.results_folder,f'checkpoint-{milestone}.pt'))
    
    def load(self,milestone):
        device = self.device
        data = torch.load(os.path.join(self.results_folder,f'checkpoint-{milestone}.pt'),map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone
    

    def train(self):
        device = self.device
        step = 0

        # track time
        tic = time.time()

        print("Start Training")
        with tqdm(initial=step,total=self.train_num_epochs) as pbar:
            while step < self.train_num_epochs:
                total_loss = 0
                for _ in range(self.gradient_accumulate_every):
                    if self.args.use_label:
                        data, label = next(self.dl).to(device)
                    else:
                        data = next(self.dl).to(device)
                        label = None

                    loss = self.model(data,target=data,label=label)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()
                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(),1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                # logging and saving
                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                
                pbar.update(1)
        
        print("Training Complete","time:{:.2f}".format(time.time()-tic))

    def sample(self,config):
        samples = np.empty([0,config['model']['params']['seq_length'],config['model']['params']['feature_size']])
        num = config['dataset']['samples']['num_sample']
        size_every = config['dataset']['samples']['size_every']
        num_cycle = int(num // size_every) + 1

        for _ in tqdm(range(num_cycle)):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples , sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples
    