import os
import time
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from io_utils import instantiate_from_config

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer:
    def __init__(self,config,args,model,logger=None):
        super().__init__()
        self.config = config
        self.model = model
        self.device = model.betas.device
        self.train_num_epochs = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.step = 0
        self.milestone = 0
        self.use_label = args.use_label
        self.history = {'loss':[],'time':0}
        self.id = time.strftime("%Y_%m_%d_%H_%M",time.gmtime())

        self.results_folder = Path(config['solver']['results_folder'] + f'_{self.id}')
        os.makedirs(self.results_folder,exist_ok=True)

        start_lr = config['solver'].get('base_lr',1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p:p.requires_grad,self.model.parameters()),lr=start_lr,betas=[0.9,0.96])
        self.ema = EMA(self.model,beta=ema_decay,update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

    def save(self,milestone):
        data = {
            'step':self.step,
            'model':self.model.state_dict(),
            'ema':self.ema.state_dict(),
            'opt':self.opt.state_dict()
        }
        torch.save(data,os.path.join(self.results_folder,f'checkpoint-{milestone}.pt'))
    
    def load(self,milestone):
        print(f"Loading milestone {os.path.join(self.results_folder,f'checkpoint-{milestone}.pt')}")
        device = self.device
        data = torch.load(os.path.join(self.results_folder,f'checkpoint-{milestone}.pt'),map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone
    

    def train(self,dataloader):
        device = self.device
        step = 0
        dl = cycle(dataloader)
        # track time
        tic = time.time()

        print("Start Training..")
        with tqdm(initial=step,total=self.train_num_epochs) as pbar:
            while step < self.train_num_epochs:
                total_loss = 0
                for _ in range(self.gradient_accumulate_every):
                    if self.use_label:
                        temp = next(dl)
                        data, label = temp[0].float().to(device), temp[1].float().to(device)
                    else:
                        data = next(dl).float().to(device)
                        label = None

                    loss = self.model(data,target=data,label=label)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()
                pbar.set_description(f'loss: {total_loss:.6f}')
                self.history['loss'].append(total_loss)


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
        
        train_time = time.time() - tic
        print("Training Complete","time:{:.2f}".format(train_time))
        self.history['time'] = train_time

    def sample(self,config,output_path='./samples'):
        print("Start Sampling..")
        samples = np.empty([0,config['model']['params']['seq_length'],config['model']['params']['feature_size']])
        labels = np.empty([0,config['model']['params']['label_dim']])
        num = config['dataset']['samples']['num_sample']
        size_every = config['dataset']['samples']['size_every']
        num_cycle = int(num // size_every)


        for _ in tqdm(range(num_cycle)):
            if self.use_label:
                sample,label = self.ema.ema_model.generate_mts(batch_size=size_every,use_label=True)
                samples = np.row_stack([samples , sample.detach().cpu().numpy()])
                labels = np.row_stack([labels,label.detach().cpu().numpy()])
            else:
                sample = self.ema.ema_model.generate_mts(batch_size=size_every,use_label=False)
                samples = np.row_stack([samples , sample.detach().cpu().numpy()])

            torch.cuda.empty_cache()

        if self.use_label:
            output = {'data':samples.transpose(0,2,1),'label':labels}
        else:
            output = {'data':samples.transpose(0,2,1)}

        np.save(os.path.join(output_path,f"{self.id}.npy"),output)
    

    def export_to_csv(self,csvPath):
        fields = self.config['csv_fields']
        data = self.config['model']['backbone']['params']
        data.update(self.config['model']['params'])
        data.update({'loss':self.history['loss'][-1],'time':self.history['time']})
        data.update({'lr':self.config['solver']['base_lr'],'epoch':self.train_num_epochs})
        with open(os.path.join(csvPath,f'{self.id}.csv'),'w') as file:
            writer = csv.DictReader(file,fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    
    def export_analysis(self,real_path,fake_path,num,image_path):
        ''' Plot Real and Fake sample '''
        image_path = os.path.join(image_path,self.id)
        os.makedirs(image_path,exist_ok=True)

        real = np.load(real_path,allow_pickle=True).item()['data']
        fake = np.load(os.path.join(fake_path,f"{self.id}.npy"),allow_pickle=True).item()['data']

        _ , C, T = real.shape
        fig,axs = plt.subplots(nrows=num,ncols=2,figsize=(2*num,8))

        for j in range(num):
            fake_sample = np.random.randint(fake.shape[0])
            real_sample = np.random.randint(real.shape[0])
            for i in range(C):
                axs[j,0].plot(real[real_sample,i,:])
                axs[j,1].plot(fake[fake_sample,i,:])

        axs[0,0].set_title("Original")
        axs[0,1].set_title("Synthesize")
        fig.subplots_adjust(hspace=0.6)
        plt.savefig(os.path.join(image_path,"Sample.png"),bbox_inches='tight')
        plt.close(fig)

        ''' Plot PCA '''
        fig = plt.figure(figsize=(8,6))
        pca = PCA(2)
        real_transform = pca.fit_transform(real.reshape(-1,C * T))
        fake_transform = pca.transform(fake.reshape(-1,C * T))
        plt.scatter(real_transform[:,0],real_transform[:,1],label="Real",s=5)
        plt.scatter(fake_transform[:,0],fake_transform[:,1],label="Fake",s=5)
        plt.legend()
        plt.savefig(os.path.join(image_path,"PCA.png"),bbox_inches='tight')
        plt.close(fig)

        ''' Plot Training Loss '''
        fig = plt.figure(figsize=(10,5))
        plt.plot(self.history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Losses')
        plt.savefig(os.path.join(image_path,"Loss.png"),bbox_inches='tight')
        plt.close(fig)
                

    