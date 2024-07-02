import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

def plot_sample(real,fake,n = 5,output_path='./image'):
    _ , C, T = real.shape
    fig,axs = plt.subplots(nrows=n,ncols=2,figsize=(2*n,8))

    for j in range(n):
        fake_sample = np.random.randint(fake.shape[0])
        real_sample = np.random.randint(real.shape[0])
        for i in range(C):
            axs[j,0].plot(real[real_sample,i,:])
            axs[j,1].plot(fake[fake_sample,i,:])

    axs[0,0].set_title("Original")
    axs[0,1].set_title("Synthesize")
    fig.subplots_adjust(hspace=0.6)
    plt.savefig(os.path.join(output_path,"Sample.png"),bbox_inches='tight')
    plt.close(fig)


def plot_all_pca(real,fake,output_path='./image'):
    _ , C , T = real.shape
    fig = plt.figure(figsize=(8,6))
    pca = PCA(2)
    real_transform = pca.fit_transform(real.reshape(-1,C * T))
    fake_transform = pca.transform(fake.reshape(-1,C * T))
    plt.scatter(real_transform[:,0],real_transform[:,1],label="Real",s=5)
    plt.scatter(fake_transform[:,0],fake_transform[:,1],label="Fake",s=5)
    plt.legend()
    plt.savefig(os.path.join(output_path,"PCA.png"))
    plt.close(fig)


def plot_training_loss(losses,output_path):
    fig = plt.figure(figsize=(10,5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.savefig(os.path.join(output_path,"Loss.png"))
    plt.close(fig)