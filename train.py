# Load all the libraries and Frameworks
import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import optim as optim
from tqdm import tqdm
from gan_networks import Generator, Descriminator
from utils import SaveBestModel, save_model

# Tensorboard writer
writer=SummaryWriter(log_dir='./runs/exp')

# Device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running the model on the {device}")

# Hyperparameters
EPOCHS=100
lr=0.0002
BATCH=100
PATH1=os.path.join("Models","best_gen_model.pth")
PATH2=os.path.join("Models","last_gen_model.pth")

# Data Transformation
transform=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ]
)

# Data Loader
data_loader=DataLoader(
    datasets.FashionMNIST('./',train=True,download=True,
    transform=transform),
    batch_size=BATCH,shuffle=True
)
print("Data loadeder created")

# Model
generator=Generator().to(device)
descriminator=Descriminator().to(device)
print("Generator and Descriminator created")

# Loss and Optimizer
loss=nn.BCELoss()
generator_optim=optim.Adam(generator.parameters(),lr=lr)
descriminator_optim=optim.Adam(descriminator.parameters(),lr=lr)
print("Loss and Optimizer created")

save_best_model=SaveBestModel(path=PATH1,metric=float('inf'),mode="min")

# Training Loop
generator.train()
descriminator.train()
print(f"Training started......")
for epoch in range(EPOCHS):
    G_loss=[]
    D_loss=[]
    for batch_idx, (real,labels) in enumerate(data_loader):
        print(f"Batch Index is {batch_idx} with epoch {epoch}")
        real=real.view(-1,784).to(device)
        current_batch_size=real.shape[0]
        con_labels=labels.to(device)
        true_out_labels=torch.ones(current_batch_size).to(device)
        # print(real.shape)
        # print(con_labels.shape)
        # print(true_out_labels.shape)
        noise=torch.randn(current_batch_size,100).to(device)
        con_fake_labels=torch.randint(0,10,(current_batch_size,)).to(device)
        fake_out_labels=torch.zeros_like(true_out_labels).to(device)
        # print(noise.shape)
        # print(con_fake_labels.shape)
        # print(fake_out_labels.shape)

        # Generated Data
        fake=generator(noise,con_fake_labels)
        # print(fake.shape)

        # Descriminator operation
    
        if fake.shape!=real.shape:
            print("Shape of true data and fake data is not same")
            exit()
        
        descriminator_optim.zero_grad()
        des_out_for_true=descriminator(real,con_labels).view(BATCH)
        des_loss_for_true=loss(des_out_for_true,true_out_labels)
        des_out_for_fake=descriminator(fake.detach(),con_fake_labels).view(BATCH)
        des_loss_for_fake=loss(des_out_for_fake,fake_out_labels)
        final_des_loss=(des_loss_for_true+des_loss_for_fake)/2
        final_des_loss.backward()
        descriminator_optim.step()
        D_loss.append(final_des_loss.data.item())

        # Generator Operation
        generator_optim.zero_grad()
        generated_data=generator(noise,con_fake_labels)
        des_out_for_generated=descriminator(generated_data,con_fake_labels).view(BATCH)
        final_gen_loss=loss(des_out_for_generated,true_out_labels)
        final_gen_loss.backward()
        generator_optim.step()
        G_loss.append(final_gen_loss.data.item())

        if ((batch_idx+1)%50==0) and ((epoch+1)%2==0):
            print(f"Epoch {epoch} Batch {batch_idx} Generator Loss {final_gen_loss.data.item()} Descriminator Loss {final_des_loss.data.item()}")
            with torch.no_grad():
                noise=torch.randn(64,100).to(device)
                con_fake_labels=torch.randint(0,10,(64,)).to(device)
                fake=generator(noise,con_fake_labels)
                fake=fake.view(64,1,28,28)
                img_grid=torchvision.utils.make_grid(fake)
                writer.add_image(f'Fake Images{epoch+1}',img_grid)
                writer.add_scalar('Generator Loss',final_gen_loss.data.item(),epoch)
                writer.add_scalar('Descriminator Loss',final_des_loss.data.item(),epoch)

    print(f"Saving the model if it is best model")
    save_best_model(model=generator,epoch=epoch, optimizer=generator_optim ,criterion=loss , metric=torch.mean(torch.tensor(G_loss)))
    print("*"*50)


print("Training completed")
print("Saving the last model")
save_model(path=PATH2, model=generator , optimizer=generator_optim , criterion=loss,epoch=EPOCHS+1)
print("Model saved")

print(f"Final mean value of the G_loss", torch.mean(torch.tensor(G_loss)))
print(f"Final mean value of the D_loss", torch.mean(torch.tensor(D_loss)))
