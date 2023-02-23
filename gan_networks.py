import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        input_dim=110
        output_dim=784
        self.label_embedding=nn.Embedding(10,10)
        self.hidden0=nn.Sequential(
            nn.Linear(input_dim,256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1=nn.Sequential(
            nn.Linear(256,512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2=nn.Sequential(
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2)
        )
        self.out=nn.Sequential(
            nn.Linear(1024,output_dim),
            nn.Tanh()
        )

    def forward(self,x,label):
        # print(f"Input Data shape is {x.shape}")
        c=self.label_embedding(label)
        # print(f"Label in Generator {label} and its shape is {label.shape}")
        # print(f"After Embedding label Shape is chaged to {c.shape}")
        x=torch.cat([x,c],1)
        x=self.hidden0(x)
        x=self.hidden1(x)
        x=self.hidden2(x)
        x=self.out(x)
        return x


class Descriminator(nn.Module):
    def __init__(self,):
        super(Descriminator,self).__init__()
        input_dim=794
        output_dim=1
        self.label_embedding=nn.Embedding(10,10)
        self.hidden0=nn.Sequential(
            nn.Linear(input_dim,1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1=nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2=nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out=nn.Sequential(
            nn.Linear(256,output_dim),
            nn.Sigmoid()
        )
        
    def forward(self,x,label):
        c=self.label_embedding(label)
        # print(c.shape)
        x=torch.cat([x,c],1)
        # print(x.shape)
        x=self.hidden0(x)
        x=self.hidden1(x)
        x=self.hidden2(x)
        x=self.out(x)
        return x
    

    
