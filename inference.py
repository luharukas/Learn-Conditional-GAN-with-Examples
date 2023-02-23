import torch
import torch.nn as nn
from gan_networks import Generator
import matplotlib.pyplot as plt

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running the model on the {device}")
best_model=torch.load("Models/best_gen_model.pth")
generator=Generator().to(device)
generator.load_state_dict(best_model['model'])
print("Generator loaded")
generator.eval()
while True:
    input_number=int(input("Enter the number "))
    if input_number>9 or input_number<0:
        print("Enter a valid number between 0 and 9")
        continue
    input_noise=torch.randn(1,100).to(device)
    input_label=torch.tensor([input_number]).to(device)
    output=generator(input_noise,input_label).detach().cpu().numpy().reshape(28,28)
    plt.imshow(output,cmap='gray')
    plt.show()
    print("Done")





