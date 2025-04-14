print("Loading Modules.....")
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as fn
import neuralNetworkDetection as net
from torchvision.utils import make_grid

import numpy as np
print("Modules Loaded")

MODEL_PATH = "Model/Resnet_reduced.pth"
DEVICE = "cuda"

BATCH_SIZE = 9

IMAGE_HEIGHT,IMAGE_WIDTH,COLOR_CHANNELS = 64,64,3

data_transform = transforms.Compose([transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])



gen = net.ResNet(net.block,[1, 2, 3],3,19).to(DEVICE)
gen.load_state_dict(torch.load(MODEL_PATH,weights_only=True))

plt.figure(figsize=(10,10))
gen.eval()


#Generated Images
with torch.no_grad():
    z = torch.randn(49, ZDIM,1,1)
    generated = gen(z).detach().cpu()
    print(generated.shape)
    grid = make_grid(generated,nrow=7, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    img = grid
    tran = fn.ToPILImage()
    img = tran(img)
    img.save("Test.jpg")
    plt.axis("off")
    plt.show()
