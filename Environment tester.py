import torch
from torch import nn
from pathlib import Path
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

class simpleUNet(nn.Module):
    def __init__(self, num_classes):
        super(simpleUNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "best_minecraft_model.pth",
    map_location=device
)


num_classes = checkpoint["num_classes"]

model = simpleUNet(num_classes)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()


print("Loaded model")
print("Classes:", num_classes)


image_path = "frame_00768.png"

image = Image.open(image_path).convert("RGB")

transform = transforms.ToTensor()

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0) # add batch dimension

image_tensor = image_tensor.to(device)


with torch.no_grad():

    output = model(image_tensor)

    # output:
    # [batch, classes, height, width]

    prediction = torch.argmax(
        output,
        dim=1
    )

prediction = prediction.squeeze(0).cpu().numpy()


print(prediction.shape)
print(np.unique(prediction))



# -------------------
# Display prediction
# -------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Input")
plt.axis("off")


plt.subplot(1,2,2)
plt.imshow(prediction)
plt.title("Prediction train IDs")
plt.axis("off")


plt.show()