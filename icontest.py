import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
#quick tests, no fun comments this time

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Icons" / "best_icon_model.pth"

IMAGE_PATH = BASE_DIR / "minecraft__blue_stained_glass.png"  # change this


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load item labels
with open(BASE_DIR / "Icons" / "item_labels.json", "r") as f:
    item_labels = json.load(f)


# Recreate same mapping as training
item_to_idx = {}
idx_to_item = {}

idx = 0

for category, items in item_labels["classes"].items():
    for item_id in items:
        item_to_idx[item_id] = idx
        idx_to_item[idx] = item_id
        idx += 1


# Same model architecture
class IconCNN(nn.Module):
    def __init__(self, num_classes):
        super(IconCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x



# Create model
model = IconCNN(len(item_to_idx)).to(device)


# Load trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Loaded model")
print("Classes:", len(item_to_idx))


# Same preprocessing as training
# Define transform 
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor()
])

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = transform(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0).to(device)


# Prediction
with torch.no_grad():
    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)


predicted_idx = prediction.item()

predicted_item = idx_to_item[predicted_idx]


print("Prediction:", predicted_item)
print("Confidence:", f"{confidence.item()*100:.2f}%")

# Show top 5 guesses
top5_conf, top5_idx = torch.topk(probabilities, 5)

print("\nTop 5 predictions:")

for conf, idx in zip(top5_conf[0], top5_idx[0]):
    print(
        idx_to_item[idx.item()],
        f"{conf.item()*100:.2f}%"
    )