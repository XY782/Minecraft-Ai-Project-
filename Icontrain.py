import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
import json

#Set path for the base directory
BASE_DIR = Path(__file__).resolve().parent
ICON_DIR = BASE_DIR / "Icons"
TRAIN_PNG_DIR = ICON_DIR / "train_data" / "PNG"
TRAIN_LABELS_DIR = ICON_DIR / "train_data" / "metadata"
TEST_PNG_DIR = ICON_DIR / "test_data"
VAL_PNG_DIR = ICON_DIR / "val_data"
BEST_ICON_PATH = ICON_DIR / "best_icon_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

item_labels = json.load(open(BASE_DIR / "Icons" / "item_labels.json", "r"))

# #Debug to check is path correct and its shape + mode
# train_images = sorted(TRAIN_PNG_DIR.glob("*.png"))
# print(f"Path for first image: {train_images[0]}")
# first_image = Image.open(train_images[0])
# print(f"First image size: {first_image.size}")
# print(f"First image mode: {first_image.mode}")
# IT WORKS LETS GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

# Maps item_id to index for easier label handling
# SO damn annoying
item_to_idx = {}

idx = 0

for category, items in item_labels["classes"].items():
    for item_id in items:
        item_to_idx[item_id] = idx
        idx += 1

# Lets create a custom dataset class for our icon dataset
class IconDataset (torch.utils.data.Dataset):
    def __init__(self, png_dir, labels_dir, transform=None):
        self.png_dir = png_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.png_files = sorted(list(png_dir.glob("*.png")))
        
    def __len__(self):
        return len(self.png_files)
    
    def __getitem__(self, idx):
        png_file = self.png_files[idx]

        image = Image.open(png_file).convert("RGB")

        if self.labels_dir is not None:
            label_file = self.labels_dir / (png_file.stem + ".json")
            with open(label_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            item_id = label_data["item_id"]
        else:
            if "__" in png_file.stem:
                namespace, name = png_file.stem.split("__", 1)
                item_id = f"{namespace}:{name}"
            else:
                item_id = png_file.stem

        if item_id not in item_to_idx:
            print("Missing item:", item_id)
            print("Available examples:", list(item_to_idx.keys())[:10])
            raise KeyError(item_id)

        label = item_to_idx[item_id]

        if self.transform:
            image = self.transform(image)

        return image, label

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
#print(f"Datset size: {len(train_dataset)}")

batch_size = 32

train_dataset = IconDataset(TRAIN_PNG_DIR, TRAIN_LABELS_DIR, transform=transform)
val_dataset = IconDataset(VAL_PNG_DIR, None, transform=transform)
test_dataset = IconDataset(TEST_PNG_DIR, None, transform=transform)

# obviously, a dataloader 
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

image, label = train_dataset[0]

# Get the metadata for this image
png_file = train_dataset.png_files[0]
label_file = TRAIN_LABELS_DIR / (png_file.stem + ".json")

# # Debug to check if the label file is correct
# with open(label_file, "r", encoding="utf-8") as f:
#     metadata = json.load(f)
# print(list(item_to_idx.items())[:3])


#CNN model
class IconCNN(nn.Module):
    def __init__(self, num_classes):
        super(IconCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Adjusted for input size of 32x32
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
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

num_classes = len(item_to_idx)

model = IconCNN(num_classes).to(device)
print("Device:", device)
print("Model device:", next(model.parameters()).device)
# Debug to check the model architecture and output shape
# print(f"Model architecture:\n{model}")
# images, labels = next(iter(train_loader))
# images = images.to(device)
# outputs = model(images)
# print(outputs.shape)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay = 1e-4)

epochs = 1000

best_val_loss = float("inf")

# I think I have way less comments in this file than the previous one
# But eh who cares do you guys even read these

#Defines the evaluation function for the model
# it returns the average loss, top-1 accuracy, and top-5 accuracy for the given data loader.
# made this since the loss was stuck at 0.45-0.55 and I wanted to see if the model was actually learning or not
def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_top1_correct = 0
    total_top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)

            top1_predictions = outputs.argmax(dim=1)
            total_top1_correct += (top1_predictions == labels).sum().item()

            k = min(5, outputs.size(1))
            topk_indices = outputs.topk(k, dim=1).indices
            total_top5_correct += topk_indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    top1_acc = total_top1_correct / total_samples if total_samples > 0 else 0.0
    top5_acc = total_top5_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, top1_acc, top5_acc

# To train your model. BTW can you tell anything different from the perception one?
# Yeah thats right, ive moved the epoch inside the training loop
# Just trying out different stuff 
def train_model(model, train_loader, val_loader, loss_fn, optimizer, start_epoch=0, num_epochs=epochs):
    global best_val_loss
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Run validation at the end of every epoch.
        val_loss, val_top1_acc, val_top5_acc = evaluate_model(model, val_loader, loss_fn)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Validation Loss: {val_loss:.2f}")
        print(f"Top-1 Acc: {val_top1_acc:.4f}")
        print(f"Top-5 Acc: {val_top5_acc:.4f}")

        # Reuse the save logic from perception folder
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    # Also for those of you wondering, if you want to train the model on N epochs
                    # you will have to do N + epochs since it starts from the existing epoch
                    # So if you had 20 epochs, and you want to train for 10 more, 
                    # you will have to set it to 30 epochs in total.
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_classes": len(item_to_idx),
                },
                BEST_ICON_PATH
            )

            print(f"Saved best model (val loss {best_val_loss:.4f})")

# oh yeah I rented a GPU from vast its so much faster now.
# I suppose though this model should run fairly well on CPU since its smaller.

if BEST_ICON_PATH.exists():
    print("Loading checkpoint...")

    checkpoint = torch.load(BEST_ICON_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("loss", float("inf")))

    print(f"Best validation loss {best_val_loss:.4f}")

else:
    print("No checkpoint, new training session.")
    start_epoch = 0

# Last piece of the puzzle
train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    start_epoch=start_epoch,
    num_epochs=epochs
)

# Final test evaluation after training loop, mirroring the environment trainer flow.
test_loss, test_top1_acc, test_top5_acc = evaluate_model(model, test_loader, loss_fn)
print(f"Final Test Loss: {test_loss:.4f}, Top-1 Acc: {test_top1_acc:.4f}, Top-5 Acc: {test_top5_acc:.4f}")

# Hello again, I mean, since you are reading this file, did you set up the perception part?
# I mean, I hope you did, anyways this is just 1/3 models for the actual HUD part
# Its kind of annoying, but this is the most complex part I guess
# We need it to learn the heart, hunger and tool bar next in another model.
# And then the inventory and all sorts of screens. Then combine it together into one
# Ill cya on the next part, I guess.