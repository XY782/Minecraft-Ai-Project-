import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = Path("best_minecraft_model.pth")
CHECKPOINT_PATH = "best_minecraft_model.pth"
#Idk what to add here, hello!
#Btw this is just the first part of multiple trainers. 
#Specifically, this only gathers the environment, it cuts out all GUI such as Crafting,
#Menu, Healthbar, Hotbar, Hungerbar, etc... Its for recognizing its surroundings.
#Gui and other stuff will be in a different trainer, so don't worry about that.

batch_size = 8

TRAIN_DATA = Path("Data/train_data")
VAL_DATA = Path("Data/val_data")
TEST_DATA = Path("Data/test_data")
IGNORE_INDEX = -1  # Pixels with this value will be ignored during loss computation

train_image = TRAIN_DATA / "PNG"
train_mask = TRAIN_DATA / "Masked"
train_metadata = TRAIN_DATA / "Metadata"
train_block_labels = TRAIN_DATA / "block_labels.json"  #yeah i din't add the test labels since its the same

val_image = VAL_DATA / "PNG"
val_mask = VAL_DATA / "Masked"
val_metadata = VAL_DATA / "Metadata"

test_image = TEST_DATA / "PNG"
test_mask = TEST_DATA / "Masked"
test_metadata = TEST_DATA / "Metadata"

#for folder in [train_image, train_mask, train_metadata, train_block_labels]:
#    print(folder.exists(), folder)   #For debugging purposes

class EnvironmentDataset(torch.utils.data.Dataset):
    def __init__(self, root):

        self.root = Path(root)
        self.ignore_index = IGNORE_INDEX

        self.image_dir = self.root / "PNG"
        self.mask_dir = self.root / "Masked"
        self.metadata_dir = self.root / "Metadata"
        self.block_labels_path = self.root / "block_labels.json"

        self.image_files = sorted(self.image_dir.glob("*.png"))

        with open(self.block_labels_path, "r") as f:
            self.labels = json.load(f)

        #Not bothered understanding this again, got comments so read through it if u want 
        #I mean it works, so why change it?
        #In a nutshell, it makes a mapping of the ids from label files and 
        #turns it into a dense mapping for training and after that
        #it makes a reverse mapping for decoding model predictions back to original semantic ids.
        #pretty cool huh? I think so.


        # Prefer id_to_semantic because it already contains BOTH block and mob labels.
        # Fallback to id_to_block for older label files that only have block labels.
        semantic_source = self.labels.get("id_to_semantic", self.labels.get("id_to_block", {}))
        self.id_to_semantic = {
            int(k): v
            for k, v in semantic_source.items()
        }

        # Build ONE global dense mapping for training targets.
        # Example: semantic ids [0, 34, 177] become train ids [0, 1, 2].
        ordered_semantic_ids = sorted(self.id_to_semantic.keys())
        self.semantic_to_train = {
            semantic_id: train_id
            for train_id, semantic_id in enumerate(ordered_semantic_ids)
        }

        # Reverse mapping used later for decoding model predictions back to original semantic ids.
        self.train_to_semantic = {
            train_id: semantic_id
            for semantic_id, train_id in self.semantic_to_train.items()
        }

        self.num_classes = len(self.semantic_to_train)

        # Fast lookup table for remapping mask ids.
        # Index = semantic id from PNG, value = dense train id.
        # Unknown ids stay at ignore_index so loss can ignore them.
        self.max_semantic_id = max(self.semantic_to_train.keys()) if self.semantic_to_train else 0

        self.semantic_lut = np.full(
            self.max_semantic_id + 1,
            self.ignore_index, 
            dtype=np.int64
        )

        for semantic_id, train_id in self.semantic_to_train.items():
            self.semantic_lut[semantic_id] = train_id



    def __len__(self):
        return len(self.image_files)




    @staticmethod
    def decode_mask_to_ids(masked_image):
        arr = np.array(masked_image, dtype=np.uint8)

        # RGB mask encodes class id as: R << 16, G << 8, B.
        # The mod samples data like this
        #Seperates R, G, and B channels and combines them into a single integer ID for each pixel.
        if arr.ndim == 3 and arr.shape[2] >= 3:
            ids = (
                (arr[:, :, 0].astype(np.int64) << 16)
                | (arr[:, :, 1].astype(np.int64) << 8)
                | arr[:, :, 2].astype(np.int64)
            )
            return ids

        # Fallback if a grayscale mask is ever used.
        if arr.ndim == 2:
            return arr.astype(np.int64)

        raise ValueError(f"Unsupported mask shape: {arr.shape}")


#Did you know, a blob of toothpaste is actually called a nurdle? I learned that today. Anyway, back to the code.


    def remap_semantic_to_train_ids(self, semantic_ids):
        # Start with all pixels marked as ignore.
        train_ids = np.full_like(semantic_ids, fill_value=self.ignore_index, dtype=np.int64)

        # Pixels within LUT range can be remapped in one vectorized step.
        #FYI LUT is acronym for lookup table
        in_range = semantic_ids <= self.max_semantic_id
        train_ids[in_range] = self.semantic_lut[semantic_ids[in_range]]

        return train_ids




    def __getitem__ (self, idx):
        
        image_path = self.image_files[idx]
        
        name = image_path.stem

        masked_path = self.mask_dir / f"{name}_masked.png"
        metadata_path = self.metadata_dir / f"{name}.json"

        image = Image.open(image_path).convert("RGB")
        masked = Image.open(masked_path)

        image_tensor = transforms.ToTensor()(image)
        masked_semantic_ids = self.decode_mask_to_ids(masked)
        masked_train_ids = self.remap_semantic_to_train_ids(masked_semantic_ids)
        masked_tensor = torch.from_numpy(masked_train_ids).long()
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return image_tensor, masked_tensor, metadata
        
train_dataset = EnvironmentDataset(TRAIN_DATA) 
val_dataset = EnvironmentDataset(VAL_DATA)
test_dataset = EnvironmentDataset(TEST_DATA)

# image, masked, metadata = train_dataset[0]
# print("Image:", image.shape)
# print("Mask:", masked.shape)
# print("Mask dtype:", masked.dtype)
# print("Num classes:", train_dataset.num_classes)
# print("Ignore index:", train_dataset.ignore_index)
# print("Unique mask values:", torch.unique(masked))
# #for debugging purposes: Checks datatype and some other stuff

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)

# for image_tensor, mask_tensor, metadata in train_loader:
#     print(image_tensor.shape)
#     print(mask_tensor.shape)
#     break
# #for debugging purposes: Checks if dataloder works

#Ur personal simple UNet, don't expect much from it though...
class simpleUNet(nn.Module):
    def __init__(self, num_classes):
        super(simpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def focal_loss(logits, targets, ignore_index=-1, gamma=2.0):
    # Compute per-pixel CE first, then down-weight eas y pixels to focus on hard classes.
    ce = nn.functional.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
        reduction="none"
    )
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return logits.sum() * 0.0

    ce_valid = ce[valid_mask]
    pt = torch.exp(-ce_valid)
    focal = ((1.0 - pt) ** gamma) * ce_valid
    return focal.mean()


def combined_loss(logits, targets, ignore_index=-1):
    # Use multiple losses so training does not collapse to dominant classes like air.
    ce = nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)
    foc = focal_loss(logits, targets, ignore_index=ignore_index, gamma=2.0)
    return 0.7 * ce + 0.3 * foc
    
model = simpleUNet(num_classes=train_dataset.num_classes)
model = model.to(device)

#Im assuming if you've got up to here, you might be a bit tired. Have a cookie!

optimizer = optim.Adam(model.parameters(), lr=1e-3)

start_epoch = 0
best_val_loss = float("inf")

#Continues training from the last checkpoint, becaause of course, im lazy 
#I mean who does want to retrain from scratch when you can just continue from the last checkpoint?
if Path(CHECKPOINT_PATH).exists():
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint.get("epoch", 0)

    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Resuming from epoch {start_epoch}")
    print(f"Best validation loss: {best_val_loss}")

images, masks, metadata = next(iter(train_loader))

# outputs = model(images)
# print(outputs.shape)
# print(masks.shape)
# loss = loss_fn(
#     outputs,
#     masks
# )
# print(loss.item())
# print(torch.unique(masks)[:50])
# print(masks.max())
# #Another debug yay!

# #OVERFIT TEST TIME!!!!!
# for i in range (100):
#     optimizer.zero_grad(),

#     output = model(images)

#     loss = loss_fn(output, masks)
#     loss.backward()
#     optimizer.step()
#     if i % 4 == 0:
#         print(f"Iteration {i}, Loss: {loss.item()}")
#         print(next(model.parameters()).device) #checks device of model parameters


epochs = 150 #Bro you know how slow this is? Im training on a darn CPU its like 1 hour per epoch

#actual training pipeline now isn't it cool
print("Starting training...")
#need to make sure it actually runs on CPU cus its so slow it doesnt output anything
#you know what maybe i'll just rent a GPU, its pretty cheap. right?
for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0.0
    for images, masks, metadata in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks, ignore_index=train_dataset.ignore_index)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

    # Run validation at the end of each epoch.
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, masks, metadata in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks, ignore_index=val_dataset.ignore_index)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Save best progress so training can resume from the strongest checkpoint.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": train_dataset.num_classes,
                "ignore_index": train_dataset.ignore_index,
                "train_to_semantic": train_dataset.train_to_semantic,
                "semantic_to_train": train_dataset.semantic_to_train,
            },
            BEST_MODEL_PATH
        )
        print(f"Saved best checkpoint to {BEST_MODEL_PATH} (val loss {best_val_loss:.4f})")




#Testing the data
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
model.eval()
total_test_loss = 0.0
with torch.no_grad():
    for images, masks, metadata in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = combined_loss(outputs, masks, ignore_index=test_dataset.ignore_index)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Final Test Loss: {avg_test_loss:.4f}")

#If you've actually read through all of this, good job!
#If you've skipped to the end, well, I guess you just wanted to see the final loss. It's okay, I don't blame you. for being ass!