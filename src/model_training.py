import torch # # main brain handles complex math (matrices and tensors)
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms, models # toolbox for computer vision: photoshop for ourself contains automatic resizing, cropping, padding, rotating, and chaning colors.
from torch.utils.data import DataLoader
import os
from PIL import Image # Pillow is a python imageing library: pytorch need a tool for file opener for image and pillow works besrt for it as pytorch doesn't know j=how to open it. 

import copy
import time

# --- : HELPER CLASSES (Preprocessing) ---

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)

        # calculate how much black space is needed
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top

        # Applying the padding (fill = 0 means black)
        return transforms.functional.pad(image, (p_left, p_top, p_right, p_bottom), fill=0, padding_mode='constant')

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):

        # generate noise
        noise = torch.randn(tensor.size()) * self.std + self.mean

        # add noise and clamp to keep value between 0 and 1
        return torch.clamp(tensor + noise, 0., 1.)

# --- DATA TRANSFORMS ---

train_transform = transforms.Compose([

     # pad the rectangular id
    SquarePad(),

    # resize to 224 * 224
    transforms.Resize((224, 224)),

    # Geometric augmentation (shape or position)

    # Rotation: for orientation Invariance.
    transforms.RandomRotation(degrees=180),

    # Perspective: simulates taking photo from an angle
    # distortion_scale=0.2 is "small/mild", p=0.5 means it happens 50% of the time.
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),

    # Crop
    # translate=(0.1, 0.1) moves the id slightly (10%)
    # scale=(0.8, 1.2) zoom in or out (80% - 120%

    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),

    # PhotoMetric augmentation (lighting/quality)

    # Brightness and contrast
    # simulates sunny days and dark rooms.
    transforms.ColorJitter(brightness=0.4, contrast=0.3),

    # Blur (light gaussian blur)
    # simulate out-of-focus camera
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),

    # Conversion
    # convert to number (tensor)(0-255) to tensor (0-1)
    transforms.ToTensor(),

     # adding noise std = 0.05 add 5% noise
    AddGaussianNoise(mean=0., std=0.05),

    # Normalization
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- LOAD DATA ---

data_dir = 'datasets'
batch_size = 8

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

'''
Dataloaders: Batch size = 8 good for small dataset and if get memory error change to 4;
shuffle = True
'''

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# Checking

print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
print(f"Classes: {train_dataset.classes}")

# --- MODEL SETUP (Transfer Learning) ---

# Setup Device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download ResNet18
print("Downloading ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze the "Brain" (Feature Extractor)
'''
we freexe cause we don;t need the whole brain we only
need to trained the final decission layer.
'''
for param in model.parameters():
    param.requires_grad = False

# unfreeze the last generic block :
for param in model.layer4.parameters():
    param.requires_grad = True

'''
modifying the final layer : 
Resnet final layer is called fully connected `fc`
'''
# Swap the "Head" (Classifier)
# Change output from 1000 classes to 3 classes (back, front, not_id)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

model = model.to(device)


'''
Define Loss and Optimizer
criterion is a way model calculates it;s mistakes
cross entory is a standard for clasification.
'''

criterion = nn.CrossEntropyLoss()


'''
optimizer fix the weight to fic the mistakes.

Only optimize the parameters of the final layer (model.fc)
'''
optimizer = optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-2)

# 5. SCHEDULER (Fixed: Removed 'verbose=True')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3
)

print("Model setup complete with Scheduler.")
# print("Model setup complete. Ready to create the Training Loop next.")


print("\nStarting training...")
start_time = time.time()

num_epochs = 40
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # Print current Learning Rate manually since we removed verbose
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current LR: {current_lr}")

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Update Scheduler (Only based on Val Loss)
        if phase == 'val':
            scheduler.step(epoch_loss)
            
            # Check if this is the best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')
                print("  --> New Best Model Saved!")

    print()

time_elapsed = time.time() - start_time
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val Acc: {best_acc:4f}')

model.load_state_dict(best_model_wts)