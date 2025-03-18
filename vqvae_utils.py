import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
import torchvision.models as models
from PIL import Image, ImageDraw

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)  # Ensure values are between 0 and 1
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()
    
# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        bce_loss = nn.BCELoss()(y_pred, y_true)
        focal_loss = self.alpha * (1 - y_pred) ** self.gamma * bce_loss
        return focal_loss.mean()
    
class PerceptualLoss(nn.Module):
    def __init__(self, layer_idx=2, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(PerceptualLoss, self).__init__()
        self.device = device
        vgg = models.vgg16(pretrained=True).features[:layer_idx+1].to(device)  # Move model to device
        self.feature_extractor = nn.Sequential(*list(vgg)).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG layers

        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred, target = pred.to(self.device), target.to(self.device)

        # Convert single-channel (grayscale) images to 3-channel format for VGG
        pred = pred.repeat(1, 3, 1, 1)  
        target = target.repeat(1, 3, 1, 1)

        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        loss = self.criterion(pred_features, target_features)
        return loss
    
# Dice Score (for evaluation)
def dice_score(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred) > 0.5  # Convert to binary
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    return ((2.0 * intersection + 1.0) / (union + 1.0)).mean().item()

# Dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image
    
# Display results in an 8x2 grid using Pillow
def display_results(image_pairs, save_path="test_results.png"):
    num_images = len(image_pairs)
    cols, rows = 2, num_images  # 8x2 layout
    img_width, img_height = image_pairs[0][0].size

    grid_img = Image.new("L", (cols * img_width, rows * img_height), color=255)
    draw = ImageDraw.Draw(grid_img)

    for i, (orig, recon) in enumerate(image_pairs):
        grid_img.paste(orig, (0, i * img_height))  # Left column: Original
        grid_img.paste(recon, (img_width, i * img_height))  # Right column: Reconstructed

        # Draw a line separator
        draw.line([(img_width, i * img_height), (img_width, (i + 1) * img_height)], fill=0, width=2)

    grid_img.save(save_path)
    print(f"Results saved at {save_path}")