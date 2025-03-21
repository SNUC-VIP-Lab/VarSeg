import torch
import os
from models.vqvae import VQVAE
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from vqvae_utils import *

import torch.distributed as dist
os.environ["MASTER_ADDR"] = "127.0.0.1"  # Change if running on multiple nodes
os.environ["MASTER_PORT"] = "29502"  # Choose an available port
os.environ["WORLD_SIZE"] = "1"  # Set to the number of processes (GPUs)
os.environ["RANK"] = "0"  # Unique rank of the process

dist.init_process_group(backend="nccl", init_method="env://")
print("World Size:", dist.get_world_size())

# Load the best model and evaluate a subset of test images
def test_vqvae(model_path, test_loader, indices=None, max_images=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # vqvae = VQVAE(in_channels=1, test_mode=False).to(device)
    vqvae = VQVAE(in_channels=1, vocab_size=4096, z_channels=32, test_mode=True).to(device)

    vqvae.load_state_dict(torch.load(model_path, map_location=device))
    vqvae.eval()

    total_dice = 0.0
    total_miou = 0.0
    total_spec = 0.0
    selected_images = []
    
    with torch.no_grad():
        for img in tqdm(test_loader, desc="Testing"):
            # if indices and i not in indices:
                # continue  # Skip images not in the selected indices

            img = img.to(device)
            rec_img, _, _ = vqvae(img)
            accuracy, sensitivity, specificity, f1_or_dsc, jaccard, miou = dice_score(rec_img.detach().cpu().numpy(), img.detach().cpu().numpy())
            total_dice += f1_or_dsc
            total_miou += miou
            total_spec += specificity
            # Convert tensors to PIL images
            # original = transforms.ToPILImage()(img.squeeze(0).cpu())
            # reconstructed = transforms.ToPILImage()(rec_img.squeeze(0).cpu())

            # selected_images.append((original, reconstructed))

            # if len(selected_images) >= max_images:
                # break  # Stop after max_images

    avg_dice = total_dice / len(test_dataset)
    avg_miou = total_miou / len(test_dataset)
    avg_spec = total_spec / len(test_dataset)
    print(f"Test Dice Score: {avg_dice:.4f}, Test miou: {avg_miou}, Test Spec: {avg_spec}")

    # Display the images in an 8x2 grid
    # display_results(selected_images, save_path="vqvae8192_results.png")

# Main script
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = CustomImageDataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/Test_GroundTruth/", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Load one image at a time

    # Set indices of test images to evaluate (or None for sequential images)
    test_indices = [0, 5, 10, 18, 20, 25, 30, 40]  # Example indices (max 8)

    test_vqvae("./checkpoints/vqvae4096_best.pth", test_loader, indices=test_indices, max_images=8)
