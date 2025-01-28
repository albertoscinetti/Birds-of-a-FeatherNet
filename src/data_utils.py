
# obtain mean and std from images 
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_batch = images.size(0)
        images = images.view(image_count_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_batch
    mean /= (total_images_count)
    std /= (total_images_count)

    return mean, std


# class for dataset of bird imagaes 
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class BirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = csv_file
        self.root_dir = root_dir  # Root directory containing images
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get the relative image path
        img_path = self.annotations.iloc[index, self.annotations.columns.get_loc('image_path')]
        img_path = img_path[14:]
        # Combine with root directory
        full_img_path = os.path.join(self.root_dir, img_path)

        # Open the image
        image = Image.open(full_img_path)
        image = image.convert('RGB')

        # Get the label (assuming label is in the second column)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        # Apply transformation if any
        if self.transform:
            image = self.transform(image)

        return image, y_label
    

