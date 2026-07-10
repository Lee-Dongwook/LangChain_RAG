import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CifakeDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.label_map = {"REAL": 0, "FAKE": 1}
        self.samples = []

        categories = sorted(os.listdir(root))
        
        for category in categories:
            if category not in self.label_map:
                continue

            category_dir = os.path.join(root, category)
            label = self.label_map[category]

            file_names = sorted(os.listdir(category_dir))
            for file_name in file_names:
                if file_name.lower().endswith(('.jpg','.jpeg','.png')):
                    file_path = os.path.join(category_dir, file_name)
                    self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        file_path, label = self.samples[idx]
        image = Image.open(file_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)

        return image, label
    

