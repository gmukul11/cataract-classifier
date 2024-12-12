import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class CataractDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'cataract']
        
        # Create dataframe with image paths and labels
        data = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                data.append({
                    'image_path': img_path,
                    'label': class_idx,
                    'class_name': class_name
                })
        
        self.data_info = pd.DataFrame(data)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_info.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

def load_and_preprocess_data(data_dir, train_transform, val_transform, batch_size=32):
    """
    Load and preprocess data with train/validation split
    """
    # Create full dataset
    full_dataset = CataractDataset(
        root_dir=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    # Create test dataset
    test_dataset = CataractDataset(
        root_dir=os.path.join(data_dir, 'test'),
        transform=val_transform
    )

    # Split train dataset into train and validation
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=[x[1] for x in full_dataset],
        random_state=42
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx)
    )

    val_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx)
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader