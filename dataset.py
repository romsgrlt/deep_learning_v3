from torch.utils.data import Dataset
from torch import stack, tensor
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


labels = ['landbird/land', 'landbird/water', 'waterbird/land', 'waterbird/water']

def load_metadata():
    metadata = pd.read_csv('data/metadata.csv')
    y_array = metadata['y'].values
    place_array = metadata['place'].values
    group_array = (y_array * 2 + place_array).astype(int)
    filename_array = metadata['img_filename'].values
    split_array = metadata['split'].values
    return y_array, group_array, filename_array, split_array


class WaterbirdsDataset(Dataset):
    def __init__(self, indices, y_array, group_array, filename_array):
        self.indices = indices
        self.images = []
        self.labels = []
        self.groups = []

        print(f"Préchargement de {len(indices)} images")
        for idx in tqdm(indices):
            img = Image.open(os.path.join('data', filename_array[idx])).convert('RGB')
            self.images.append(transform(img))
            self.labels.append(int(y_array[idx]))
            self.groups.append(int(group_array[idx]))
        self.images = stack(self.images)
        self.labels = tensor(self.labels)
        self.groups = tensor(self.groups)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.images[i], self.labels[i], self.groups[i]


def load_dataset():
    y_array, group_array, filename_array, split_array = load_metadata()

    train_idx = np.where(split_array == 0)[0]
    val_idx = np.where(split_array == 1)[0]
    test_idx = np.where(split_array == 2)[0]

    train_dataset = WaterbirdsDataset(train_idx, y_array, group_array, filename_array)
    val_dataset = WaterbirdsDataset(val_idx, y_array, group_array, filename_array)
    test_dataset = WaterbirdsDataset(test_idx, y_array, group_array, filename_array)

    return train_dataset, val_dataset, test_dataset
