import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import imageio



class musicianDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform= transform
        
    def __len__(self):
      return len(self.annotations)
        


    def __getitem__(self, index):
        imgPath= os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image=imageio.imread(imgPath)
        y_label=torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            image=self.transform(image)
        return (image, y_label)    
    
