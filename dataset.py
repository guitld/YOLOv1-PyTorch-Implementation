import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


class PascalVOCDataset(torch.utils.data.Dataset):
    """
    Generates the PyTorch dataset that loads the Pascal VOC dataset 
    Args:
        filenames_path (str): indicates the .csv file that contains tuples (image_filename, label_filename).
        img_path (str): indicates the folder which contains the images.
        label_path (str): indicates the folder which contains the labels.
        grid_size (int, optional): By default, the dataset will use 7 as the grid size for the Yolo algorithm.
        n_boxes (int, optional): By default, the algorithm will detect 2 boxes.
        n_classes (int, optional): By default, the number of classes is 20.
        transform (object, optional): By default, no transformations are applied to the dataset.
    """
    def __init__(self, filenames_path, img_path, label_path, grid_size=7, n_boxes=2, n_classes=20, transform=None):
        self.filenames  = pd.read_csv(filenames_path)
        self.img_path   = img_path
        self.label_path = label_path
        self.transform  = transform
        self.grid_size  = grid_size
        self.n_boxes    = n_boxes
        self.n_classes  = n_classes
        self.dataframe  = self.process_labels()

    # Loops through the entire .csv to create the label output matrix (7, 7, 30)
    def process_labels(self):
        instances = []
        with tqdm(total=len(self.filenames)) as pbar:
            for _, row in self.filenames.iterrows():
                pbar.update(1)
                boxes = []
                with open(f"{self.label_path}/{row['text']}") as f:
                    for label in f.readlines():
                        class_label, x, y, width, height = [float(x) for x in label.replace("\n", "").split()]
                        boxes.append([int(class_label), x, y, width, height])
                
                output_matrix = torch.zeros((self.grid_size, self.grid_size, self.n_classes + 5 * self.n_boxes))
        
                for box in boxes:
                    class_label, x, y, width, height = box
                    # cell = (i, j)
                    cell = (int(self.grid_size * y), int(self.grid_size * x))
                    # cell_coords = (y, x)
                    cell_coords = (self.grid_size * y - cell[0], self.grid_size * x - cell[1])
                    cell_shape = (width * self.grid_size, height * self.grid_size)


                    if output_matrix[cell[0], cell[1], 20] == 0:
                        output_matrix[cell[0], cell[1], 20] = 1
                        box_coords = torch.tensor([cell_coords[1], cell_coords[0], cell_shape[0], cell_shape[1]])
                        output_matrix[cell[0], cell[1], 21:25] = box_coords
                        output_matrix[cell[0], cell[1], class_label] = 1
                    
                instances.append((row['image'], output_matrix))
        return pd.DataFrame(instances, columns=['image', 'label'])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_file, label = self.dataframe.iloc[idx]
        img = Image.open(f"{self.img_path}/{img_file}")
        
        if self.transform:
            img, label = self.transform(img, label)
        
        return img, label

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes