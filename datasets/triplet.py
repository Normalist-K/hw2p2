import torchvision
import numpy as np


class TripletDataset(torchvision.datasets.VisionDataset):
  def __init__(self, root, transform):  
    # For "root", note that you're making this dataset on top of the regular classification dataset.
    self.dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    
    # map class indices to dataset image indices
    self.classes_to_img_indices = [[] for _ in range(len(self.dataset.classes))]
    for img_idx, (_, class_id) in enumerate(self.dataset.samples):
      self.classes_to_img_indices[class_id].append(img_idx)
    
    # VisionDataset attributes for display
    self.root = root
    self.length = len(self.dataset.classes) # pseudo length! Length of this dataset is 7000, *not* the actual # of images in the dataset. You can just increase the # of epochs you train for.
    self.transforms = self.dataset.transforms
          
  def __len__(self):
    return self.length
    
  def __getitem__(self, anchor_class_idx):
    """Treat the given index as the anchor class and pick a triplet randomly"""
    anchor_class = self.classes_to_img_indices[anchor_class_idx]
    # choose positive pair (assuming each class has at least 2 images)
    anchor, positive = np.random.choice(a=anchor_class, size=2, replace=False)
    # choose negative image
    # hint for further exploration: you can choose 2 negative images to make it a Quadruplet Loss

    classes_to_choose_negative_class_from = list(range(self.length))
    classes_to_choose_negative_class_from.pop(anchor_class_idx) # TODO: What are we removing?
    negative_class_idx = np.random.choice(classes_to_choose_negative_class_from, replace=False)
    negative_class = self.classes_to_img_indices[negative_class_idx] # TODO: How do we randomly choose a negative class?
    negative = np.random.choice(negative_class, replace=False)# TODO: How do we get a sample from that negative class?
    
    # self.dataset[idx] will return a tuple (image tensor, class label). You can use its outputs to train for classification alongside verification
    # If you do not want to train for classification, you can use self.dataset[idx][0] to get the image tensor
    return self.dataset[anchor], self.dataset[positive], self.dataset[negative]