from dataset.Dataset import MultiTaskCabDataset

data = MultiTaskCabDataset("data", split="train")

print(len(data))

X_labeled, X_unlabeled, y_labeled, y_unlabeled = data.split_labeled_unlabeled_data(data.data, data.targets)

print(f"labeled_image{len(y_labeled)}")
for i in X_labeled:
    print(X_labeled)
    

print(f"unlabeled_image{len(y_unlabeled)}")
for i in y_unlabeled:
    print(i)



# from dataset.BaseDataset import BaseDataset

import numpy as np

def open_image(img_path):
    return np.array(Image.open(img_path).convert('RGB'))