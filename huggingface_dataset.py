from datasets import load_dataset, Features, ClassLabel, Array3D
from PIL import Image
import numpy as np

dataset = load_dataset("mw_images", 
                       data_files={
                           "train": "train",
                             "test": "test",
                             "validation": "valid",
                             "symbols": "symbols",
                             "synthetic": "synthetic"}

)
# d = dataset[0]
print(dataset)
dataset.push_to_hub("us4/mathwriting-dataset-image")
# img = Image.open(d['image'])
# ar = np. array(img)
# print(ar)