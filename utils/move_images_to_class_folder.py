import shutil, os
import pandas as pd
from util_script import create_dir
# read label
labels = pd.read_csv("/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/histopathologic-cancer-detection/train_labels.csv")
# sort label
labels = labels.sort_values('label')

# take unique labels
class_names = list(labels.label.unique())

# loc of train data
train_images = '/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/histopathologic-cancer-detection/train'
# loc of labels with img
train_cat = '/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/histopathologic-cancer-detection/train_new'

#creating subfolders
for c in class_names:
    dest = train_cat+'/' + str(c)
    create_dir(dest)
    for i in list(labels[labels['label']==c]['id']): # Image Id
        get_image = os.path.join(train_images, i) + ".tif" # Path to Images
        move_image_to_cat = shutil.move(get_image, dest)