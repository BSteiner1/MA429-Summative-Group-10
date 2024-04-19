import os
from PIL import Image
import numpy as np
import random

#--------------------------------------------------------------------
def get_data():
    
    path = '../Data/Images'
    folders = os.listdir(path)
    
    data = []
    
    for folder in folders:
    
        # Label as healthy or diseased
        if folder.endswith('healthy'):
            label = 1
        else:
            label = 0

        # Iterate through the folders
        folder_images = os.listdir(path + '/' + folder)
        folder_path = path + '/' + folder + '/'

        # Extract images from each folder
        for image in folder_images:
            image_path = os.path.join(folder_path, image)
            image_path = Image.open(image_path)

            # Convert image to pixel values
            pixel_values = np.array(image_path)

            # Add image and label
            data.append((pixel_values, label))
        
    return data

#--------------------------------------------------------------------

def train_val_test_split(data):
    
    # Shorten data to 10,000 images
    data = data[:10000]
    
    # Shuffle the data
    random.shuffle(data)
    
    # Split into images and labels
    images = [pair[0] for pair in data]
    labels = [pair[1] for pair in data]
    
    # Define the proportions for train, validation, and test sets
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    
    # Calculate the number of samples for each set
    num_samples = len(data)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    num_test = num_samples - num_train - num_val
    
    # Split the images
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    # Split the labels
    train_labels = labels[:num_train]
    val_labels = labels[num_train:num_train + num_val]
    test_labels = labels[num_train + num_val:]
    
    # Normalize the training data
    train_images = [image/255 for image in train_images]
    val_images = [image/255 for image in val_images]
    test_images = [image/255 for image in test_images]
    
    return train_images, val_images, test_images, train_labels, val_labels, test_labels

#--------------------------------------------------------------------