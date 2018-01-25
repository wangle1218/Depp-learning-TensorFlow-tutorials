# encoding:utf8
import tflearn
from tflearn.data_utils import build_hdf5_image_dataset
import h5py
import numpy as np
import os
from random import sample
from scipy.misc import imresize
import matplotlib.image as mpimg
from collections import defaultdict

def load_data(img_dir):
    flower_classes = sorted([dirname for dirname in os.listdir(img_dir)])
    flower_classes = flower_classes[1:]
    flowers_root_path = img_dir
    image_paths = defaultdict(list)

    for flower_class in flower_classes:
        image_dir = os.path.join(flowers_root_path, flower_class)
        for filepath in os.listdir(image_dir):
            if filepath.endswith(".jpg"):
                image_paths[flower_class].append(os.path.join(image_dir, filepath))

    for paths in image_paths.values():
        paths.sort()    

    flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
    flower_paths_and_classes = []
    for flower_class, paths in image_paths.items():
        for path in paths:
            flower_paths_and_classes.append((path, flower_class_ids[flower_class]))

    test_ratio = 0.2
    train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))

    np.random.shuffle(flower_paths_and_classes)

    flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
    flower_paths_and_classes_test = flower_paths_and_classes[train_size:]
    return flower_paths_and_classes_train,flower_paths_and_classes_test


def prepare_image(image, target_width = 128, target_height = 128, max_zoom = 0.2, train=True):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height
    
    if train:
        # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
        # between 1.0 and 1.0 + `max_zoom`.
        resize_factor = np.random.rand() * max_zoom + 1.0
        crop_width = int(crop_width / resize_factor)
        crop_height = int(crop_height / resize_factor)
        
        # Next, we can select a random location on the image for this bounding box.
        x0 = np.random.randint(0, width - crop_width)
        y0 = np.random.randint(0, height - crop_height)
    else:
        x0, y0 = 0,0

    x1 = x0 + crop_width
    y1 = y0 + crop_height
    
    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1]

    # Let's also flip the image horizontally with 50% probability:
    if train:
        if np.random.rand() < 0.5:
            image = np.fliplr(image)

    # Now, let's resize the image to the target dimensions.
    image = imresize(image, (target_width, target_height))
    
    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return image.astype(np.float32) / 255

def batch_iter(train = True,flower_paths_and_classes, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(flower_paths_and_classes)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1

    if shuffle:
        shuffled_data = sample(flower_paths_and_classes, data_size)
    else:
        shuffled_data = flower_paths_and_classes
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if end_index - start_index < batch_size:
            continue
        batch_data = shuffled_data[start_index:end_index]

        images = [mpimg.imread(path)[:, :, :3] for path, labels in batch_data]
        prepared_images = [prepare_image(image, train = train) for image in images]
        X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
        y_batch = np.array([labels for path, labels in batch_data], dtype=np.int32)
        yield X_batch, y_batch