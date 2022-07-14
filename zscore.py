import numpy as np
import h5py
import pandas as pd
import os
import skimage.morphology as sm
from skimage import measure
from skimage import filters


def data_generator(patient_indexes, h5_file_path):
    file = h5py.File(h5_file_path, 'r')
    imgs = file['data']
    labels = file['label']
    mask = []

    batch_img = np.zeros((229 * 184, 224, 192))

    # 输入的是病人的index，转换成切片的index
    slice_indexes = []
    for patient_index in patient_indexes:
        for slice_index in range(2, 186):  # delete the first two and the last three slices
            slice_indexes.append(patient_index * 189 + slice_index)
    num_of_slices = len(slice_indexes)
    print(num_of_slices)

    for i in range(num_of_slices):
        img = np.array(imgs[slice_indexes[i]][5:229, 2:194])

        zero = np.zeros_like(img)
        one = np.ones_like(img)
        p_mask = np.where(img < img.mean(), one, zero)
        p__mask = p_mask
        label = measure.label(p__mask, connectivity=2)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area > 10000:
                valid_label.add(prop.label)
        p__mask = np.in1d(label, list(valid_label)).reshape(label.shape)
        p__mask = np.where(p__mask < 1., one, zero)

        mask.append(p__mask)
        logical_mask = p__mask > 0.
        mean = img[logical_mask].mean()
        std = img[logical_mask].std() + 1e-5
        img = (img - mean) / std

        batch_img[i] = img

    batch_label = []
    batch_mask = []

    for i in range(num_of_slices):
        current_label = labels[slice_indexes[i]][5:229, 2:194]
        current_mask = mask[i]

        batch_label.append(current_label)
        batch_mask.append(current_mask)

    batch_label = np.array(batch_label)
    batch_mask = np.array(batch_mask)

    batch_img = np.expand_dims(batch_img, 1)
    batch_label = np.expand_dims(batch_label, 1)
    batch_mask = np.expand_dims(batch_mask, 1)

    print(batch_img.shape)

    np.save("./image.npy", batch_img)
    np.save("./label.npy", batch_label)
    np.save("./mask.npy", batch_mask)


if __name__ == '__main__':
    data_file_path = './train.h5'
    num_patients = 229
    patient_indexes = np.array([i for i in range(num_patients)])
    data_generator(patient_indexes=patient_indexes, h5_file_path=data_file_path)
