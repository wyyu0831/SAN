import numpy as np


def train_data_generator(patient_indexes, fold):
    imgs = np.load('/data/image.npy')
    labels = np.load('/data/label.npy')
    masks = np.load('/data/mask.npy')

    slice_indexes = []
    for patient_index in patient_indexes:
        for slice_index in range(184):
            slice_indexes.append(patient_index * 184 + slice_index)
    num_of_slices = len(slice_indexes)

    batch_img = np.zeros((num_of_slices, 1, 224, 192))
    batch_label = np.zeros((num_of_slices, 1, 224, 192))
    batch_mask = np.zeros((num_of_slices, 1, 224, 192))

    for i in range(num_of_slices):
        batch_img[i] = imgs[slice_indexes[i]]
        batch_label[i] = labels[slice_indexes[i]]
        batch_mask[i] = masks[slice_indexes[i]]

    domain = np.zeros(229 * 184)
    domain[:55*184] = 0
    domain[55*184:89*184] = 1
    domain[89*184:116*184] = 2
    domain[116*184:128*184] = 3
    domain[128*184:155*184] = 4
    domain[155*184:169*184] = 5
    domain[169*184:180*184] = 6
    domain[180*184:215*184] = 7
    domain[215*184:] = 8

    p = np.where(domain == fold-1)
    q = np.zeros_like(domain)
    q[(p[0][-1] + 1):] = 1
    batch_domain = domain - q
    batch_domain = np.delete(batch_domain, p)
    print('Testing site: {}'.format(fold))

    return batch_img, batch_label, batch_domain, batch_mask


def val_data_generator(patient_indexes):
    imgs = np.load('/data/wyyu/image.npy')
    labels = np.load('/data/wyyu/label.npy')
    masks = np.load('/data/wyyu/mask.npy')


    slice_indexes = []
    for patient_index in patient_indexes:
        for slice_index in range(184):
            slice_indexes.append(patient_index * 184 + slice_index)
    num_of_slices = len(slice_indexes)
    print(num_of_slices)

    batch_img = np.zeros((num_of_slices, 1, 224, 192))
    batch_label = np.zeros((num_of_slices, 1, 224, 192))
    batch_mask = np.zeros((num_of_slices, 1, 224, 192))

    for i in range(num_of_slices):
        batch_img[i] = imgs[slice_indexes[i]]
        batch_label[i] = labels[slice_indexes[i]]
        batch_mask[i] = masks[slice_indexes[i]]

    return batch_img, batch_label, batch_mask
