import torch

def collect(batch):
    """Collect the data for one batch.
    """
    imgs = []
    targets = []
    filenames = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        filenames.append(sample[2])
    return imgs, targets, filenames