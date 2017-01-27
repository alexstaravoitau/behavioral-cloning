import numpy as np
from skimage import transform, exposure
import random
import matplotlib.image as mpimg
import warnings
import os


# Cameras we will use
cameras = ['left', 'center', 'right']
cameras_steering_correction = [0.25, 0., -0.25]

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1, 2]] += 0.5
    return np.float32(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1, 2]] -= 0.5
    return np.float32(rgb.dot(xform.T))

def preprocess(image, top_offset=0.375, bottom_offset=0.125):
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = transform.resize(image[top:-bottom, :], (66, 200, 3))
    return rgb2ycbcr(image)

def generate_samples(data, root_path, augment=True):
    while 1:
        # Generate random batch of indices
        indices = np.random.permutation(data.count()[0])
        batch_size = 128
        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
            # Output arrays
            x = np.empty([0, 66, 200, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images
            for i in batch_indices:
                # Randomly select camera
                #camera = np.random.choice(len(cameras), p=[.25, .5, .25])
                camera = np.random.randint(len(cameras)) if augment else 1
                # Read frame image and work out steering angle
                image = mpimg.imread(os.path.join(root_path, data[cameras[camera]].values[i].strip()))
                angle = data.steering.values[i] + cameras_steering_correction[camera]
                if augment:
                    # Add random shadow
                    x1, y1 = random.randint(0, image.shape[1]), random.randint(0, image.shape[0])
                    x2, y2 = random.randint(x1, image.shape[1]), random.randint(y1, image.shape[0])
                    image[y1:y2, x1:x2, :] = (image[y1:y2, x1:x2, :] * .5).astype(np.int32)
                # Append to batch
                x = np.append(x, [preprocess(image)], axis=0)
                y = np.append(y, [angle])
            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)
