import numpy as np
from skimage import transform, exposure
import random
import matplotlib.image as mpimg
import warnings
import os


# Cameras we will use
cameras = ['left', 'center', 'right']
cameras_steering_correction = [0.1, 0., -0.1]

def preprocess(image, top_offset=0.375, bottom_offset=0.125):
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = transform.resize(image[top:-bottom, :], (66, 200, 3))
    return image

def generate_samples(data, root_path, augment=True):
    while True:
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
                camera = np.random.randint(len(cameras)) if augment else 1
                # Read frame image and work out steering angle
                image = mpimg.imread(os.path.join(root_path, data[cameras[camera]].values[i].strip()))
                angle = data.steering.values[i] + cameras_steering_correction[camera]
                if augment:
                    # Add random shadow
                    h, w = image.shape[0], image.shape[1]
                    [x1, x2] = np.random.choice(w, 2, replace=False)
                    k = h / (x2 - x1)
                    b = - k * x1
                    shadow_adjustment = random.uniform(.4, .6)
                    non_shadow_adjustment = random.uniform(.75, 1.25)
                    for i in range(h):
                        c = int((i - b) / k)
                        # Randomly decrease brightness for shadowed part
                        image[i, :c, :] = (image[i, :c, :] * shadow_adjustment).astype(np.int32)
                        # And apply random brightness adjustment to the rest of the image
                        image[i, c:, :] = (np.minimum(
                            np.ones_like(image[i, c:, :]) * 255,
                            image[i, c:, :] * non_shadow_adjustment
                        )).astype(np.int32)
                # Randomly shift up and down while preprocessing
                v_delta = .05 if augment else 0
                image = preprocess(
                    image,
                    top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
                )
                # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])
            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)
