import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from tensorflow import io as tf_io
from tensorflow import clip_by_value
from tensorflow import image as tf_image
from tensorflow import random

class Preprocess:
    def __init__(self, resize=(256,256), size=(224, 224), rotation=0.2, SEED=44):
        self.SEED = SEED
        self.resize = resize
        self.size = size
        self.rotation = rotation
        self.resize_image = keras.layers.Resizing(*resize, crop_to_aspect_ratio=True)
        self.rotate_image = keras.layers.RandomRotation(rotation, seed=self.SEED)
        self.blur = keras.layers.RandomGaussianBlur(seed=self.SEED)
        self.jitter = keras.layers.RandomColorJitter(seed=self.SEED)
        
    def resize_augment_image(self, filepath: str, augment=False, c_jitter=False):
        """ Loads an image then resizes and flips* it
        *Random flip 1/2 of the time
        Args:
            filename (str): image path
            augment (bool, optional): whether to augment the image. Defaults to False.
            c_jitter (bool, optional): whether to apply color jitter. Defaults to False.
        Returns:
            (tuple): image tensor and label
        """
        image = tf_io.read_file(filepath)  # Read image file
        image = tf_image.decode_jpeg(image, channels=3)  # Decode JPEG image
        image = self.resize_image(image)  # Resize image
        image = tf_image.random_crop(image, (*self.size,3), seed=self.SEED)  # Random crop
        if augment:
            image = tf_image.random_flip_left_right(image, seed=self.SEED)  # Random flip
            image = tf_image.random_flip_up_down(image, seed=self.SEED)  # Random flip
            image = self.rotate_image(image) 
            image = self.blur(image)
        if c_jitter:
            image = self.jitter(image)
        image = tf.cast(image, tf.uint8)
        return image
    
def apply_model_specific_preprocessing(image, model):
    """Applies preprocessing for specific models
    
    Args:
        image (_type_): image tensor
        model (str): model name
    Returns:
        _type_
    """
    if model == "resnet":
        image = tf.cast(image, tf.float32)
        image = keras.applications.resnet_v2.preprocess_input(image)
    elif model == "efficientnet":
        image = keras.applications.efficientnet.preprocess_input(image)
    # TODO: add the preprocessing for swintransformer
    # elif model == "swintransformer":
        
    else:
        raise ValueError(f"Model {model} not supported")
    return image

class Mix:
    """CutMix data augmentation technique
    adapted from https://keras.io/examples/vision/cutmix/
    """
    def __init__(self, img_size=224):
        self.IMG_SIZE = img_size

    def sample_beta_distribution(self, size: list[int], concentration_0=0.2, concentration_1=0.2):
        """Generates samples from a Beta distribution

        Args:
            size (list[int]): shape of the sample
            concentration_0 (float, optional): concentration of the first gamma. Defaults to 0.2.
            concentration_1 (float, optional): concentration of the second gamma. Defaults to 0.2.

        Returns:
            _type_: beta distribution sample
        """
        gamma_1_sample = random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


    def get_box(self, lambda_value: float):
        """Helper function to get the bounding box

        Args:
            lambda_value (float): [0,1] percentage of the image to cut

        Returns:
            _type_: where the cuts will be made for each image
        """
        IMG_SIZE = self.IMG_SIZE
        cut_rat = keras.ops.sqrt(1.0 - lambda_value)

        cut_w = IMG_SIZE * cut_rat  # rw
        cut_w = keras.ops.cast(cut_w, "int32")

        cut_h = IMG_SIZE * cut_rat  # rh
        cut_h = keras.ops.cast(cut_h, "int32")

        cut_x = keras.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # rx
        cut_x = keras.ops.cast(cut_x, "int32")
        cut_y = keras.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # ry
        cut_y = keras.ops.cast(cut_y, "int32")

        boundaryx1 = clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
        boundaryy1 = clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
        bbx2 = clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
        bby2 = clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

        target_h = bby2 - boundaryy1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundaryx1
        if target_w == 0:
            target_w += 1

        return boundaryx1, boundaryy1, target_h, target_w


    def cutmix(self, train_ds_one, train_ds_two, alpha=0.2):
        """CutMix augmentation box size is determined by a beta distribution 
        instead of mixing the two images directly as in MixUp.

        Args:
            train_ds_one (_type_): first dataset
            train_ds_two (_type_): second dataset
            alpha (float, optional): alpha of the beta Defaults to 0.2.

        Returns:
            _type_: cutmix image and label
        """
        IMG_SIZE = self.IMG_SIZE

        (image1, label1), (image2, label2) = train_ds_one, train_ds_two

        # Get a sample from the Beta distribution
        lambda_value = self.sample_beta_distribution(1, alpha, alpha)[0]

        # Get the bounding box offsets, heights and widths
        boundaryx1, boundaryy1, target_h, target_w = self.get_box(lambda_value)

        # Crop and pad
        crop2 = tf_image.crop_to_bounding_box(
            image2, boundaryy1, boundaryx1, target_h, target_w)
        image2 = tf_image.pad_to_bounding_box(
            crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
        crop1 = tf_image.crop_to_bounding_box(
            image1, boundaryy1, boundaryx1, target_h, target_w)
        img1 = tf_image.pad_to_bounding_box(
            crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)

        # Modify the first image by subtracting the patch from `image1`
        image1 = image1 - img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        image = image1 + image2

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = keras.ops.cast(lambda_value, "float32")

        # Combine the labels of both images
        label = lambda_value * label1 + (1 - lambda_value) * label2
        return image, label


    def mix_up(self, ds_one, ds_two, alpha=0.2):
        """Mix of the two datasets mixing amount is determined by 
        a beta distribution

        Args:
            ds_one (_type_): first dataset
            ds_two (_type_): second dataset
            alpha (float, optional): mixing alpha. Defaults to 0.2.

        Returns:
            _type_: the mixed data and label
        """
        # Unpack two datasets
        images_one, labels_one = ds_one
        images_two, labels_two = ds_two
        batch_size = keras.ops.shape(images_one)[0]

        # Sample lambda and reshape it to do the mixup
        l = self.sample_beta_distribution(batch_size, alpha, alpha)
        x_l = keras.ops.reshape(l, (batch_size, 1, 1, 1))
        y_l = keras.ops.reshape(l, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)
        return (images, labels)