"""Ordered composition and three locally implemented PIL augmentations."""
import random

from PIL import Image, ImageEnhance, ImageFilter


class AugmentationPipeline:
    """Apply named transforms in insertion order."""

    def __init__(self):
        self._augmentations = {}

    def add_augmentation(self, name, aug):
        """Register or replace a callable."""
        self._augmentations[name] = aug

    def remove_augmentation(self, name):
        """Remove a registered transform by name."""
        del self._augmentations[name]

    def apply(self, image):
        """Return the result of sequential transforms."""
        for aug in self._augmentations.values():
            image = aug(image)
        return image

    def get_augmentations(self):
        """Return the ordered names without exposing mutable state."""
        return tuple(self._augmentations)


def blur(image):
    """Sample a Gaussian blur radius."""
    return image.filter(ImageFilter.GaussianBlur(random.uniform(.5, 2.5)))


def shear(image):
    """Apply a random horizontal shear centered vertically."""
    slope = random.uniform(-.25, .25)
    return image.transform(image.size, Image.Transform.AFFINE,
                           (1, slope, -slope * image.height / 2, 0, 1, 0),
                           resample=Image.Resampling.BILINEAR)


def brightness_contrast(image):
    """Independently vary brightness and contrast."""
    image = ImageEnhance.Brightness(image).enhance(random.uniform(.65, 1.35))
    return ImageEnhance.Contrast(image).enhance(random.uniform(.65, 1.35))
