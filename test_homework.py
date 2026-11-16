"""Test augmentation order, transform shape and held-out indices."""
import unittest

from PIL import Image
from homework_augmentations import AugmentationPipeline, blur, shear, brightness_contrast
from homework_experiments import split_indices, pipelines


class AugmentationTests(unittest.TestCase):
    def test_pipeline_order_and_removal(self):
        pipeline = AugmentationPipeline()
        pipeline.add_augmentation('add', lambda x: x + 2)
        pipeline.add_augmentation('multiply', lambda x: x * 3)
        self.assertEqual(pipeline.get_augmentations(), ('add', 'multiply'))
        self.assertEqual(pipeline.apply(1), 9)
        pipeline.remove_augmentation('add')
        self.assertEqual(pipeline.apply(1), 3)

    def test_transforms_preserve_shape_and_mode(self):
        image = Image.new('RGB', (64, 64))
        checks = [blur, shear, brightness_contrast, *(p.apply for p in pipelines(64).values())]
        for transform in checks:
            result = transform(image)
            self.assertEqual(result.size, image.size)
            self.assertEqual(result.mode, image.mode)

    def test_split_is_deterministic_disjoint_and_stratified(self):
        labels = [0] * 10 + [1] * 5
        train, val = split_indices(labels)
        self.assertEqual((train, val), split_indices(labels))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train + val), list(range(len(labels))))
        self.assertEqual(sorted(labels[i] for i in val), [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
