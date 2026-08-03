import math
import random
import numbers
import numpy as np


class Compose(object):
    """Minimal replacement for torchvision.transforms.Compose."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class NormalizeUnitSphere(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  (N, 3) numpy array
        """
        if center is None:
            p_max = pcl.max(axis=0, keepdims=True)
            p_min = pcl.min(axis=0, keepdims=True)
            center = (p_max + p_min) / 2  # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = np.sqrt((pcl ** 2).sum(axis=1, keepdims=True)).max(axis=0, keepdims=True)  # (1,1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert 'pcl_noisy' not in data, 'Point clouds must be normalized before applying noise perturbation.'
        data['pcl_clean'], center, scale = self.normalize(data['pcl_clean'])
        data['center'] = center.astype(np.float32)
        data['scale'] = scale.astype(np.float32)
        return data


class AddNoise(object):

    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        data['pcl_noisy'] = (data['pcl_clean'] +
                              np.random.randn(*data['pcl_clean'].shape).astype(np.float32) * noise_std)
        data['noise_std'] = np.float32(noise_std)
        return data


class AddLaplacianNoise(object):

    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        noise = np.random.laplace(0, noise_std, size=data['pcl_clean'].shape).astype(np.float32)
        data['pcl_noisy'] = data['pcl_clean'] + noise
        data['noise_std'] = np.float32(noise_std)
        return data


class AddUniformBallNoise(object):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, data):
        N = data['pcl_clean'].shape[0]
        phi = np.random.uniform(0, 2 * np.pi, size=N)
        costheta = np.random.uniform(-1, 1, size=N)
        u = np.random.uniform(0, 1, size=N)
        theta = np.arccos(costheta)
        r = self.scale * u ** (1 / 3)

        noise = np.zeros([N, 3], dtype=np.float32)
        noise[:, 0] = r * np.sin(theta) * np.cos(phi)
        noise[:, 1] = r * np.sin(theta) * np.sin(phi)
        noise[:, 2] = r * np.cos(theta)
        data['pcl_noisy'] = data['pcl_clean'] + noise
        return data


class AddCovNoise(object):

    def __init__(self, cov, std_factor=1.0):
        super().__init__()
        self.cov = np.asarray(cov, dtype=np.float64)
        self.std_factor = std_factor

    def __call__(self, data):
        num_points = data['pcl_clean'].shape[0]
        noise = np.random.multivariate_normal(np.zeros(3), self.cov, num_points).astype(np.float32)
        data['pcl_noisy'] = data['pcl_clean'] + noise * self.std_factor
        data['noise_std'] = np.float32(self.std_factor)
        return data


class AddDiscreteNoise(object):

    def __init__(self, scale, prob=0.1):
        super().__init__()
        self.scale = scale
        self.prob = prob
        self.template = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=np.float32)

    def __call__(self, data):
        num_points = data['pcl_clean'].shape[0]
        uni_rand = np.random.uniform(size=num_points)
        noise = np.zeros([num_points, 3], dtype=np.float32)
        for i in range(self.template.shape[0]):
            idx = np.logical_and(0.1 * i <= uni_rand, uni_rand < 0.1 * (i + 1))
            noise[idx] = self.template[i].reshape(1, 3)
        data['pcl_noisy'] = data['pcl_clean'] + noise * self.scale
        data['noise_std'] = np.float32(self.scale)
        return data


class RandomScale(object):

    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data['pcl_clean'] = data['pcl_clean'] * np.float32(scale)
        if 'pcl_noisy' in data:
            data['pcl_noisy'] = data['pcl_noisy'] * np.float32(scale)
        return data


class RandomRotate(object):

    def __init__(self, degrees=180.0, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        matrix = np.array(matrix, dtype=np.float32)

        data['pcl_clean'] = data['pcl_clean'] @ matrix
        if 'pcl_noisy' in data:
            data['pcl_noisy'] = data['pcl_noisy'] @ matrix

        return data


def standard_train_transforms(noise_std_min, noise_std_max, rotate=True, scale_d=0.2):
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        RandomScale([1.0 - scale_d, 1.0 + scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)
