from .regs import Registry, inspect_empty
from typing import Union, Sequence, Optional, Callable
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2
import torch


__all__ = ['TRANSREG', 'parse_transform']


TRANSREG = Registry()


def parse_transform(tdict: dict):
    if len(tdict) == 1:
        key: str = list(tdict.keys())[0]
        if key.capitalize() == 'Compose':
            tdict[key]['transforms'] = [parse_transform(t) for t in tdict[key]['transforms']]
        elif key.capitalize() == 'RandomApply':
            tdict[key]['transform'] = parse_transform(tdict[key]['transform'])
            tdict[key]['prob'] = float(tdict[key]['prob'])
        elif key.capitalize() == 'RandomChoice':
            tdict[key]['transforms'] = [parse_transform(t) for t in tdict[key]['transforms']]
            tdict[key]['probs'] = [float(p) for p in tdict[key]['probs']]
        return TRANSREG.get(key, **tdict[key])
    raise ValueError('Invalid transform dictionary')


class OneLineTransform:
    def __init__(self, _str: str):
        self._str = _str

    def __str__(self, level: int = 0, indent: str = '  '):
        return level * indent + self._str


@TRANSREG.reg()
class ToTensor(OneLineTransform):
    def __init__(self, dtype: str = 'float32', scale: bool = True):
        super().__init__(f"ToTensor(dtype={dtype}, scale={scale})")
        self.dtype = getattr(torch, dtype)
        self.scale = scale

    def __call__(self, img):
        return F.to_dtype(F.to_image(img), dtype=self.dtype, scale=self.scale)


@TRANSREG.reg(params={'policy': 'imagenet', 'interpolation': 'nearest'},
              param_desc={'policy': ["imagenet", "cifar10", "svhn"], 'interpolation': ["nearest", "bilinear", "bicubic"]})
class AutoAugment(OneLineTransform):
    def __init__(self, policy: str, interpolation: str, fill: int = 0):
        super().__init__(f"AutoAugment(policy={policy}, interpolation={interpolation}, fill={fill})")
        self.autoaugment = v2.AutoAugment(
            policy=getattr(v2.AutoAugmentPolicy, policy.upper()),
            interpolation=getattr(F.InterpolationMode, interpolation.upper()),
            fill=fill
        )

    def __call__(self, img):
        return self.autoaugment(img)


@TRANSREG.reg(params={'size': inspect_empty, 'interpolation': 'bilinear'},
              param_desc={'size': ['Int', 'Tuple[Int, Int]'], 'interpolation': ["nearest", "bilinear", "bicubic"]})
class Resize(OneLineTransform):
    def __init__(self, size: Union[int, Sequence[int]], interpolation: str,
                 max_size: Optional[int] = None, antialias: bool = True):
        super().__init__(f"Resize(size={size}, interpolation={interpolation}, max_size={max_size}, antialias={antialias})")
        assert isinstance(size, int) or len(size) == 2, "Size must be an int or a tuple of length 2"
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        assert all(s > 0 for s in self.size), "Size must be positive"
        self.interpolation = getattr(F.InterpolationMode, interpolation.upper()) # error if not valid
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, img):
        return F.resize(img, size=self.size,
                        interpolation=self.interpolation,
                        max_size=self.max_size,
                        antialias=self.antialias)


@TRANSREG.reg(params={'prob_hflip': 0, 'prob_vflip': 0},
              param_desc={'prob_hflip': 'Float (0 ~ 1)', 'prob_vflip': 'Float (0 ~ 1)'})
class RandomFlip(OneLineTransform):
    def __init__(self, prob_hflip: float, prob_vflip: float):
        super().__init__(f"RandomFlip(prob_hflip={prob_hflip}, prob_vflip={prob_vflip})")
        assert 0 <= prob_hflip <= 1 and 0 <= prob_vflip <= 1, "Flip probabilities must be in [0,1]"
        self.prob_hflip = prob_hflip
        self.prob_vflip = prob_vflip
        self.uniform = torch.distributions.Uniform(0, 1)

    def __call__(self, img):
        img = F.horizontal_flip(img) if self.uniform.sample((1,)).item() < self.prob_hflip else img
        img = F.vertical_flip(img) if self.uniform.sample((1,)).item() < self.prob_vflip else img
        return img


@TRANSREG.reg(params={'transforms': inspect_empty},
              param_desc={'transforms': 'Sequence[Callable]'})
class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __str__(self, level=0, indent='  '):
        reprs = [t.__str__(level=level + 1, indent=indent) for t in self.transforms]
        return level * indent + "Compose(\n" + ',\n'.join(reprs) + '\n' + level * indent + ")"


@TRANSREG.reg(params={'transform': inspect_empty, 'prob': inspect_empty},
              param_desc={'transform': 'Callable', 'prob': 'Float (0 ~ 1)'})
class RandomApply:
    def __init__(self, transform: Callable, prob: float):
        self.transform = transform
        self.prob = prob
        self.uniform = torch.distributions.Uniform(0, 1)

    def __call__(self, img):
        return self.transform(img) if self.uniform.sample((1,)).item() < self.prob else img

    def __str__(self, level=0, indent='  '):
        rep = self.transform.__str__(level=level + 1, indent=indent)
        return level * indent + f"RandomApply(\n{rep}, p={self.prob}" + '\n' + level * indent + ")"


@TRANSREG.reg(params={'transforms': inspect_empty, 'probs': inspect_empty},
              param_desc={'transforms': 'Sequence[Callable]', 'probs': 'Sequence[Float (0 ~ 1)]'})
class RandomChoice:
    def __init__(self, transforms: Sequence[Callable], probs: Sequence[float]):
        assert len(transforms) == len(probs), "len(transforms) must be equal to len(probs)"
        self.transforms = transforms
        self.probs = probs
        self.uniform = torch.distributions.Uniform(0, 1)

    def __call__(self, img):
        choice = self.uniform.sample((1,)).item()
        cumulative = 0.0
        for t, p in zip(self.transforms, self.probs):
            cumulative += p
            if choice <= cumulative:
                return t(img)
        return img

    def __str__(self, level=0, indent='  '):
        lines = []
        for i, (t, p) in enumerate(zip(self.transforms, self.probs), 1):
            rep = t.__str__(level=level+1, indent=indent)
            lines.append(f"transform{i}={rep}, p{i}={p}")
        return level * indent + "RandomChoice(\n" + ',\n'.join(lines) + ")"


@TRANSREG.reg()
class Identity(OneLineTransform):
    def __init__(self):
        super().__init__("Identity()")

    def __call__(self, img):
        return img


@TRANSREG.reg(params={'brightness': 0, 'contrast': 0, 'saturation': 0},
              param_desc={'brightness': ['Float(>=0) -> (1-val,1+val)', 'Tuple[Float,Float(>=0)] -> (val[0],val[1])'],
                          'contrast': ['Float(>=0) -> (1-val,1+val)', 'Tuple[Float,Float(>=0)] -> (val[0],val[1])'],
                          'saturation': ['Float(>=0) -> (1-val,1+val)', 'Tuple[Float,Float(>=0)] -> (val[0],val[1])']})
class ColorJitter(OneLineTransform):
    def __init__(
            self,
            brightness: Union[float, Sequence[float]],
            contrast: Union[float, Sequence[float]],
            saturation: Union[float, Sequence[float]],
    ):
        super().__init__(f"ColorJitter(brightness={brightness}, contrast={contrast}, saturation={saturation})")
        def _to_range_float(val):
            if isinstance(val, (float, int)):
                low = max(0., 1. - val)
                high = 1. + val
            else:
                assert len(val) == 2, f"Range must be a tuple of length 2: {val}"
                low = max(0, float(val[0]))
                high = max(0, float(val[1]))
            assert low <= high, f"low > high: {low} > {high}"
            high = low + 1e-6 if low + 1e-6 > high else high
            return (low, high)
        
        self.brightness = _to_range_float(brightness)
        self.contrast = _to_range_float(contrast)
        self.saturation = _to_range_float(saturation)

        self.brightness_sampler = torch.distributions.Uniform(*self.brightness)
        self.contrast_sampler = torch.distributions.Uniform(*self.contrast)
        self.saturation_sampler = torch.distributions.Uniform(*self.saturation)

    def __call__(self, img):
        img = F.adjust_brightness(img, self.brightness_sampler.sample((1,)).item())
        img = F.adjust_contrast(img, self.contrast_sampler.sample((1,)).item())
        img = F.adjust_saturation(img, self.saturation_sampler.sample((1,)).item())
        return img


@TRANSREG.reg(params={'kernel_size': 5, 'sigma': (0.1, 2.0)},
              param_desc={'kernel_size': ['Tuple[Odd,Odd(>=3)]', 'Odd(>=3)'], 'sigma': ['Tuple[Float,Float(>0)]', 'Float(>0)']})
class GaussianBlur(OneLineTransform):
    def __init__(self, kernel_size: Union[int, Sequence[int]], sigma: Union[float, Sequence[float]]):
        def _check_kernel_size(kernel_size):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            assert len(kernel_size) == 2 and all(isinstance(k, int) for k in kernel_size), \
                f"Kernel size must be a tuple of 2 integers: {kernel_size}"
            assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1, f"Kernel size must be odd: {kernel_size}"
            assert kernel_size[0] >= 3 and kernel_size[1] >= 3, f"Kernel size must be >= 3: {kernel_size}"
            return list(kernel_size)
        
        def _check_sigma(sigma):
            if isinstance(sigma, (int, float)):
                sigma = (sigma, sigma + 1e-6)  # for safe range check
            elif isinstance(sigma, (list, tuple)):
                assert len(sigma) == 2, f"Range must be a tuple of length 2: {sigma}"
                assert sigma[0] < sigma[1], f"low > high: {sigma[0]} > {sigma[1]}"
                sigma = (float(sigma[0]), float(sigma[0]) + 1e-6) if sigma[0] == sigma[1] else (float(sigma[0]), float(sigma[1]))
            return sigma

        self.kernel_size = _check_kernel_size(kernel_size)
        self.sigma = _check_sigma(sigma)
        self.sigma_sampler = torch.distributions.Uniform(*self.sigma)
        super().__init__(f"GaussianBlur(kernel_size={kernel_size}, sigma={sigma})")

    def __call__(self, img):
        sigma = self.sigma_sampler.sample((1,)).item()
        return F.gaussian_blur(img, kernel_size=self.kernel_size, sigma=[sigma, sigma])


@TRANSREG.reg(params={'degrees': inspect_empty, 'alpha': 2, 'beta': 2, 'interpolation': 'nearest'},
              param_desc={'degrees': ['Float', 'Tuple[Float, Float]'],
                          'alpha': 'Float (> 0)', 'beta': 'Float (> 0)',
                          'interpolation': ["nearest", "bilinear", "bicubic"]})
class RandomRotate(OneLineTransform):
    def __init__(self, degrees: Union[float, int, Sequence[float]],
                 alpha: float, beta: float, interpolation: str,
                 expand: bool = False, fill: int = 0):
        super().__init__(f"RandomRotate(degrees={degrees}, distribution=Beta({alpha},{beta}), interpolation={interpolation})")
        if isinstance(degrees, (float, int)):
            assert degrees > 0, "Degrees must be positive"
            degrees = (-degrees, degrees)
        else:
            assert len(degrees) == 2, "Degrees must be a tuple of length 2"
            degrees = tuple(float(d) for d in degrees)
        
        self.beta = torch.distributions.Beta(alpha, beta)
        self.degrees = degrees
        self.interpolation = getattr(F.InterpolationMode, interpolation.upper())
        self.expand = expand
        self.fill = fill

    def __call__(self, img):
        p = self.beta.sample((1,)).item()
        angle = self.degrees[0] * p + self.degrees[1] * (1 - p)
        return F.rotate(img, angle=angle, interpolation=self.interpolation,
                        expand=self.expand, fill=self.fill)
