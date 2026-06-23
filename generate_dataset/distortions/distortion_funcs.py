import cupy as cp
import numpy as np
import cv2
from PIL import Image

import skimage as sk
from skimage.filters import gaussian


from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from io import BytesIO
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import zoom as scizoom

import math

import collections
from pathlib import Path


FROST_FIG_DIR = Path(__file__).resolve().parent / "figs"

def get_distortion_methods():
    distortion_methods = collections.OrderedDict()
    ######## blur category
    distortion_methods['glass_blur'] = glass_blur
    distortion_methods['motion_blur'] = motion_blur
    distortion_methods['gaussian_blur'] = gaussian_blur
    distortion_methods['defocus_blur'] = defocus_blur

    ######## weather category
    distortion_methods['snow'] = snow
    distortion_methods['frost'] = frost
    distortion_methods['fog'] = fog
    distortion_methods['brightness'] = brightness
    distortion_methods['darkness'] = darkness

    ### digtial category
    distortion_methods['contrast'] = contrast
    distortion_methods['pixelate'] = pixelate
    distortion_methods['saturate'] = saturate
    distortion_methods['jpeg_compression'] = jpeg_compression

    ### noise category
    distortion_methods['gaussian_noise'] = gaussian_noise
    distortion_methods['impulse_noise'] = impulse_noise
    return distortion_methods



###########################################
def get_distortion_methods_total():
    distortion_methods = collections.OrderedDict()
    ######## blur category
    distortion_methods['glass_blur'] = glass_blur
    distortion_methods['motion_blur'] = motion_blur
    distortion_methods['gaussian_blur'] = gaussian_blur
    distortion_methods['defocus_blur'] = defocus_blur
    distortion_methods['zoom_blur'] = zoom_blur

    ######## weather category
    distortion_methods['snow'] = snow
    distortion_methods['frost'] = frost
    distortion_methods['fog'] = fog
    distortion_methods['brightness'] = brightness
    distortion_methods['darkness'] = darkness

    ### digtial category
    distortion_methods['contrast'] = contrast
    distortion_methods['elastic_transform'] = elastic_transform
    distortion_methods['pixelate'] = pixelate
    distortion_methods['saturate'] = saturate
    distortion_methods['jpeg_compression'] = jpeg_compression
    distortion_methods['spatter'] = spatter

    ### noise category
    distortion_methods['shot_noise'] = shot_noise
    distortion_methods['gaussian_noise'] = gaussian_noise
    distortion_methods['impulse_noise'] = impulse_noise
    distortion_methods['speckle_noise'] = speckle_noise
    # ['contrast', 'pixelate', 'elastic_transform', 'gaussian_noise', 'zoom_blur', 'motion_blur', 'snow', 'frost', 'saturate', 'glass_blur']

    # for method_name, distortion_fn in distortion_methods.items():
    #     print(method_name, distortion_fn)

    return distortion_methods

def get_distortion_methods1():
    distortion_methods = collections.OrderedDict()

    distortion_methods['gaussian_noise'] = gaussian_noise
    distortion_methods['glass_blur'] = glass_blur
    distortion_methods['motion_blur'] = motion_blur
    distortion_methods['zoom_blur'] = zoom_blur
    distortion_methods['snow'] = snow
    distortion_methods['frost'] = frost
    distortion_methods['contrast'] = contrast
    distortion_methods['elastic_transform'] = elastic_transform
    distortion_methods['pixelate'] = pixelate
    distortion_methods['saturate'] = saturate
    return distortion_methods

def get_distortion_methods2():
    distortion_methods = collections.OrderedDict()

    distortion_methods['shot_noise'] = shot_noise
    distortion_methods['impulse_noise'] = impulse_noise
    distortion_methods['defocus_blur'] = defocus_blur
    distortion_methods['fog'] = fog
    distortion_methods['brightness'] = brightness
    distortion_methods['darkness'] = darkness

    distortion_methods['jpeg_compression'] = jpeg_compression
    distortion_methods['speckle_noise'] = speckle_noise
    distortion_methods['gaussian_blur'] = gaussian_blur
    distortion_methods['spatter'] = spatter
    return distortion_methods

def gaussian_noise(x, c):
    x = cp.array(x) / 255.
    return cp.clip(x + cp.random.normal(size=x.shape, scale=c), 0, 1) * 255

def shot_noise(x, c):
    # c = [60, 25, 12, 5, 3][severity - 1]
    x = cp.array(x) / 255.
    return cp.clip(cp.random.poisson(x * c) / float(c), 0, 1) * 255



def _bernoulli(p, shape, *, random_state):
    if p == 0:
        return cp.zeros(shape, dtype=bool)
    if p == 1:
        return cp.ones(shape, dtype=bool)
    return random_state.random(shape) <= p
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):

    mode = mode.lower()

    # Detect if a signed image was input
    if image.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    # image = img_as_float(image)

    rng = cp.random.default_rng(seed)

    allowedtypes = {
        'gaussian': 'gaussian_values',
        'localvar': 'localvar_values',
        'poisson': 'poisson_values',
        'salt': 'sp_values',
        'pepper': 'sp_values',
        's&p': 's&p_values',
        'speckle': 'gaussian_values'}

    kwdefaults = {
        'mean': 0.,
        'var': 0.01,
        'amount': 0.05,
        'salt_vs_pepper': 0.5,
        'local_vars': cp.zeros_like(image) + 0.01}

    allowedkwargs = {
        'gaussian_values': ['mean', 'var'],
        'localvar_values': ['local_vars'],
        'sp_values': ['amount'],
        's&p_values': ['amount', 'salt_vs_pepper'],
        'poisson_values': []}

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[mode]]:
            raise ValueError(
                f"{key} keyword not in allowed keywords "
                f"{allowedkwargs[allowedtypes[mode]]}"
            )

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[mode]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    if mode == 'gaussian':
        noise = rng.normal(kwargs['mean'], kwargs['var'] ** 0.5, image.shape)
        out = image + noise

    elif mode == 'localvar':
        # Ensure local variance input is correct
        if (kwargs['local_vars'] <= 0).any():
            raise ValueError('All values of `local_vars` must be > 0.')

        # Safe shortcut usage broadcasts kwargs['local_vars'] as a ufunc
        out = image + rng.normal(0, kwargs['local_vars'] ** 0.5)

    elif mode == 'poisson':
        # Determine unique values in image & calculate the next power of two
        vals = len(cp.unique(image))
        vals = 2 ** cp.ceil(cp.log2(vals))

        # Ensure image is exclusively positive
        if low_clip == -1.:
            old_max = image.max()
            image = (image + 1.) / (old_max + 1.)

        # Generating noise for each unique value in image.
        out = rng.poisson(image * vals) / float(vals)

        # Return image to original range if input was signed
        if low_clip == -1.:
            out = out * (old_max + 1.) - 1.

    elif mode == 'salt':
        # Re-call function with mode='s&p' and p=1 (all salt noise)
        out = random_noise(image, mode='s&p', seed=rng,
                           amount=kwargs['amount'], salt_vs_pepper=1.)

    elif mode == 'pepper':
        # Re-call function with mode='s&p' and p=1 (all pepper noise)
        out = random_noise(image, mode='s&p', seed=rng,
                           amount=kwargs['amount'], salt_vs_pepper=0.)

    elif mode == 's&p':
        out = image.copy()
        p = kwargs['amount']
        q = kwargs['salt_vs_pepper']
        flipped = _bernoulli(p, image.shape, random_state=rng)
        salted = _bernoulli(q, image.shape, random_state=rng)
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = low_clip

    elif mode == 'speckle':
        noise = rng.normal(kwargs['mean'], kwargs['var'] ** 0.5, image.shape)
        out = image + image * noise

    # Clip back to original range, if necessary
    if clip:
        out = cp.clip(out, low_clip, 1.0)

    return out

def impulse_noise(x, c):
    # c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = random_noise(cp.array(x) / 255., mode='s&p', amount=c)
    return cp.clip(x, 0, 1) * 255

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def defocus_blur(x, c):
    # c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def glass_blur(x, c):
    sigma, max_delta, iterations = c
    max_delta, iterations = int(max_delta), int(iterations)
    x = np.uint8(gaussian(np.array(x) / 255., sigma=sigma, channel_axis=-1) * 255)
    x_shape = x.shape

    # locally shuffle pixels
    h_indices = np.tile(np.arange(x_shape[0]), (x_shape[1], 1)).T
    w_indices = np.tile(np.arange(x_shape[1]), (x_shape[0], 1))

    for i in range(iterations):
        dh, dw = np.random.randint(-max_delta, max_delta, size=(2, x_shape[0], x_shape[1]))
        h_prime = np.clip(h_indices + dh, 0, x_shape[0] - 1)
        w_prime = np.clip(w_indices + dw, 0, x_shape[1] - 1)
        x[h_indices, w_indices] = x[h_prime, w_prime]

    return np.clip(gaussian(x / 255., sigma=sigma, channel_axis=-1), 0, 1) * 255


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def motion_blur(x, c):
    # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if len(x.shape) == 3:
        return np.clip(x[..., [2, 1, 0]], 0, 255)
    # # if x.shape != (224, 224):
    # #     return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

def clipped_zoom(img, zoom_factor):
    # clipping along the width dimension:
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2

    # clipping along the height dimension:
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                  (zoom_factor, zoom_factor, 1), order=1)

    return img


def zoom_blur(x, params):
    # c = [np.arange(1, 1.11, 0.01),
    #      np.arange(1, 1.16, 0.01),
    #      np.arange(1, 1.21, 0.02),
    #      np.arange(1, 1.26, 0.02),
    #      np.arange(1, 1.31, 0.03)][severity - 1]
    c = np.arange(params[0], params[1], params[2])
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)

    # set_exception = False
    for zoom_factor in c:
        if len(x.shape) < 3 or x.shape[2] < 3:
            x_channels = np.array([x, x, x]).transpose((1, 2, 0))
            zoom_layer = clipped_zoom(x_channels, zoom_factor)
            zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], 0]
        else:
            zoom_layer = clipped_zoom(x, zoom_factor)
            zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]

        try:
            out += zoom_layer
        except ValueError:
            # set_exception = True
            out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer

    # if set_exception:
    #     print('ValueError for zoom blur, Exception handling')
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255

def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1

def gauss_function(x, mean, sigma):
    return (np.exp(- x**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)

def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k/Z

def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted
def _motion_blur(x, radius, sigma, angle):
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(int(width)):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
            # simulated motion exceeded image borders
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred

def snow(x, c):
    # c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
    #      (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
    #      (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
    #      (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
    #      (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0],
                                  scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = np.clip(snow_layer.squeeze(), 0, 1)

    snow_layer = _motion_blur(snow_layer, radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    # The snow layer is rounded and cropped to the img dims
    snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    snow_layer = snow_layer[:x.shape[0], :x.shape[1], :]

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, x.reshape(x.shape[0],
                                                            x.shape[
                                                                1]) * 1.5 + 0.5)
        snow_layer = snow_layer.squeeze(-1)
    else:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x,
                                                               cv2.COLOR_RGB2GRAY).reshape(
            x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    try:
        return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    except ValueError:
        # print('ValueError for Snow, Exception handling')
        x[:snow_layer.shape[0], :snow_layer.shape[1]] += snow_layer + np.rot90(
            snow_layer, k=2)
        return np.clip(x, 0, 1) * 255

def rgb2gray(rgb):
    return cp.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def frost(x, c):
    # c = [(1, 0.4),
    #      (0.8, 0.6),
    #      (0.7, 0.7),
    #      (0.65, 0.7),
    #      (0.6, 0.75)][severity - 1]
    # idx = np.random.randint(6)
    filename = [
        FROST_FIG_DIR / 'frost1.png',
        FROST_FIG_DIR / 'frost2.png',
        FROST_FIG_DIR / 'frost3.jpg',
        FROST_FIG_DIR / 'frost4.jpg',
        FROST_FIG_DIR / 'frost5.jpg',
    ]
    frost = cv2.imread(str(filename[c[2] - 1]))
    frost_shape = frost.shape
    x_shape = np.array(x).shape

    # resize the frost image so it fits to the image dimensions
    scaling_factor = 1
    if frost_shape[0] >= x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = 1
    elif frost_shape[0] < x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = x_shape[0] / frost_shape[0]
    elif frost_shape[0] >= x_shape[0] and frost_shape[1] < x_shape[1]:
        scaling_factor = x_shape[1] / frost_shape[1]
    elif frost_shape[0] < x_shape[0] and frost_shape[1] < x_shape[
        1]:  # If both dims are too small, pick the bigger scaling factor
        scaling_factor_0 = x_shape[0] / frost_shape[0]
        scaling_factor_1 = x_shape[1] / frost_shape[1]
        scaling_factor = np.maximum(scaling_factor_0, scaling_factor_1)

    scaling_factor *= 1.1
    new_shape = (int(np.ceil(frost_shape[1] * scaling_factor)),
                 int(np.ceil(frost_shape[0] * scaling_factor)))
    frost_rescaled = cv2.resize(frost, dsize=new_shape,
                                interpolation=cv2.INTER_CUBIC)

    # randomly crop
    x_start, y_start = np.random.randint(0, frost_rescaled.shape[0] - x_shape[
        0]), np.random.randint(0, frost_rescaled.shape[1] - x_shape[1])

    if len(x_shape) < 3 or x_shape[2] < 3:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                         y_start:y_start + x_shape[1]]
        frost_rescaled = rgb2gray(frost_rescaled)
    else:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                         y_start:y_start + x_shape[1]][..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * frost_rescaled, 0, 255)


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = cp.empty((mapsize, mapsize), dtype=cp.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * cp.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + cp.roll(cornerref, shift=-1, axis=0)
        squareaccum += cp.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + cp.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + cp.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + cp.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + cp.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def fog(x, c):
    # c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    shape = cp.array(x).shape
    max_side = max(shape)
    map_size = next_power_of_2(int(max_side))

    x = cp.array(x) / 255.
    max_val = x.max()

    x_shape = cp.array(x).shape
    if len(x_shape) < 3 or x_shape[2] < 3:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0], :shape[1]]
    else:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0], :shape[1]][..., cp.newaxis]
    return cp.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def brightness(x,c):
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = cp.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return cp.clip(x, 0, 1) * 255

def darkness(x, c):
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = cp.clip(x[:, :, 2] - c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return cp.clip(x, 0, 1) * 255

def contrast(x, c):
    # c = [0.4, .3, .2, .1, .05][severity - 1]

    x = cp.array(x) / 255.
    means = cp.mean(x, axis=(0, 1), keepdims=True)
    return cp.clip((x - means) * c + means, 0, 1) * 255

def elastic_transform(image, c):
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    sigma = np.array(shape_size) * 0.01
    # alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][severity - 1]
    alpha = 250 * c
    max_dx = shape[0] * 0.005
    max_dy = shape[0] * 0.005

    dx = (gaussian(np.random.uniform(-max_dx, max_dx, size=shape[:2]),
                   sigma, mode='reflect', truncate=3) * alpha).astype(
        np.float32)
    dy = (gaussian(np.random.uniform(-max_dy, max_dy, size=shape[:2]),
                   sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)

    if len(image.shape) < 3 or image.shape[2] < 3:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    else:
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                              np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx,
                                                          (-1, 1)), np.reshape(
            z, (-1, 1))
    return np.clip(
        map_coordinates(image, indices, order=1, mode='reflect').reshape(
            shape), 0, 1) * 255

def pixelate(x, c):
    # c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x_shape = cp.array(x).shape

    x = x.resize((int(x_shape[1] * c), int(x_shape[0] * c)), Image.BOX)

    x = x.resize((x_shape[1], x_shape[0]), Image.NEAREST)

    return x

def jpeg_compression(x, c):
    # c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    gray_scale = False
    if x.mode != 'RGB':
        gray_scale = True
        x = x.convert('RGB')
    x.save(output, 'JPEG', quality=int(c))
    x = Image.open(output)
    if gray_scale:
        x = x.convert('L')

    return x


def speckle_noise(x, c):
    # c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = cp.array(x) / 255.
    return cp.clip(x + x * cp.random.normal(size=x.shape, scale=c), 0, 1) * 255

def gaussian_blur(x, c):
    # c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255

def spatter(x, c):
    # c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
    #      (0.65, 0.3, 3, 0.68, 0.6, 0),
    #      (0.65, 0.3, 2, 0.68, 0.5, 0),
    #      (0.65, 0.3, 1, 0.65, 1.5, 1),
    #      (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x_PIL = x
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if int(c[5]) == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)

        if len(x.shape) < 3 or x.shape[2] < 3:
            add_spatter_color = cv2.cvtColor(np.clip(m * color, 0, 1),
                                             cv2.COLOR_BGRA2BGR)
            add_spatter_gray = rgb2gray(add_spatter_color)

            return np.clip(x + add_spatter_gray, 0, 1) * 255

        else:

            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return cv2.cvtColor(np.clip(x + m * color, 0, 1),
                                cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        x_rgb = np.array(x_PIL.convert('RGB'))

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x_rgb[..., :1]),
                                42 / 255. * np.ones_like(x_rgb[..., :1]),
                                20 / 255. * np.ones_like(x_rgb[..., :1])),
                               axis=2)
        color *= m[..., np.newaxis]
        if len(x.shape) < 3 or x.shape[2] < 3:
            x *= (1 - m)
            return np.clip(x + rgb2gray(color), 0, 1) * 255

        else:
            x *= (1 - m[..., np.newaxis])
            return np.clip(x + color, 0, 1) * 255



def saturate(x, c):
    # c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.

    gray_scale = False
    if len(x.shape) < 3 or x.shape[2] < 3:
        x = cp.array([x, x, x]).transpose((1, 2, 0))
        gray_scale = True
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = cp.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    if gray_scale:
        x = x[:, :, 0]

    return cp.clip(x, 0, 1) * 255

