import cv2
import numpy as np
import os


def load_and_preprocess(image_path, size=512):
    """
    Load image, resize to fixed size, convert to grayscale.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def compute_fft_magnitude(gray_image):
    """
    Compute log-magnitude of centered 2D FFT.
    """
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)
    magnitude_log = np.log1p(magnitude)

    return magnitude_log


def radial_profile(magnitude_spectrum):
    """
    Compute radial average of frequency magnitude.
    """
    h, w = magnitude_spectrum.shape
    center = (h // 2, w // 2)

    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())

    radialprofile = tbin / (nr + 1e-8)

    return radialprofile


def normalize_spectrum(spec):
    """
    Normalize spectrum to [0,1] for consistent visualization.
    """
    spec = spec - np.min(spec)
    spec = spec / (np.max(spec) + 1e-8)
    return spec


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)