"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PIXLENS — filters.py                                           ║
║              Pixel Intelligence & Learning ENhancement System               ║
║                                                                              ║
║  This file contains EVERY image processing algorithm for PIXLENS.           ║
║  All operations are implemented using NumPy mathematics —                   ║
║  no black-box OpenCV filter calls. This is pure DIVP theory in code.        ║
║                                                                              ║
║  CATEGORIES:                                                                 ║
║    1.  Utility Functions        — helpers used by all filters               ║
║    2.  Spatial Domain Filters   — convolution, smoothing, sharpening        ║
║    3.  Edge Detection           — Sobel, Prewitt, Roberts, LoG, Canny       ║
║    4.  Frequency Domain         — FFT-based low/high/band pass filters      ║
║    5.  Histogram Operations     — equalization, stretching, gamma           ║
║    6.  Noise Models             — Gaussian, S&P, Poisson, Speckle           ║
║    7.  Image Restoration        — Wiener, adaptive median, constrained LS   ║
║    8.  Morphological Operations — erosion, dilation, opening, closing, etc. ║
║    9.  Color Processing         — RGB↔HSI, RGB↔HSV, pseudo-color           ║
║   10.  Image Transforms         — DCT, Walsh-Hadamard, Haar Wavelet         ║
║   11.  Main Dispatch            — apply_filter() called by Flask server     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from PIL import Image
import io
import base64
import time

# Optional: only used in a few specific operations
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from scipy.ndimage import median_filter as scipy_median
    from scipy.signal import wiener as scipy_wiener
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def img_to_array(pil_img):
    """
    Convert a PIL Image to a float64 NumPy array with shape (H, W, 3).
    Values range from 0.0 to 255.0.
    Forces RGB mode so all images have exactly 3 channels (no RGBA/grayscale issues).
    """
    return np.array(pil_img.convert('RGB'), dtype=np.float64)


def array_to_pil(arr):
    """
    Convert a NumPy array back to a PIL Image.
    Clips values to [0, 255] and converts to uint8 first.
    """
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped)


def array_to_b64(arr):
    """
    Convert a NumPy array to a base64-encoded PNG string.
    This is how the processed image is sent from Python backend to the browser.

    Returns a string like: 'iVBORw0KGgoAAAANSUhEUg...'
    The browser prefixes this with 'data:image/png;base64,' to display it.
    """
    pil_img = array_to_pil(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def to_grayscale(image):
    """
    Convert an RGB image array to a 2D grayscale array.
    Uses ITU-R BT.601 luminance weights:
        Y = 0.299*R + 0.587*G + 0.114*B
    These weights reflect human eye sensitivity (most sensitive to green).
    """
    return 0.299 * image[:, :, 0] + \
           0.587 * image[:, :, 1] + \
           0.114 * image[:, :, 2]


def gray_to_rgb(gray):
    """
    Stack a 2D grayscale array into a 3-channel (H, W, 3) array.
    Used to convert grayscale results back to the expected 3-channel format.
    """
    return np.stack([gray, gray, gray], axis=2)


def pad_image(image, pad_h, pad_w, mode='reflect'):
    """
    Pad an image to handle convolution boundary effects.

    Modes:
        'reflect'  — mirror pixels at borders (avoids dark edges)
        'constant' — fill with zeros (black borders)
        'replicate'— repeat edge pixels

    For a 3-channel image, pads all channels simultaneously.
    """
    if image.ndim == 3:
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=mode)
    else:
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)


def normalize_to_255(array):
    """
    Linearly rescale any array so its values span [0, 255].
    Used for displaying edge magnitude maps and frequency spectra.
    """
    arr_min = array.min()
    arr_max = array.max()
    if arr_max == arr_min:
        return np.zeros_like(array)
    return (array - arr_min) / (arr_max - arr_min) * 255.0


def compute_psnr(original, processed):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between original and processed image.

    Formula:  PSNR = 20 * log10(MAX_I / sqrt(MSE))
    where MAX_I = 255 for 8-bit images.

    Higher PSNR = processed image is closer to original (better quality).
    PSNR > 40 dB  → excellent quality
    PSNR 30–40 dB → good quality
    PSNR < 30 dB  → visible degradation

    Returns:
        (psnr: float, mse: float)
    """
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    mse  = np.mean((orig - proc) ** 2)
    if mse == 0:
        return float('inf'), 0.0
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    return round(psnr, 4), round(float(mse), 4)


def compute_metrics(original, processed):
    """
    Compute all quality metrics comparing original to processed image.

    Returns a dict with:
        psnr     — Peak Signal-to-Noise Ratio (dB)
        mse      — Mean Squared Error
        mae      — Mean Absolute Error
        std_dev  — Standard deviation of the processed image
    """
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    diff = orig - proc
    mse  = float(np.mean(diff ** 2))
    mae  = float(np.mean(np.abs(diff)))
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    std  = float(np.std(proc))
    return {
        'psnr':    round(psnr, 4),
        'mse':     round(mse, 4),
        'mae':     round(mae, 4),
        'std_dev': round(std, 4)
    }


def build_gaussian_kernel(size, sigma):
    """
    Build a 2D Gaussian kernel of given size and sigma.

    Formula:  G(x,y) = exp(-(x²+y²) / (2σ²))
    Kernel is normalised so all values sum to 1 (no brightness change).

    Args:
        size  (int)   — kernel width and height (must be odd, e.g. 3, 5, 7)
        sigma (float) — controls spread; larger sigma = more blur

    Returns:
        2D NumPy array of shape (size, size)
    """
    half = size // 2
    ax   = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def convolve2d(image, kernel, pad_mode='reflect'):
    """
    Perform manual 2D convolution of an image with a kernel.

    This is the CORE operation of spatial domain filtering.
    For every pixel (i, j), we:
        1. Extract a region of the same size as the kernel centered at (i, j)
        2. Multiply element-wise with the kernel
        3. Sum all products → that is the new pixel value

    Mathematical definition:
        g(x,y) = Σ Σ f(x+m, y+n) · h(m, n)

    Args:
        image    — NumPy array (H, W, 3) or (H, W)
        kernel   — 2D NumPy array (kH, kW)
        pad_mode — boundary handling mode

    Returns:
        Filtered image as float64 array, same shape as input.
    """
    kh, kw   = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    is_color = (image.ndim == 3)
    padded   = pad_image(image, pad_h, pad_w, mode=pad_mode)
    output   = np.zeros_like(image, dtype=np.float64)

    if is_color:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i + kh, j:j + kw]         # (kH, kW, 3)
                for c in range(3):
                    output[i, j, c] = np.sum(region[:, :, c] * kernel)
    else:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i + kh, j:j + kw]
                output[i, j] = np.sum(region * kernel)

    return output


def fast_convolve(image, kernel):
    """
    Faster 2D convolution using numpy stride tricks + einsum.
    Produces same results as convolve2d but significantly faster on large images.
    Used automatically when image width > 256 pixels.
    """
    from numpy.lib.stride_tricks import as_strided

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W   = image.shape[:2]

    is_color = (image.ndim == 3)
    padded   = pad_image(image, ph, pw, mode='reflect')
    output   = np.zeros_like(image, dtype=np.float64)

    if is_color:
        for c in range(3):
            chan = padded[:, :, c]
            shape   = (H, W, kh, kw)
            strides = (chan.strides[0], chan.strides[1], chan.strides[0], chan.strides[1])
            windows = as_strided(chan, shape=shape, strides=strides)
            output[:, :, c] = np.einsum('ijkl,kl->ij', windows, kernel)
    else:
        shape   = (H, W, kh, kw)
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        windows = as_strided(padded, shape=shape, strides=strides)
        output  = np.einsum('ijkl,kl->ij', windows, kernel)

    return output


def smart_convolve(image, kernel):
    """
    Automatically chooses between fast_convolve and convolve2d
    based on image size. Use this everywhere instead of calling either directly.
    """
    if image.shape[1] > 256:
        return fast_convolve(image, kernel)
    return convolve2d(image, kernel)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SPATIAL DOMAIN FILTERS
# ═════════════════════════════════════════════════════════════════════════════

def mean_filter(image, ksize=3):
    """
    Mean (Box) Filter — simplest smoothing filter.

    Replaces each pixel with the AVERAGE of all pixels in the kernel window.
    The kernel is a flat matrix where every value = 1/(ksize × ksize).

    Effect: Blurs the image. Reduces noise but also blurs edges.
    DIVP Topic: Spatial domain filtering, smoothing filters (Ch. 3, Gonzalez)

    Args:
        ksize (int) — kernel size, must be odd (3, 5, 7, ...)
    """
    ksize  = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    return smart_convolve(image, kernel)


def gaussian_filter(image, ksize=5, sigma=1.5):
    """
    Gaussian Filter — most commonly used smoothing filter.

    Unlike the mean filter, nearby pixels contribute more than distant ones.
    The weights follow a Gaussian (bell curve) distribution.

    Kernel formula:  G(x,y) = (1/2πσ²) · exp(-(x²+y²)/2σ²)

    Effect: Smooth blur. Better edge preservation than mean filter.
           Larger sigma → more blur. Larger ksize → wider influence area.
    DIVP Topic: Gaussian smoothing, low-pass spatial filtering
    """
    ksize = max(3, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    sigma  = max(0.1, float(sigma))
    kernel = build_gaussian_kernel(ksize, sigma)
    return smart_convolve(image, kernel)


def median_filter(image, ksize=3):
    """
    Median Filter — best filter for removing Salt & Pepper (impulse) noise.

    Instead of averaging, it replaces each pixel with the MEDIAN value
    of all pixels in the kernel window. Since extreme values (0=pepper, 255=salt)
    are at the ends of the sorted list, the median ignores them completely.

    Effect: Removes salt & pepper noise without blurring edges.
           This is a NON-LINEAR filter — no convolution kernel involved.
    DIVP Topic: Order-statistic (rank) filters, non-linear filtering
    """
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1

    H, W = image.shape[:2]
    half  = ksize // 2
    padded = pad_image(image, half, half, mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(H):
        for j in range(W):
            region = padded[i:i + ksize, j:j + ksize]
            if image.ndim == 3:
                for c in range(3):
                    output[i, j, c] = np.median(region[:, :, c])
            else:
                output[i, j] = np.median(region)

    return output


def bilateral_filter(image, ksize=5, sigma_space=1.5, sigma_color=50.0):
    """
    Bilateral Filter — edge-preserving smoothing filter.

    Gaussian filter blurs edges. The bilateral filter adds a second Gaussian
    weight based on pixel value difference — so pixels with very different
    colors contribute less, preserving sharp edges.

    Two weight components:
        spatial weight   — nearby pixels weighted higher (like Gaussian)
        intensity weight — similar-color pixels weighted higher

    Effect: Smoothing without blurring edges. Used in photo editing.
    DIVP Topic: Non-linear spatial filtering, edge-preserving smoothing
    """
    ksize = max(3, int(ksize))
    if ksize % 2 == 0:
        ksize += 1

    half   = ksize // 2
    H, W   = image.shape[:2]
    padded = pad_image(image, half, half, mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)

    # Precompute spatial Gaussian weights
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    spatial_w = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_space ** 2))

    for i in range(H):
        for j in range(W):
            region = padded[i:i + ksize, j:j + ksize].astype(np.float64)
            for c in range(3):
                center      = image[i, j, c]
                intensity_w = np.exp(-((region[:, :, c] - center) ** 2) / (2.0 * sigma_color ** 2))
                combined_w  = spatial_w * intensity_w
                w_sum       = combined_w.sum()
                output[i, j, c] = np.sum(region[:, :, c] * combined_w) / (w_sum if w_sum > 0 else 1)

    return output


def sharpen_filter(image, amount=1.5):
    """
    Unsharp Masking — classic image sharpening technique.

    Formula:  sharpened = original + amount × (original − blurred)
    The (original − blurred) part isolates the edges/details (high frequencies).
    Adding them back to the original enhances perceived sharpness.

    When amount = 1: standard unsharp mask
    When amount > 1: high-boost filtering (over-sharpened, more aggressive)

    Effect: Makes edges crisper, details more visible.
    DIVP Topic: Unsharp masking, high-boost filtering, spatial enhancement
    """
    blurred   = gaussian_filter(image, ksize=5, sigma=1.0)
    detail    = image.astype(np.float64) - blurred
    sharpened = image.astype(np.float64) + float(amount) * detail
    return sharpened


def laplacian_filter(image, kernel_type='4-connected', scale=1.0):
    """
    Laplacian Filter — second-order derivative edge enhancement.

    The Laplacian measures the rate of change of the gradient (2nd derivative).
    It highlights regions of rapid intensity change (edges, fine details).

    4-connected kernel (detects horizontal + vertical edges):
        [ 0  -1   0]
        [-1   4  -1]
        [ 0  -1   0]

    8-connected kernel (detects diagonal edges too):
        [-1  -1  -1]
        [-1   8  -1]
        [-1  -1  -1]

    The Laplacian response is ADDED back to the original to sharpen it.

    Effect: Enhances edges and fine detail.
    DIVP Topic: Second-derivative operators, Laplacian of image, spatial sharpening
    """
    if kernel_type == '8-connected':
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float64)
    else:  # 4-connected
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]], dtype=np.float64)

    lap    = smart_convolve(image, kernel)
    result = image.astype(np.float64) + float(scale) * lap
    return result


def high_boost_filter(image, blur_sigma=1.0, boost_factor=2.0):
    """
    High-Boost Filter — controlled version of sharpening.

    Formula:  output = (boost_factor) × original − blurred
    When boost_factor = 1: identical to unsharp mask
    When boost_factor > 1: increases the contribution of high frequencies

    Effect: Tunable sharpening; larger boost = more aggressive enhancement.
    DIVP Topic: High-boost filtering, frequency emphasis
    """
    blurred = gaussian_filter(image, ksize=5, sigma=blur_sigma)
    return float(boost_factor) * image.astype(np.float64) - blurred


def average_filter_3x3(image):
    """
    Fixed 3×3 averaging filter. Simple, fast, no parameters.
    Good for a quick demo of the fundamental convolution concept.
    """
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    return smart_convolve(image, kernel)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EDGE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def roberts_filter(image):
    """
    Roberts Cross Operator — oldest and simplest edge detector.

    Uses two 2×2 kernels to compute diagonal gradients:
        Gx = [+1  0]     Gy = [ 0 +1]
             [ 0 -1]          [-1  0]

    Gradient magnitude:  |G| = sqrt(Gx² + Gy²)

    Effect: Fast but noisy; sensitive to diagonal edges.
    DIVP Topic: First-order derivative edge detection, gradient operators
    """
    gray    = to_grayscale(image)
    Kx = np.array([[1,  0], [0, -1]], dtype=np.float64)
    Ky = np.array([[0,  1], [-1, 0]], dtype=np.float64)

    padded = pad_image(gray, 1, 1, mode='reflect')
    H, W   = gray.shape
    gx = np.zeros((H, W), dtype=np.float64)
    gy = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            region = padded[i:i + 2, j:j + 2]
            gx[i, j] = np.sum(region * Kx)
            gy[i, j] = np.sum(region * Ky)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return gray_to_rgb(normalize_to_255(magnitude))


def prewitt_filter(image, direction='Combined'):
    """
    Prewitt Operator — improved gradient-based edge detector.

    Uses two 3×3 kernels to detect horizontal and vertical edges:
        Kx (vertical edges):       Ky (horizontal edges):
        [-1  0 +1]                 [-1 -1 -1]
        [-1  0 +1]                 [ 0  0  0]
        [-1  0 +1]                 [+1 +1 +1]

    Each kernel has equal weights (unlike Sobel which emphasises center row).

    direction: 'Combined', 'X only' (vertical edges), 'Y only' (horizontal edges)
    DIVP Topic: First-order derivative, gradient operators, edge detection
    """
    gray = to_grayscale(image)
    Kx   = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    Ky   = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

    gx = fast_convolve(gray, Kx)
    gy = fast_convolve(gray, Ky)

    if direction == 'X only':
        mag = np.abs(gx)
    elif direction == 'Y only':
        mag = np.abs(gy)
    else:
        mag = np.sqrt(gx ** 2 + gy ** 2)

    return gray_to_rgb(normalize_to_255(mag))


def sobel_filter(image, direction='Combined'):
    """
    Sobel Operator — most widely used gradient-based edge detector.

    Similar to Prewitt but gives more weight to the center row/column,
    providing better smoothing and noise resistance:
        Kx (detects vertical edges):    Ky (detects horizontal edges):
        [-1  0 +1]                      [-1 -2 -1]
        [-2  0 +2]                      [ 0  0  0]
        [-1  0 +1]                      [+1 +2 +1]

    Gradient magnitude:  |G| = sqrt(Gx² + Gy²)
    Gradient direction:  θ = arctan(Gy / Gx)  — used in Canny

    direction: 'Combined', 'X only', 'Y only'
    DIVP Topic: Sobel operator, gradient, first-order derivative
    """
    gray = to_grayscale(image)
    Kx   = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky   = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = fast_convolve(gray, Kx)
    gy = fast_convolve(gray, Ky)

    if direction == 'X only':
        mag = np.abs(gx)
    elif direction == 'Y only':
        mag = np.abs(gy)
    else:
        mag = np.sqrt(gx ** 2 + gy ** 2)

    return gray_to_rgb(normalize_to_255(mag))


def log_filter(image, sigma=1.5):
    """
    Laplacian of Gaussian (LoG) — Marr-Hildreth Edge Detector.

    Two-stage operation:
        Stage 1: Gaussian smoothing — suppresses noise
        Stage 2: Laplacian          — detects edges via zero crossings

    The combined LoG kernel is computed directly as:
        LoG(x,y) = -[1/(πσ⁴)] · [1 - (x²+y²)/(2σ²)] · exp(-(x²+y²)/(2σ²))

    Edges appear as ZERO CROSSINGS in the LoG response.

    Effect: Good edge detector, smoother than Sobel for noisy images.
    DIVP Topic: LoG, Marr-Hildreth, Laplacian, second-order derivative
    """
    # Step 1: Gaussian smooth
    smoothed = gaussian_filter(image, ksize=max(3, int(4 * sigma + 1) | 1), sigma=sigma)

    # Step 2: Laplacian on smoothed image
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    gray   = to_grayscale(smoothed)
    log_resp = fast_convolve(gray, kernel)

    # Normalize for display
    return gray_to_rgb(normalize_to_255(np.abs(log_resp)))


def canny_edge(image, sigma=1.0, low_thresh=20.0, high_thresh=60.0):
    """
    Canny Edge Detector — Full 5-Stage Pipeline Implemented From Scratch.

    John Canny (1986) — the gold standard edge detector.
    Criteria: Good detection, good localisation, single response per edge.

    ─── STAGE 1: Gaussian Smoothing ───────────────────────────────────────
    Suppress noise before gradient computation. σ controls smoothing amount.

    ─── STAGE 2: Gradient Magnitude & Direction ───────────────────────────
    Sobel operators compute Gx and Gy at every pixel.
        |G|  = sqrt(Gx² + Gy²)    — edge strength
        θ    = arctan(Gy/Gx)      — edge direction (in degrees)

    ─── STAGE 3: Non-Maximum Suppression (NMS) ────────────────────────────
    For each pixel, check its two neighbours in the gradient direction.
    If this pixel is NOT the local maximum → suppress it to 0.
    Result: edges become 1-pixel thin.

    ─── STAGE 4: Double Thresholding ──────────────────────────────────────
    Classify each surviving pixel as:
        STRONG (255) — magnitude ≥ high_thresh
        WEAK   (80)  — low_thresh ≤ magnitude < high_thresh
        NONE   (0)   — magnitude < low_thresh

    ─── STAGE 5: Edge Tracking by Hysteresis ──────────────────────────────
    WEAK pixels adjacent (8-connected) to STRONG pixels → promoted to STRONG.
    WEAK pixels not connected to any STRONG pixel → discarded (set to 0).

    Effect: Clean, thin, well-connected edges.
    DIVP Topic: Canny algorithm, NMS, hysteresis, multi-stage edge detection
    """
    gray = to_grayscale(image)
    H, W = gray.shape

    # ── Stage 1: Gaussian Smoothing ──────────────────────────────────────
    ksize    = max(3, int(4 * sigma + 1) | 1)   # Ensure odd kernel size
    smooth_k = build_gaussian_kernel(ksize, sigma)
    smoothed = fast_convolve(gray, smooth_k)

    # ── Stage 2: Gradient using Sobel ────────────────────────────────────
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx  = fast_convolve(smoothed, Kx)
    gy  = fast_convolve(smoothed, Ky)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180   # Angle in [0, 180)

    # ── Stage 3: Non-Maximum Suppression ─────────────────────────────────
    nms = np.zeros((H, W), dtype=np.float64)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = ang[i, j]
            m = mag[i, j]
            # Quantise angle to one of 4 directions: 0°, 45°, 90°, 135°
            if (a < 22.5) or (a >= 157.5):                  # Horizontal
                n1, n2 = mag[i, j + 1], mag[i, j - 1]
            elif (22.5 <= a < 67.5):                         # Diagonal /
                n1, n2 = mag[i + 1, j - 1], mag[i - 1, j + 1]
            elif (67.5 <= a < 112.5):                        # Vertical
                n1, n2 = mag[i + 1, j], mag[i - 1, j]
            else:                                            # Diagonal \
                n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]

            nms[i, j] = m if (m >= n1 and m >= n2) else 0.0

    # ── Stage 4: Double Thresholding ─────────────────────────────────────
    STRONG = 255
    WEAK   = 80
    edges  = np.zeros((H, W), dtype=np.uint8)
    edges[nms >= high_thresh]                          = STRONG
    edges[(nms >= low_thresh) & (nms < high_thresh)]   = WEAK

    # ── Stage 5: Hysteresis Edge Tracking ────────────────────────────────
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if edges[i, j] == WEAK:
                # Check all 8 neighbours for a STRONG pixel
                neighbourhood = edges[i - 1:i + 2, j - 1:j + 2]
                if STRONG in neighbourhood:
                    edges[i, j] = STRONG
                else:
                    edges[i, j] = 0

    return gray_to_rgb(edges.astype(np.float64))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FREQUENCY DOMAIN FILTERS (FFT-Based)
# ═════════════════════════════════════════════════════════════════════════════

def fft_spectrum(image):
    """
    Compute and return the log-magnitude FFT spectrum for visualisation.

    The 2D DFT F(u,v) represents the image in the FREQUENCY DOMAIN:
        Low frequencies (center) = smooth regions
        High frequencies (edges) = sharp edges, fine detail, noise

    Log scale used because magnitudes span many orders of magnitude:
        D(u,v) = log(1 + |F(u,v)|)

    Returns:
        magnitude_spectrum as a displayable RGB image (normalised 0–255)
    DIVP Topic: 2D DFT, frequency spectrum, Fourier transform
    """
    gray = to_grayscale(image)
    F    = np.fft.fftshift(np.fft.fft2(gray))
    spec = np.log1p(np.abs(F))                 # log(1 + |F|) for display
    return gray_to_rgb(normalize_to_255(spec))


def _build_frequency_mask(rows, cols, filter_type, cutoff, order=2):
    """
    Internal helper: build the filter mask H(u,v) in the frequency domain.

    D(u,v) = distance from frequency center (u=0, v=0 after fftshift)

    Filter types:
        ideal_lp       — H = 1 if D ≤ D0 else 0        (brick-wall, causes ringing)
        ideal_hp       — H = 0 if D ≤ D0 else 1
        butterworth_lp — H = 1 / [1 + (D/D0)^2n]       (smooth rolloff, no ringing)
        butterworth_hp — H = 1 / [1 + (D0/D)^2n]
        gaussian_lp    — H = exp(-D²/2D0²)              (smoothest rolloff)
        gaussian_hp    — H = 1 - exp(-D²/2D0²)
        band_pass      — passes a ring of frequencies around cutoff
        band_stop      — blocks a ring of frequencies (notch-like)
    """
    cx, cy = cols // 2, rows // 2
    u = np.arange(cols) - cx
    v = np.arange(rows) - cy
    U, V = np.meshgrid(u, v)
    D    = np.sqrt(U ** 2 + V ** 2)
    D[D == 0] = 1e-10    # avoid division by zero at center

    D0 = float(cutoff)
    n  = float(order)

    if filter_type == 'ideal_lp':
        H = (D <= D0).astype(np.float64)
    elif filter_type == 'ideal_hp':
        H = (D > D0).astype(np.float64)
    elif filter_type == 'butterworth_lp':
        H = 1.0 / (1.0 + (D / D0) ** (2 * n))
    elif filter_type == 'butterworth_hp':
        H = 1.0 / (1.0 + (D0 / D) ** (2 * n))
    elif filter_type == 'gaussian_lp':
        H = np.exp(-(D ** 2) / (2.0 * D0 ** 2))
    elif filter_type == 'gaussian_hp':
        H = 1.0 - np.exp(-(D ** 2) / (2.0 * D0 ** 2))
    elif filter_type == 'band_pass':
        W  = D0 * 0.5   # bandwidth = half the cutoff
        H  = 1.0 / (1.0 + ((D * W) / (D ** 2 - D0 ** 2 + 1e-10)) ** (2 * n))
    elif filter_type == 'band_stop':
        W  = D0 * 0.5
        H  = 1.0 - 1.0 / (1.0 + ((D * W) / (D ** 2 - D0 ** 2 + 1e-10)) ** (2 * n))
    else:
        H = np.ones((rows, cols), dtype=np.float64)

    return H


def fft_filter(image, filter_type='gaussian_lp', cutoff=30, order=2):
    """
    Frequency Domain Filtering Pipeline.

    Full pipeline for every frequency domain filter:
        Step 1: For each channel, compute 2D FFT → F(u,v)
        Step 2: Shift zero frequency to center → fftshift
        Step 3: Multiply by filter mask H(u,v) → G(u,v) = F(u,v) · H(u,v)
        Step 4: Inverse shift → ifftshift
        Step 5: Inverse FFT → g(x,y) = IDFT[G(u,v)]
        Step 6: Take real part (small imaginary parts from floating-point errors)

    Available filter types:
        'ideal_lp'       — Ideal Low-Pass  (sharp cutoff, Gibbs ringing)
        'ideal_hp'       — Ideal High-Pass
        'butterworth_lp' — Butterworth Low-Pass  (smooth rolloff, no ringing)
        'butterworth_hp' — Butterworth High-Pass
        'gaussian_lp'    — Gaussian Low-Pass     (smoothest, no ringing)
        'gaussian_hp'    — Gaussian High-Pass
        'band_pass'      — Band-Pass (passes frequencies near cutoff)
        'band_stop'      — Band-Stop / Notch

    Args:
        cutoff (int)   — D0: cutoff frequency in pixels from center
        order  (int)   — n: filter order (Butterworth only; higher = sharper rolloff)

    DIVP Topic: 2D DFT, frequency domain filtering, convolution theorem
    """
    rows, cols = image.shape[:2]
    H_mask = _build_frequency_mask(rows, cols, filter_type, cutoff, order)
    result  = np.zeros_like(image, dtype=np.float64)

    for c in range(3):
        channel     = image[:, :, c].astype(np.float64)
        F           = np.fft.fftshift(np.fft.fft2(channel))     # FFT + shift
        F_filtered  = F * H_mask                                  # Apply mask
        channel_out = np.fft.ifft2(np.fft.ifftshift(F_filtered)) # Inverse FFT
        result[:, :, c] = np.abs(channel_out)                     # Take magnitude

    return result


def ideal_lpf(image, cutoff=30):
    """Ideal Low-Pass Filter. Simple wrapper for fft_filter."""
    return fft_filter(image, 'ideal_lp', cutoff)

def ideal_hpf(image, cutoff=30):
    """Ideal High-Pass Filter."""
    return fft_filter(image, 'ideal_hp', cutoff)

def butterworth_lpf(image, cutoff=40, order=2):
    """Butterworth Low-Pass Filter — smooth rolloff, no ringing."""
    return fft_filter(image, 'butterworth_lp', cutoff, order)

def butterworth_hpf(image, cutoff=40, order=2):
    """Butterworth High-Pass Filter."""
    return fft_filter(image, 'butterworth_hp', cutoff, order)

def gaussian_lpf(image, cutoff=30):
    """Gaussian Low-Pass Filter — smoothest rolloff possible."""
    return fft_filter(image, 'gaussian_lp', cutoff)

def gaussian_hpf(image, cutoff=30):
    """Gaussian High-Pass Filter — enhances edges without ringing."""
    return fft_filter(image, 'gaussian_hp', cutoff)

def band_pass_filter(image, cutoff=40, order=2):
    """Band-Pass Filter — passes frequencies in a ring around cutoff."""
    return fft_filter(image, 'band_pass', cutoff, order)

def band_stop_filter(image, cutoff=40, order=2):
    """Band-Stop (Notch) Filter — blocks frequencies near cutoff."""
    return fft_filter(image, 'band_stop', cutoff, order)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HISTOGRAM OPERATIONS
# ═════════════════════════════════════════════════════════════════════════════

def compute_histogram(image):
    """
    Compute per-channel histograms for an RGB image.

    Returns a dict with:
        'R', 'G', 'B' — each a 256-element array (counts per intensity level)
        'gray'        — grayscale luminance histogram
    """
    hist = {}
    for i, ch in enumerate(['R', 'G', 'B']):
        hist[ch], _ = np.histogram(image[:, :, i].flatten(), bins=256, range=(0, 256))
    gray_ch = to_grayscale(image)
    hist['gray'], _ = np.histogram(gray_ch.flatten(), bins=256, range=(0, 256))
    return hist


def histogram_equalization(image):
    """
    Histogram Equalization — global contrast enhancement.

    Redistributes pixel intensities so the histogram is approximately UNIFORM
    (flat) across all 256 levels, maximising the use of the full dynamic range.

    Algorithm for each channel:
        1. Compute histogram h(rk) — count of pixels at each level rk
        2. Compute CDF: CDF(rk) = Σ h(rj) for j = 0..k
        3. Apply mapping: T(rk) = round( (CDF(rk) - CDF_min) / (N - CDF_min) × 255 )
           where N = total number of pixels

    Effect: Dark images become brighter; low-contrast images gain contrast.
           Particularly effective for underexposed photographs.
    DIVP Topic: Histogram equalization, intensity transformation, CDF mapping
    """
    result = np.zeros_like(image, dtype=np.float64)
    total  = image.shape[0] * image.shape[1]   # total pixels

    for c in range(3):
        channel = image[:, :, c].astype(np.uint8)
        hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
        cdf     = hist.cumsum()
        cdf_min = int(cdf[cdf > 0][0])         # first non-zero CDF value

        # Build lookup table T: for each level rk → equalized level
        lut = np.zeros(256, dtype=np.float64)
        for k in range(256):
            lut[k] = round((cdf[k] - cdf_min) / (total - cdf_min) * 255) \
                     if (total - cdf_min) > 0 else 0
        lut = np.clip(lut, 0, 255)

        result[:, :, c] = lut[channel]

    return result


def histogram_specification(image, target_shape='uniform'):
    """
    Histogram Specification (Matching) — map histogram to a target distribution.

    Unlike equalization (which targets uniform distribution), specification
    can target ANY desired histogram shape.

    Available target shapes:
        'uniform'   — flat histogram (same as equalization)
        'gaussian'  — bell-curve distribution (natural-looking)
        'rayleigh'  — right-skewed (used in radar imagery)
        'exponential'— exponential decay

    Algorithm:
        1. Compute source CDF
        2. Compute target CDF
        3. For each source level, find the target level with nearest CDF value
           (the mapping T is a lookup table)

    DIVP Topic: Histogram specification/matching, intensity transformation
    """
    result = np.zeros_like(image, dtype=np.float64)
    total  = image.shape[0] * image.shape[1]
    levels = np.arange(256)

    # Build target histogram
    if target_shape == 'gaussian':
        target_hist = np.exp(-((levels - 128) ** 2) / (2 * 50 ** 2))
    elif target_shape == 'rayleigh':
        sigma = 60.0
        target_hist = (levels / sigma ** 2) * np.exp(-(levels ** 2) / (2 * sigma ** 2))
    elif target_shape == 'exponential':
        target_hist = np.exp(-levels / 60.0)
    else:  # uniform
        target_hist = np.ones(256)

    target_hist = target_hist / target_hist.sum()       # normalise
    target_cdf  = np.cumsum(target_hist)                # target CDF

    for c in range(3):
        channel  = image[:, :, c].astype(np.uint8)
        src_hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
        src_cdf  = src_hist.cumsum() / total

        # Build mapping: for each source level, find nearest target CDF value
        lut = np.zeros(256, dtype=np.uint8)
        for k in range(256):
            diff    = np.abs(target_cdf - src_cdf[k])
            lut[k]  = np.argmin(diff)

        result[:, :, c] = lut[channel].astype(np.float64)

    return result


def contrast_stretch(image, low_pct=1.0, high_pct=99.0):
    """
    Contrast Stretching — simple linear intensity rescaling.

    Finds the low_pct and high_pct percentile values in the image.
    All pixels below the low percentile → 0
    All pixels above the high percentile → 255
    Everything in between → linearly mapped to [0, 255]

    Effect: Expands the effective dynamic range of the image.
    Simpler than histogram equalization but often produces more natural results.
    DIVP Topic: Contrast stretching, linear point transform, intensity scaling
    """
    result = np.zeros_like(image, dtype=np.float64)
    for c in range(3):
        channel = image[:, :, c].astype(np.float64)
        lo = np.percentile(channel, low_pct)
        hi = np.percentile(channel, high_pct)
        if hi == lo:
            result[:, :, c] = channel
        else:
            result[:, :, c] = np.clip((channel - lo) / (hi - lo) * 255.0, 0, 255)
    return result


def gamma_correction(image, gamma=1.5):
    """
    Gamma Correction — Power Law (Gamma) Transform.

    Formula:  s = c · r^(1/γ)   where c = 1 (normalised), r ∈ [0, 1]

    In code: output = 255 × (input/255)^(1/γ)

    γ < 1: Brightens dark images (expands lower intensities)
    γ > 1: Darkens bright images (compresses higher intensities)
    γ = 1: No change

    This is also used in display calibration and image compression.
    Effect: Non-linear brightness adjustment.
    DIVP Topic: Power law transform, gamma correction, point transform
    """
    gamma    = max(0.01, float(gamma))
    norm     = image.astype(np.float64) / 255.0
    corrected = np.power(np.clip(norm, 0, 1), 1.0 / gamma)
    return corrected * 255.0


def log_transform(image, c=1.0):
    """
    Logarithmic Transform — s = c × log(1 + r)

    Expands dark pixel values and compresses bright ones.
    Used to display Fourier spectrum (which has huge dynamic range).

    DIVP Topic: Log transform, point transform, intensity mapping
    """
    c = float(c)
    return c * np.log1p(image.astype(np.float64))


def negative_image(image):
    """
    Image Negative — s = (L-1) - r    where L = 256

    Inverts all pixel values. Equivalent to film negative in photography.
    Useful for enhancing white detail in dark regions.
    DIVP Topic: Image negative, complement, point transform
    """
    return 255.0 - image.astype(np.float64)


def threshold_image(image, thresh=128, mode='binary'):
    """
    Image Thresholding — converts grayscale to binary image.

    Modes:
        'binary'    — pixels > thresh → 255, rest → 0
        'otsu'      — automatically finds optimal threshold (Otsu's method)
        'adaptive'  — per-region threshold (handles uneven lighting)

    Otsu's method: finds threshold that minimises within-class variance
        σ²_w(t) = w0(t)·σ0²(t) + w1(t)·σ1²(t)

    DIVP Topic: Thresholding, segmentation, Otsu's method
    """
    gray = to_grayscale(image).astype(np.uint8)

    if mode == 'otsu':
        # Otsu's optimal threshold computation
        hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
        total    = gray.size
        sum_all  = np.sum(np.arange(256) * hist)
        sum_b = w0 = w1 = 0
        max_var = best_t = 0
        for t in range(256):
            w0 += hist[t]
            if w0 == 0: continue
            w1 = total - w0
            if w1 == 0: break
            sum_b += t * hist[t]
            m0 = sum_b / w0
            m1 = (sum_all - sum_b) / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > max_var:
                max_var = var
                best_t  = t
        thresh = best_t

    binary = np.where(gray > thresh, 255.0, 0.0)
    return gray_to_rgb(binary)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NOISE MODELS
# ═════════════════════════════════════════════════════════════════════════════

def add_gaussian_noise(image, sigma=20.0):
    """
    Gaussian (Additive) Noise — models sensor/thermal noise.

    Adds random noise drawn from a Normal distribution N(0, σ²).
    Each pixel becomes: p' = p + N(0, σ²)

    sigma = 0   → no noise
    sigma = 20  → moderate noise (visible but manageable)
    sigma = 60+ → severe noise

    This is the most common noise model in image processing.
    Best removed by: Gaussian filter, Wiener filter
    DIVP Topic: Gaussian noise model, additive noise, degradation model
    """
    sigma  = float(sigma)
    noise  = np.random.normal(0, sigma, image.shape)
    return image.astype(np.float64) + noise


def add_salt_pepper_noise(image, density=0.05, salt_ratio=0.5):
    """
    Salt & Pepper (Impulse) Noise — models transmission errors and dead pixels.

    Randomly corrupts pixels:
        Salt (white): pixel → 255
        Pepper (black): pixel → 0

    density    — fraction of total pixels corrupted (0.05 = 5%)
    salt_ratio — fraction of corrupted pixels that are salt vs pepper

    Best removed by: Median filter (mean filter makes it worse!)
    DIVP Topic: Impulse noise, salt and pepper, noise model
    """
    result = image.copy().astype(np.float64)
    total  = image.shape[0] * image.shape[1]
    n_corrupt = int(total * float(density))

    # Salt pixels (white)
    n_salt = int(n_corrupt * salt_ratio)
    salt_y = np.random.randint(0, image.shape[0], n_salt)
    salt_x = np.random.randint(0, image.shape[1], n_salt)
    result[salt_y, salt_x] = 255.0

    # Pepper pixels (black)
    n_pepper = n_corrupt - n_salt
    pep_y = np.random.randint(0, image.shape[0], n_pepper)
    pep_x = np.random.randint(0, image.shape[1], n_pepper)
    result[pep_y, pep_x] = 0.0

    return result


def add_poisson_noise(image, scale=1.0):
    """
    Poisson (Shot) Noise — models photon counting noise in low-light photography.

    Unlike Gaussian noise (additive, independent of signal),
    Poisson noise is SIGNAL-DEPENDENT — brighter pixels have more noise.

    p' = Poisson(p × scale) / scale

    Used in: astronomy, medical imaging (X-ray, PET scans), low-light cameras
    DIVP Topic: Poisson noise model, signal-dependent noise
    """
    scale  = max(0.1, float(scale))
    scaled = (image.astype(np.float64) / 255.0) * scale
    noisy  = np.random.poisson(np.clip(scaled, 0, None)).astype(np.float64)
    return np.clip(noisy / scale * 255.0, 0, 255)


def add_speckle_noise(image, variance=0.05):
    """
    Speckle (Multiplicative) Noise — models coherent imaging noise.

    Formula:  p' = p + p × N(0, var)   (signal-dependent, multiplicative)

    Used in: ultrasound images, synthetic aperture radar (SAR), laser images.
    More difficult to remove than additive noise because it scales with signal.

    Best removed by: log transformation + Wiener filter, or non-local means
    DIVP Topic: Speckle noise, multiplicative noise model
    """
    variance = float(variance)
    noise    = np.random.normal(0, np.sqrt(variance), image.shape)
    return image.astype(np.float64) * (1 + noise)


def add_uniform_noise(image, low=-30, high=30):
    """
    Uniform Noise — noise drawn from a uniform distribution [low, high].

    Less common than Gaussian but useful for modelling quantisation noise.
    DIVP Topic: Uniform noise model, quantisation noise
    """
    noise = np.random.uniform(float(low), float(high), image.shape)
    return image.astype(np.float64) + noise


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — IMAGE RESTORATION
# ═════════════════════════════════════════════════════════════════════════════

def wiener_filter(image, noise_ratio=0.01):
    """
    Wiener Filter — optimal linear restoration filter (MSE-minimising).

    The Wiener filter minimises the Mean Squared Error between the restored
    image and the true (noise-free) image. It is the best possible linear filter
    when noise statistics are known.

    In frequency domain:
        Ĝ(u,v) = H*(u,v) / (|H(u,v)|² + K) × F_degraded(u,v)

    where:
        H(u,v)  = degradation function (here we assume identity: H = 1)
        K       = noise_ratio = σ²_noise / σ²_signal  (SNR inverse)
        H*      = complex conjugate of H

    When K is small → less smoothing (better for low-noise images)
    When K is large → more smoothing (better for high-noise images)

    DIVP Topic: Wiener filter, constrained restoration, frequency domain
    """
    K      = float(noise_ratio)
    result = np.zeros_like(image, dtype=np.float64)

    for c in range(3):
        channel = image[:, :, c].astype(np.float64)
        F       = np.fft.fft2(channel)
        H       = np.ones_like(F)                      # identity degradation
        H_conj  = np.conj(H)
        H_sq    = np.abs(H) ** 2

        # Wiener formula in frequency domain
        W          = H_conj / (H_sq + K)
        restored_F = W * F
        restored   = np.real(np.fft.ifft2(restored_F))
        result[:, :, c] = np.clip(restored, 0, 255)

    return result


def adaptive_median_filter(image, max_ksize=7):
    """
    Adaptive Median Filter — handles variable-density salt & pepper noise.

    Unlike the standard median filter (fixed window size), this filter
    adaptively increases the window size until a valid (non-noise) median
    is found. This allows it to handle denser noise while preserving edges better.

    Algorithm for each pixel (i, j):
        Stage A: If median is not noise:
            If center pixel is not noise → keep it unchanged
            Else → replace with median
        Stage B: If median IS noise → increase window size and repeat

    DIVP Topic: Adaptive filtering, variable-window median, noise removal
    """
    H, W   = image.shape[:2]
    result = image.copy().astype(np.float64)

    for i in range(H):
        for j in range(W):
            ksize = 3
            while ksize <= max_ksize:
                half  = ksize // 2
                r0 = max(0, i - half); r1 = min(H, i + half + 1)
                c0 = max(0, j - half); c1 = min(W, j + half + 1)

                for ch in range(3):
                    region = image[r0:r1, c0:c1, ch].flatten()
                    z_min, z_max = region.min(), region.max()
                    z_med = np.median(region)
                    z_xy  = image[i, j, ch]

                    A1 = z_med - z_min
                    A2 = z_med - z_max

                    if A1 > 0 and A2 < 0:
                        # Stage A: median is not a noise point
                        B1 = z_xy - z_min
                        B2 = z_xy - z_max
                        if B1 > 0 and B2 < 0:
                            result[i, j, ch] = z_xy       # preserve original
                        else:
                            result[i, j, ch] = z_med      # replace with median
                        break
                    else:
                        ksize += 2                         # increase window
                        if ksize > max_ksize:
                            result[i, j, ch] = z_med

    return result


def constrained_ls_filter(image, gamma=0.1):
    """
    Constrained Least Squares (CLS) Restoration Filter.

    Minimises: ||Pĝ||²  subject to  ||f - Hĝ|| = ||n||²

    where P is the Laplacian operator (second derivative — smoothness constraint).

    In frequency domain:
        Ĝ(u,v) = H*(u,v) / (|H(u,v)|² + γ|P(u,v)|²) × F(u,v)

    Larger γ → smoother result (more regularisation)
    Smaller γ → sharper result (less regularisation)

    DIVP Topic: Constrained restoration, regularisation, Laplacian constraint
    """
    gamma  = float(gamma)
    result = np.zeros_like(image, dtype=np.float64)

    # Laplacian operator in spatial domain (pad to image size in freq domain)
    rows, cols = image.shape[:2]
    p          = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    p_padded   = np.zeros((rows, cols))
    p_padded[:3, :3] = p
    P_freq     = np.fft.fft2(p_padded)

    for c in range(3):
        channel = image[:, :, c].astype(np.float64)
        F       = np.fft.fft2(channel)
        H       = np.ones((rows, cols), dtype=complex)       # identity degradation
        H_conj  = np.conj(H)
        H_sq    = np.abs(H) ** 2
        P_sq    = np.abs(P_freq) ** 2

        G          = H_conj / (H_sq + gamma * P_sq) * F
        restored   = np.real(np.fft.ifft2(G))
        result[:, :, c] = np.clip(restored, 0, 255)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MORPHOLOGICAL OPERATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _morphology_op(image, op, ksize=3, iterations=1):
    """
    Internal helper: apply erosion or dilation using sliding window.
    Uses grayscale morphology (per-channel min or max in the window).
    """
    ksize  = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    half   = ksize // 2
    H, W   = image.shape[:2]
    result = image.copy().astype(np.float64)

    for _ in range(int(iterations)):
        padded = pad_image(result, half, half, mode='reflect')
        out    = np.zeros_like(result)
        for i in range(H):
            for j in range(W):
                region = padded[i:i + ksize, j:j + ksize]
                if image.ndim == 3:
                    for c in range(3):
                        out[i, j, c] = region[:, :, c].min() if op == 'erode' else region[:, :, c].max()
                else:
                    out[i, j] = region.min() if op == 'erode' else region.max()
        result = out

    return result


def erode(image, ksize=3, iterations=1):
    """
    Erosion (A ⊖ B) — shrinks bright regions, removes small bright objects.

    For each pixel, the minimum value in the kernel window replaces the center.
    With a binary image: a pixel is 1 only if ALL kernel positions are 1.

    Effect: Shrinks white regions; removes thin lines and small noise blobs.
    DIVP Topic: Erosion, morphological processing, structuring element
    """
    return _morphology_op(image, 'erode', ksize, iterations)


def dilate(image, ksize=3, iterations=1):
    """
    Dilation (A ⊕ B) — expands bright regions, fills small holes.

    For each pixel, the maximum value in the kernel window replaces the center.
    With binary image: a pixel is 1 if ANY kernel position is 1.

    Effect: Expands white regions; connects nearby objects; fills small holes.
    DIVP Topic: Dilation, morphological processing, structuring element
    """
    return _morphology_op(image, 'dilate', ksize, iterations)


def opening(image, ksize=3):
    """
    Opening (A ∘ B) = Erosion then Dilation.

    Removes small bright objects (noise, thin lines) while preserving the
    shape and size of larger objects. Smooths contours from outside.

    Effect: Cleans up small isolated bright noise.
    DIVP Topic: Opening, morphological filtering, noise removal
    """
    eroded = _morphology_op(image, 'erode',  ksize, 1)
    return   _morphology_op(eroded, 'dilate', ksize, 1)


def closing(image, ksize=3):
    """
    Closing (A • B) = Dilation then Erosion.

    Fills small dark holes/gaps within bright regions.
    Opposite of opening — smooths contours from inside.

    Effect: Fills small holes and narrow dark gaps.
    DIVP Topic: Closing, morphological filtering
    """
    dilated = _morphology_op(image, 'dilate', ksize, 1)
    return    _morphology_op(dilated, 'erode', ksize, 1)


def morphological_gradient(image, ksize=3):
    """
    Morphological Gradient = Dilation − Erosion.

    The difference between dilation and erosion highlights edges/boundaries.
    Similar to the Sobel gradient but purely morphological.

    Effect: Extracts the outline/boundary of objects.
    DIVP Topic: Morphological edge detection, gradient
    """
    d = _morphology_op(image, 'dilate', ksize, 1).astype(np.float64)
    e = _morphology_op(image, 'erode',  ksize, 1).astype(np.float64)
    return d - e


def top_hat_transform(image, ksize=5):
    """
    Top-Hat Transform = Original − Opening.

    Highlights bright features that are smaller than the structuring element.
    Very useful for finding bright objects on uneven backgrounds.

    Effect: Extracts small bright details relative to background.
    DIVP Topic: Top-hat transform, morphological analysis
    """
    opened = opening(image, ksize).astype(np.float64)
    return np.clip(image.astype(np.float64) - opened, 0, 255)


def black_hat_transform(image, ksize=5):
    """
    Black-Hat Transform = Closing − Original.

    Highlights dark features smaller than the structuring element.
    Complement of top-hat.

    Effect: Extracts small dark details relative to background.
    DIVP Topic: Black-hat transform, morphological analysis
    """
    closed = closing(image, ksize).astype(np.float64)
    return np.clip(closed - image.astype(np.float64), 0, 255)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — COLOR IMAGE PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def rgb_to_hsi(image):
    """
    Convert RGB image to HSI color model.

    HSI (Hue, Saturation, Intensity) separates color info from intensity,
    which is useful for color-based processing without affecting brightness.

    H = arccos { [(R-G) + (R-B)] / [2√((R-G)²+(R-B)(G-B))] }
    S = 1 - [3/(R+G+B)] × min(R,G,B)
    I = (R+G+B)/3

    Returns:
        HSI image scaled so H∈[0,255], S∈[0,255], I∈[0,255] for display.
    DIVP Topic: Color models, RGB to HSI conversion
    """
    norm = image.astype(np.float64) / 255.0
    R, G, B = norm[:, :, 0], norm[:, :, 1], norm[:, :, 2]

    I = (R + G + B) / 3.0

    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 0, 1.0 - min_rgb / (I + 1e-10), 0.0)

    num  = 0.5 * ((R - G) + (R - B))
    den  = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1, 1))

    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)    # normalise to [0, 1]

    result = np.stack([H * 255, S * 255, I * 255], axis=2)
    return np.clip(result, 0, 255)


def rgb_to_hsv(image):
    """
    Convert RGB to HSV (Hue, Saturation, Value) color model.

    HSV is more intuitive for human perception than RGB.
    H → color type (0–360°), S → color purity, V → brightness

    Returns:
        HSV image scaled for display (H∈[0,255], S∈[0,255], V∈[0,255])
    DIVP Topic: Color models, HSV/HSB color space
    """
    norm = image.astype(np.float64) / 255.0
    R, G, B = norm[:, :, 0], norm[:, :, 1], norm[:, :, 2]

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    D    = Cmax - Cmin

    V = Cmax.copy()
    S = np.where(Cmax != 0, D / Cmax, 0.0)

    H = np.zeros_like(R)
    mask_r = (Cmax == R) & (D != 0)
    mask_g = (Cmax == G) & (D != 0)
    mask_b = (Cmax == B) & (D != 0)
    H[mask_r] = (60 * ((G - B) / D))[mask_r] % 360
    H[mask_g] = (60 * ((B - R) / D + 2))[mask_g]
    H[mask_b] = (60 * ((R - G) / D + 4))[mask_b]

    result = np.stack([H / 360 * 255, S * 255, V * 255], axis=2)
    return np.clip(result, 0, 255)


def pseudocolor(image, colormap='jet'):
    """
    Pseudo-Color Processing — map grayscale intensities to colors.

    Humans can distinguish far more colors than gray shades.
    Pseudo-coloring maps intensity levels to a color spectrum,
    making subtle intensity differences visible to the eye.

    Available colormaps: 'jet', 'hot', 'cool', 'rainbow'

    DIVP Topic: Pseudo-color processing, intensity to color mapping
    """
    gray = to_grayscale(image)
    norm = gray / 255.0
    result = np.zeros((*gray.shape, 3), dtype=np.float64)

    if colormap == 'jet':
        result[:, :, 0] = np.clip(1.5 - np.abs(4 * norm - 3), 0, 1) * 255
        result[:, :, 1] = np.clip(1.5 - np.abs(4 * norm - 2), 0, 1) * 255
        result[:, :, 2] = np.clip(1.5 - np.abs(4 * norm - 1), 0, 1) * 255
    elif colormap == 'hot':
        result[:, :, 0] = np.clip(norm * 3, 0, 1) * 255
        result[:, :, 1] = np.clip(norm * 3 - 1, 0, 1) * 255
        result[:, :, 2] = np.clip(norm * 3 - 2, 0, 1) * 255
    elif colormap == 'cool':
        result[:, :, 0] = norm * 255
        result[:, :, 1] = (1 - norm) * 255
        result[:, :, 2] = 255
    else:  # rainbow
        result[:, :, 0] = np.abs(np.sin(norm * np.pi)) * 255
        result[:, :, 1] = np.abs(np.sin(norm * np.pi + np.pi / 3)) * 255
        result[:, :, 2] = np.abs(np.cos(norm * np.pi)) * 255

    return result


def color_balance(image, r_scale=1.0, g_scale=1.0, b_scale=1.0):
    """
    Per-channel color balance scaling.
    Multiplies each channel by its respective scale factor.
    Values > 1 boost that channel; < 1 reduces it.

    DIVP Topic: Color image processing, channel manipulation
    """
    result = image.astype(np.float64).copy()
    result[:, :, 0] *= float(r_scale)
    result[:, :, 1] *= float(g_scale)
    result[:, :, 2] *= float(b_scale)
    return np.clip(result, 0, 255)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — IMAGE TRANSFORMS
# ═════════════════════════════════════════════════════════════════════════════

def dct_compress(image, quality=50):
    """
    DCT-Based Image Compression (JPEG-style).

    This is the core of the JPEG compression standard.
    Demonstrates exactly how image compression sacrifices quality for file size.

    Algorithm (per 8×8 block):
        1. Subtract 128 (center the values around zero)
        2. Apply 2D DCT to each 8×8 block
        3. Quantise DCT coefficients using the quality matrix
           (low quality = round to coarser values = more detail lost)
        4. Dequantise (multiply back)
        5. Apply inverse DCT (IDCT) to reconstruct

    The DCT formula:
        F(u,v) = (2/N) · C(u)·C(v) · ΣΣ f(x,y)·cos[(2x+1)uπ/2N]·cos[(2y+1)vπ/2N]

    quality 1   → maximum compression, severe blocking artifacts
    quality 50  → standard JPEG quality
    quality 95+ → near-lossless

    DIVP Topic: DCT, image compression, quantization, JPEG
    """
    from scipy.fft import dct, idct

    quality   = max(1, min(100, int(quality)))
    # Standard JPEG luminance quantisation matrix (baseline)
    Q_base = np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68, 109, 103,  77],
        [24, 35, 55, 64, 81, 104, 113,  92],
        [49, 64, 78, 87, 103,121, 120, 101],
        [72, 92, 95, 98, 112,100, 103,  99],
    ], dtype=np.float64)

    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    Q = np.clip(np.round(Q_base * scale / 100.0), 1, 255)

    H, W    = image.shape[:2]
    result  = np.zeros_like(image, dtype=np.float64)

    for c in range(3):
        channel = image[:, :, c].astype(np.float64) - 128.0
        out     = np.zeros_like(channel)

        for i in range(0, H - 7, 8):
            for j in range(0, W - 7, 8):
                block    = channel[i:i+8, j:j+8]
                dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                q_block  = np.round(dct_block / Q) * Q   # quantise + dequantise
                idct_blk = idct(idct(q_block, axis=0, norm='ortho'), axis=1, norm='ortho')
                out[i:i+8, j:j+8] = idct_blk

        result[:, :, c] = np.clip(out + 128.0, 0, 255)

    return result


def haar_wavelet(image, levels=2):
    """
    Haar Wavelet Transform — multiresolution image decomposition.

    The Haar wavelet is the simplest wavelet. It decomposes the image into:
        LL — low-low  (approximation, looks like blurred original)
        LH — low-high (horizontal edges)
        HL — high-low (vertical edges)
        HH — high-high (diagonal edges)

    Each level splits the LL subband into 4 more subbands.
    This multiresolution representation is used in JPEG-2000 compression.

    Algorithm (1 level, 1D on rows then columns):
        Forward: avg = (a + b)/2,  diff = (a - b)/2
        Inverse: a   = avg + diff, b   = avg - diff

    DIVP Topic: Wavelet transform, Haar wavelet, multiresolution analysis
    """
    def haar_1d_forward(signal):
        """1D Haar transform on rows or columns."""
        n = signal.shape[-1]
        if n < 2:
            return signal
        even = signal[..., 0::2]
        odd  = signal[..., 1::2]
        avg  = (even + odd) / 2.0
        diff = (even - odd) / 2.0
        return np.concatenate([avg, diff], axis=-1)

    gray = to_grayscale(image)
    H, W = gray.shape
    # Make square power of 2 for clean decomposition
    size = min(H, W)
    size = 2 ** int(np.log2(size))
    block = gray[:size, :size].copy()

    for _ in range(int(levels)):
        h, w = block.shape
        # Transform rows
        block[:h, :w] = haar_1d_forward(block[:h, :w])
        # Transform columns (transpose trick)
        block[:h, :w] = haar_1d_forward(block[:h, :w].T).T

    # Normalise for display
    display = normalize_to_255(block)
    full = gray.copy()
    full[:size, :size] = display
    return gray_to_rgb(full)


def walsh_hadamard_transform(image):
    """
    Walsh-Hadamard Transform (WHT) — non-sinusoidal orthogonal transform.

    Unlike FFT (uses sinusoids), WHT uses rectangular basis functions (±1 only).
    This makes it computationally cheaper — only additions and subtractions.

    The Hadamard matrix H of order N (N must be power of 2):
        H_1 = [1]
        H_N = (1/√N) × [H_{N/2}  H_{N/2}]
                        [H_{N/2} -H_{N/2}]

    Used in: image compression, spread spectrum communications, cryptography.
    DIVP Topic: Walsh-Hadamard transform, orthogonal transforms
    """
    def hadamard_matrix(n):
        """Build normalised Hadamard matrix of order n (n must be power of 2)."""
        if n == 1:
            return np.array([[1.0]])
        H_half = hadamard_matrix(n // 2)
        top    = np.hstack([H_half,  H_half])
        bot    = np.hstack([H_half, -H_half])
        return np.vstack([top, bot]) / np.sqrt(2)

    gray = to_grayscale(image)
    H, W = gray.shape
    # Use a 64×64 block for speed (WHT is O(N²) for matrix multiply)
    size = 64
    block = gray[:size, :size].astype(np.float64)
    Had   = hadamard_matrix(size)

    wht_block  = Had @ block @ Had.T     # 2D WHT = H·f·H^T
    display    = normalize_to_255(np.abs(wht_block))

    full = np.zeros_like(gray)
    full[:size, :size] = display
    return gray_to_rgb(full)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN DISPATCH FUNCTION (Called by Flask server in app.py)
# ═════════════════════════════════════════════════════════════════════════════

# Maps filter_name strings to their functions and categories
FILTER_REGISTRY = {
    # ── Spatial Domain ──────────────────────────────────────────────────────
    'mean':               ('Spatial',         mean_filter),
    'gaussian':           ('Spatial',         gaussian_filter),
    'median':             ('Spatial',         median_filter),
    'bilateral':          ('Spatial',         bilateral_filter),
    'sharpen':            ('Spatial',         sharpen_filter),
    'laplacian':          ('Spatial',         laplacian_filter),
    'high_boost':         ('Spatial',         high_boost_filter),
    'negative':           ('Point Transform', negative_image),
    'log_transform':      ('Point Transform', log_transform),

    # ── Edge Detection ───────────────────────────────────────────────────────
    'roberts':            ('Edge Detection',  roberts_filter),
    'prewitt':            ('Edge Detection',  prewitt_filter),
    'sobel':              ('Edge Detection',  sobel_filter),
    'log':                ('Edge Detection',  log_filter),
    'canny':              ('Edge Detection',  canny_edge),

    # ── Frequency Domain ─────────────────────────────────────────────────────
    'lpf_ideal':          ('Frequency',       ideal_lpf),
    'hpf_ideal':          ('Frequency',       ideal_hpf),
    'lpf_butterworth':    ('Frequency',       butterworth_lpf),
    'hpf_butterworth':    ('Frequency',       butterworth_hpf),
    'lpf_gaussian':       ('Frequency',       gaussian_lpf),
    'hpf_gaussian':       ('Frequency',       gaussian_hpf),
    'band_pass':          ('Frequency',       band_pass_filter),
    'band_stop':          ('Frequency',       band_stop_filter),
    'fft_spectrum':       ('Frequency',       fft_spectrum),

    # ── Histogram ────────────────────────────────────────────────────────────
    'hist_eq':            ('Histogram',       histogram_equalization),
    'hist_spec':          ('Histogram',       histogram_specification),
    'contrast_stretch':   ('Histogram',       contrast_stretch),
    'gamma':              ('Point Transform', gamma_correction),
    'threshold':          ('Segmentation',    threshold_image),

    # ── Noise ────────────────────────────────────────────────────────────────
    'noise_gaussian':     ('Noise',           add_gaussian_noise),
    'noise_sp':           ('Noise',           add_salt_pepper_noise),
    'noise_poisson':      ('Noise',           add_poisson_noise),
    'noise_speckle':      ('Noise',           add_speckle_noise),
    'noise_uniform':      ('Noise',           add_uniform_noise),

    # ── Restoration ──────────────────────────────────────────────────────────
    'wiener':             ('Restoration',     wiener_filter),
    'adaptive_median':    ('Restoration',     adaptive_median_filter),
    'cls_filter':         ('Restoration',     constrained_ls_filter),

    # ── Morphology ───────────────────────────────────────────────────────────
    'erode':              ('Morphology',      erode),
    'dilate':             ('Morphology',      dilate),
    'opening':            ('Morphology',      opening),
    'closing':            ('Morphology',      closing),
    'morph_gradient':     ('Morphology',      morphological_gradient),
    'top_hat':            ('Morphology',      top_hat_transform),
    'black_hat':          ('Morphology',      black_hat_transform),

    # ── Color Processing ─────────────────────────────────────────────────────
    'rgb_to_hsi':         ('Color',           rgb_to_hsi),
    'rgb_to_hsv':         ('Color',           rgb_to_hsv),
    'pseudocolor':        ('Color',           pseudocolor),
    'color_balance':      ('Color',           color_balance),

    # ── Transforms ───────────────────────────────────────────────────────────
    'dct_compress':       ('Transform',       dct_compress),
    'haar_wavelet':       ('Transform',       haar_wavelet),
    'walsh_hadamard':     ('Transform',       walsh_hadamard_transform),
}


def apply_filter(image_array, filter_name, params):
    """
    Main dispatch function — routes filter requests to the correct function.
    Called by app.py every time the user clicks 'Apply Filter'.

    Args:
        image_array (np.ndarray) — float64 RGB image, shape (H, W, 3)
        filter_name (str)        — must be a key in FILTER_REGISTRY
        params      (dict)       — parameter dict from frontend sliders

    Returns:
        dict with keys:
            'result'    — processed image as float64 array (H, W, 3)
            'category'  — DIVP category string
            'metrics'   — dict of psnr, mse, mae, std_dev
            'time_ms'   — processing time in milliseconds
    """
    if filter_name not in FILTER_REGISTRY:
        raise ValueError(
            f"Unknown filter: '{filter_name}'. "
            f"Available filters: {list(FILTER_REGISTRY.keys())}"
        )

    category, fn = FILTER_REGISTRY[filter_name]
    img = np.clip(image_array.astype(np.float64), 0, 255)

    # ── Time the processing ───────────────────────────────────────────────────
    t_start = time.perf_counter()

    # ── Smart parameter passing ───────────────────────────────────────────────
    # Each function only receives the parameters it needs
    import inspect
    sig        = inspect.signature(fn)
    fn_params  = list(sig.parameters.keys())
    call_kwargs = {}

    # Helper: get param, coerce type, use default if missing
    def gp(key, default, cast=float):
        val = params.get(key, default)
        try:
            return cast(val)
        except (TypeError, ValueError):
            return default

    # Map param names to values
    param_map = {
        'ksize':        lambda: max(1, int(gp('ksize',  3, int))),
        'sigma':        lambda: gp('sigma',       1.5),
        'sigma_space':  lambda: gp('sigma_space', 1.5),
        'sigma_color':  lambda: gp('sigma_color', 50.0),
        'amount':       lambda: gp('amount',      1.5),
        'blur_sigma':   lambda: gp('blur_sigma',  1.0),
        'boost_factor': lambda: gp('boost_factor',2.0),
        'kernel_type':  lambda: str(params.get('type',   '4-connected')),
        'scale':        lambda: gp('scale',       1.0),
        'direction':    lambda: str(params.get('dir',    'Combined')),
        'low_thresh':   lambda: gp('lowT',        20.0),
        'high_thresh':  lambda: gp('highT',       60.0),
        'filter_type':  lambda: str(params.get('filter_type', 'gaussian_lp')),
        'cutoff':       lambda: gp('cutoff',      30.0),
        'order':        lambda: max(1, int(gp('order', 2, int))),
        'target_shape': lambda: str(params.get('target_shape', 'uniform')),
        'low_pct':      lambda: gp('low',         1.0),
        'high_pct':     lambda: gp('high',        99.0),
        'gamma':        lambda: gp('gamma',       1.5),
        'c':            lambda: gp('c',           1.0),
        'thresh':       lambda: gp('thresh',      128.0),
        'mode':         lambda: str(params.get('mode', 'binary')),
        'density':      lambda: gp('density',     0.05),
        'salt_ratio':   lambda: gp('salt_ratio',  0.5),
        'variance':     lambda: gp('variance',    0.05),
        'low':          lambda: gp('low',         -30.0),
        'high':         lambda: gp('high',        30.0),
        'noise_ratio':  lambda: gp('noise_ratio', 0.01),
        'max_ksize':    lambda: max(3, int(gp('max_ksize', 7, int))),
        'gamma_cls':    lambda: gp('gamma',       0.1),
        'iterations':   lambda: max(1, int(gp('iters', 1, int))),
        'colormap':     lambda: str(params.get('colormap', 'jet')),
        'r_scale':      lambda: gp('r_scale',     1.0),
        'g_scale':      lambda: gp('g_scale',     1.0),
        'b_scale':      lambda: gp('b_scale',     1.0),
        'quality':      lambda: max(1, min(100, int(gp('quality', 50, int)))),
        'levels':       lambda: max(1, int(gp('levels', 2, int))),
        'max_level':    lambda: max(1, int(gp('max_level', 2, int))),
    }

    for pname in fn_params[1:]:    # skip 'image' (first param)
        if pname in param_map:
            call_kwargs[pname] = param_map[pname]()

    # ── Call the filter ────────────────────────────────────────────────────────
    result = fn(img, **call_kwargs)

    t_end    = time.perf_counter()
    time_ms  = round((t_end - t_start) * 1000, 2)

    # ── Compute quality metrics ────────────────────────────────────────────────
    result   = np.clip(result, 0, 255)
    metrics  = compute_metrics(img, result)

    return {
        'result':   result,
        'category': category,
        'metrics':  metrics,
        'time_ms':  time_ms
    }


def list_filters():
    """
    Return a list of all available filters grouped by category.
    Useful for building the frontend filter menu dynamically.
    """
    grouped = {}
    for name, (cat, _) in FILTER_REGISTRY.items():
        grouped.setdefault(cat, []).append(name)
    return grouped


# ═════════════════════════════════════════════════════════════════════════════
# QUICK TEST  (Run this file directly: python filters.py)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PIXLENS — filters.py  Quick Test")
    print("=" * 60)

    # Create a synthetic 64×64 test image (gradient with noise)
    H, W = 64, 64
    test_img = np.zeros((H, W, 3), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            test_img[i, j, 0] = i / H * 255
            test_img[i, j, 1] = j / W * 255
            test_img[i, j, 2] = 128
    test_img += np.random.normal(0, 10, test_img.shape)
    test_img = np.clip(test_img, 0, 255)

    print(f"\n[✓] Test image created: {test_img.shape}, dtype={test_img.dtype}")
    print(f"    Pixel range: {test_img.min():.1f} – {test_img.max():.1f}")

    # Test a selection of filters from each category
    test_cases = [
        ('mean',           {'ksize': 3}),
        ('gaussian',       {'ksize': 5, 'sigma': 1.5}),
        ('median',         {'ksize': 3}),
        ('sharpen',        {'amount': 1.5}),
        ('laplacian',      {'type': '8-connected'}),
        ('sobel',          {'dir': 'Combined'}),
        ('canny',          {'sigma': 1.0, 'lowT': 20, 'highT': 60}),
        ('lpf_butterworth',{'cutoff': 20, 'order': 2}),
        ('hpf_gaussian',   {'cutoff': 15}),
        ('hist_eq',        {}),
        ('gamma',          {'gamma': 1.8}),
        ('noise_gaussian', {'sigma': 20}),
        ('noise_sp',       {'density': 0.05}),
        ('wiener',         {'noise_ratio': 0.01}),
        ('erode',          {'ksize': 3, 'iters': 1}),
        ('dilate',         {'ksize': 3, 'iters': 1}),
        ('opening',        {'ksize': 3}),
        ('closing',        {'ksize': 3}),
        ('top_hat',        {'ksize': 5}),
        ('rgb_to_hsi',     {}),
        ('pseudocolor',    {'colormap': 'jet'}),
        ('dct_compress',   {'quality': 50}),
        ('haar_wavelet',   {'levels': 2}),
    ]

    passed = 0
    failed = 0

    for filter_name, params in test_cases:
        try:
            out = apply_filter(test_img, filter_name, params)
            r   = out['result']
            m   = out['metrics']
            assert r.shape == test_img.shape, f"Shape mismatch: {r.shape}"
            assert r.dtype == np.float64
            assert 0 <= r.min() and r.max() <= 255
            print(f"  [✓] {filter_name:<22} → PSNR={m['psnr']:>7.2f} dB  "
                  f"MSE={m['mse']:>8.2f}  time={out['time_ms']:>6.1f}ms")
            passed += 1
        except Exception as e:
            print(f"  [✗] {filter_name:<22} → FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")

    print("\n[✓] Available filters by category:")
    for cat, names in list_filters().items():
        print(f"    {cat:<20}: {', '.join(names)}")

    print("\n" + "=" * 60)
    print("  filters.py test complete.")
    print("=" * 60)