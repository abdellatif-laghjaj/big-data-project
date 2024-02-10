# Convert the frames to grayscale
import cv2
import numpy as np
from scipy.signal import wiener

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Égalisation de l'histogramme
def equalize_histogram(image):
    """Equalize the histogram of a grayscale image."""
    # Séparer les canaux de couleur
    canal_b, canal_g, canal_r = cv2.split(image)

    # Appliquer l'égalisation d'histogramme à chaque canal de couleur
    canal_b_eq = cv2.equalizeHist(canal_b)
    canal_g_eq = cv2.equalizeHist(canal_g)
    canal_r_eq = cv2.equalizeHist(canal_r)

    # Fusionner les canaux de couleur égalisés
    image_eq = cv2.merge((canal_b_eq, canal_g_eq, canal_r_eq))

    return image_eq

# Appliquer un filtre gaussien avec un noyau de taille 5x5
def apply_gaussian_blur(image, ksize=5):
    """Apply a Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

# Apply laplacian filter
def apply_laplacian(image):
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Apply Laplacian filter to each channel
    b_lap = cv2.Laplacian(b, cv2.CV_64F)
    g_lap = cv2.Laplacian(g, cv2.CV_64F)
    r_lap = cv2.Laplacian(r, cv2.CV_64F)

    # Convert back to uint8
    b_lap = np.uint8(np.absolute(b_lap))
    g_lap = np.uint8(np.absolute(g_lap))
    r_lap = np.uint8(np.absolute(r_lap))

    # Merge the sharpened channels
    sharpened = cv2.merge((b_lap, g_lap, r_lap))

    return sharpened

# Apply Sobel filter
def apply_sobel_filter(image):
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Apply Sobel filter to each channel
    b_sobel_x = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
    b_sobel_y = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
    g_sobel_x = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    g_sobel_y = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    r_sobel_x = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)
    r_sobel_y = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3)

    # Convert back to uint8
    b_sobel_x = np.uint8(np.absolute(b_sobel_x))
    b_sobel_y = np.uint8(np.absolute(b_sobel_y))
    g_sobel_x = np.uint8(np.absolute(g_sobel_x))
    g_sobel_y = np.uint8(np.absolute(g_sobel_y))
    r_sobel_x = np.uint8(np.absolute(r_sobel_x))
    r_sobel_y = np.uint8(np.absolute(r_sobel_y))

    # Merge the Sobel channels
    sobel = cv2.merge((b_sobel_x, g_sobel_y, r_sobel_x))

    return sobel

# Sharpen the image
def sharpen_image(image):
    # Create the sharpening kernel 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

    # Sharpen the image 
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image

# Enhance RGB image quality
def enhance_rgb_quality(image):
    """Enhance the quality of an RGB image using histogram equalization."""
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

    # Convert back to RGB color space
    enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    return enhanced_image

# Remove blur effect from RGB image
def remove_blur_effect(image):
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Apply Wiener deconvolution to each color channel to deblur
    deblurred_b = wiener(b, (5, 5), 0.01)
    deblurred_g = wiener(g, (5, 5), 0.01)
    deblurred_r = wiener(r, (5, 5), 0.01)

    # Convert the deblurred images back to uint8
    deblurred_b = np.uint8(deblurred_b)
    deblurred_g = np.uint8(deblurred_g)
    deblurred_r = np.uint8(deblurred_r)

    # Merge the deblurred color channels
    deblurred = cv2.merge((deblurred_b, deblurred_g, deblurred_r))

    return deblurred


# Apply Mediane filter
def apply_median_filter(image, ksize=5):
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Apply median blur to each color channel separately
    blurred_b = cv2.medianBlur(b, ksize)
    blurred_g = cv2.medianBlur(g, ksize)
    blurred_r = cv2.medianBlur(r, ksize)

    # Merge the blurred color channels
    blurred = cv2.merge((blurred_b, blurred_g, blurred_r))

    return blurred

# Adjusts the brightness by adding 10 to each pixel value 
def adjust_brightness(image, brightness=10, contrast=1.0):
    adjusted_image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
    return adjusted_image  

# Rotation
def rotate_image(image, angle=90, scale=1.0):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

# Cropping
def crop_image(image, x, y, w, h):
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Resize
def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized_image = cv2.resize(image, dim, interpolation=inter)
    return resized_image

# Zooming
def zoom_image(image, zoom_factor=1.5):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
    zoomed_image = cv2.warpAffine(image, M, (w, h))
    return zoomed_image

# Edge detection
def detect_edges(image):
    """Detect edges in the input image using the Canny Edge Detector."""
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Setting parameter values
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold

    # Applying the Canny Edge filter
    edge = cv2.Canny(gray_image, t_lower, t_upper)

    return edge

# Dilation
def dilate_image(image, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# Erosion
def erode_image(image, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# Opening
def open_image(image, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing
def close_image(image, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Bilateral filter
def apply_bilateral_filter(image, d=3, sigmaColor=15, sigmaSpace=15):
    # Apply Bilateral Filter
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)