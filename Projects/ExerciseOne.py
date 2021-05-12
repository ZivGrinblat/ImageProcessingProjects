import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


r_to_y = np.array([[0.299, 0.587, 0.114],
                   [0.596, -0.275, -0.321],
                   [0.212, -0.523, 0.311]])
y_to_r = np.array([[1, 0.956, 0.619],
                   [1, -0.272, -0.647],
                   [1, -1.106, 1.703]])


def read_image(filename, representation):
    """
    Returns representation of image at filename in float64
    :param filename: the filename of an image on disk (could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
           image (1) or an RGB image (2)
    :return: an image
    """
    img = imageio.imread(filename) / 255
    return img if representation == 2 else rgb2gray(img)


def imdisplay(filename, representation):
    """
    Displays image according to representation
    :param filename: the filename of an image on disk (could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
           image (1) or an RGB image (2)
    :return: None
    """
    img = read_image(filename, representation)
    if representation == 1: plt.imshow(img, cmap='gray')
    if representation == 2: plt.imshow(img)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transforms RGB img into the YIQ color space
    :param imRGB:
    :return: imRGB transformed into YIQ color space
    """
    return imRGB @ r_to_y.T


def yiq2rgb(imYIQ):
    """
    Transforms YIQ img into the RGB color space
    :param imYIQ:
    :return: imYIQ transformed into RGB color space
    """
    return imYIQ @ y_to_r.T


def histogram_equalize(im_orig):
    """
    Perform hist equalization on RGB or grayscale im_orig
    :param im_orig:
    :return:
    """
    im_yiq = None
    im_cur = im_orig
    if len(im_orig.shape) == 3: # RGB case
        im_yiq = rgb2yiq(im_orig)
        im_cur = im_yiq[:, :, 0]

    # Steps 1-7 of equalization
    hist_orig, bins = np.histogram(im_cur, bins = np.arange(257) / 255)
    bins = bins[:-1]
    hist_cum = hist_orig.cumsum()
    hist_cum = 255 * hist_cum / hist_cum[-1]
    hist_min, hist_max = hist_cum[hist_cum > 0][0], np.max(hist_cum)
    hist_cum = np.round(255 * (hist_cum - hist_min) / (hist_max - hist_min)).astype(np.int)
    im_eq = hist_cum[(im_cur * 255).astype(np.int)] / 255

    hist_eq, bins_eq = np.histogram(im_eq, bins = np.arange(257) / 255)
    if len(im_orig.shape) == 3: # If needed, insert image into YIQ image and transform into RGB
        im_yiq[:, :, 0] = im_eq
        im_eq = yiq2rgb(im_yiq)

    return [im_eq, hist_orig, hist_eq]


def q_calc(z_indices, hist):
    """
    Calculate qs based on z indices and histogram
    :param z_indices:
    :param hist:
    :return:
    """
    q_arr = np.array([])
    for i in range(len(z_indices) - 1):
        numerator, denominator = 0, 0
        for g in range(int(np.floor(z_indices[i])) + 1, int(np.floor(z_indices[i + 1])) + 1):
            numerator += g * hist[g]
            denominator += hist[g]
        q_arr = np.append(q_arr, numerator / denominator)
    return q_arr


def z_calc(q_arr):
    """
    Calculate z indices based on q_arr
    :param q_arr:
    :return:
    """
    z_indices = np.array([0])
    for q_idx in range(1, len(q_arr)):
        z_indices = np.append(z_indices, ((q_arr[q_idx - 1] + q_arr[q_idx]) / 2))
    return np.append(z_indices, 255)


def error_calc(hist_orig, z_indices, q_arr):
    """
    Calculate error between original image and calculated z and q arrays
    :param hist_orig:
    :return:
    """
    error = 0
    for i in range(len(z_indices) - 1):
        for g in range(int(np.floor(z_indices[i])) + 1, int(np.floor(z_indices[i + 1])) + 1):
            error += ((q_arr[i] - g) ** 2) * hist_orig[g]
    return error


def quantize(im_orig, n_quant, n_iter):
    """
    Return quantized image and array of errors in each iteration
    :param im_orig:
    :param n_quant: number of intensities in new image
    :param n_iter: number of iterations for computing z and q
    :return: quantized image and errors
    """
    im_yiq = None
    im_cur = im_orig
    if len(im_orig.shape) == 3: # RGB case
        im_yiq = rgb2yiq(im_orig)
        im_cur = im_yiq[:, :, 0]

    # Calculate initial z indices
    hist_orig, bins_orig = np.histogram(im_cur, bins = np.arange(257) / 255)
    z_indices = np.array([0])
    for i in range(1, n_quant):
        z_indices = np.append(z_indices, np.quantile(im_orig * 255, i / n_quant))
    z_indices = np.append(z_indices, 255)

    # Calculate initial qs, initialize error array
    q_arr = q_calc(z_indices, hist_orig)
    error = np.array([])

    # Iterate quantization
    for step in range(n_iter - 1):
        new_z_indices = z_calc(q_arr)
        if np.array_equal(z_indices, new_z_indices): break
        z_indices = new_z_indices
        q_arr = q_calc(z_indices, hist_orig)
        error = np.append(error, error_calc(hist_orig, z_indices, q_arr))
    z_indices = np.ceil(z_indices).astype(int)

    # Create new mapping and apply
    new_map = np.array([])
    for i in range(len(q_arr)):
        new_map = np.append(new_map, np.array([[q_arr[i]] * (z_indices[i + 1] - z_indices[i])]))
    new_map = np.append(new_map, q_arr[-1])
    im_quant = new_map[(im_cur * 255).astype(np.int)] / 255

    if len(im_orig.shape) == 3: # If needed, insert image into YIQ image and transform into RGB
        im_yiq[:, :, 0] = im_quant
        im_quant = yiq2rgb(im_yiq)
    return [im_quant, error]


def greatest_range(im_orig):
    """
    Find which color has the greatest difference between max and min
    :param img:
    :return:
    """
    r_range, g_range, b_range = None, None, None
    if len(im_orig.shape) == 2:
        r_range = np.max(im_orig[:, 0]) - np.min(im_orig[:, 0])
        g_range = np.max(im_orig[:, 1]) - np.min(im_orig[:, 1])
        b_range = np.max(im_orig[:, 2]) - np.min(im_orig[:, 2])
    if len(im_orig.shape) == 3:
        r_range = np.max(im_orig[:, :, 0]) - np.min(im_orig[:, :, 0])
        g_range = np.max(im_orig[:, :, 1]) - np.min(im_orig[:, :, 1])
        b_range = np.max(im_orig[:, :, 2]) - np.min(im_orig[:, :, 2])
    arr = np.array([r_range, g_range, b_range])
    return np.argmax(arr)


def quantize_rgb_helper(im_orig, n_quant, color_arr, idx):
    """
    Recursive function that splits image into boxes and takes the average color.
    Adds newly calculated colors to color_arr
    :param im_orig:
    :param n_quant:
    :param x1: left side x bound
    :param x2: right side x bound
    :param y1: top side y bound
    :param y2: bottom side y bound
    :return:
    """
    # Base case
    if n_quant == 1:
        im_temp = im_orig.reshape(-1, 3)
        new_color = np.mean(im_temp, axis=0)
        color_arr[idx[0]] = new_color
        idx[0] += 1
    else:
        # Calculate splits for both halves
        small_half, large_half = int(2 ** np.floor(np.log2(n_quant) - 1)), None
        if n_quant - small_half < small_half / 2: small_half /= 2
        large_half = n_quant - small_half

        # Find color with greatest range to split the chunk
        color = greatest_range(im_orig)
        im_orig = im_orig.reshape(-1, 3)
        im_orig = im_orig[im_orig[:, color].argsort()]
        first_half = im_orig[:im_orig.shape[0]//2]
        second_half = im_orig[im_orig.shape[0]//2:]
        quantize_rgb_helper(first_half, small_half, color_arr, idx)
        quantize_rgb_helper(second_half, large_half, color_arr, idx)


def quantize_rgb(im_orig, n_quant):
    """
    Uses quantize_rgb_helper to calculate new colors, then maps them
    :param im_orig:
    :param n_quant:
    :return: quantized RGB image
    """
    copy_im = im_orig.copy()
    my_shape = copy_im.shape
    color_arr = [0] * n_quant # Array for storing new palette
    quantize_rgb_helper(copy_im, n_quant, color_arr, [0])

    # Pick R,G or B for quantization and sort it
    color = greatest_range(copy_im)
    color_arr = np.array(color_arr)
    color_arr = color_arr[color_arr[:, color].argsort()]

    # Set bounds based on color amount
    copy_im = copy_im.reshape(-1, 3)
    z_indices = np.array([0])
    for i in range(1, n_quant):
        z_indices = np.append(z_indices, np.quantile(copy_im[:, color] * 255, i / n_quant))
    z_indices = np.append(z_indices, 255) / 255

    # Map colors
    for i in range(len(color_arr)):
        copy_im[np.logical_and(copy_im[:, color] >= z_indices[i], copy_im[:, color] < z_indices[i + 1])] = color_arr[i]
    copy_im[copy_im[:, color] == z_indices[-1]] = color_arr[-1] # map color intensities = 1

    return copy_im.reshape(my_shape)