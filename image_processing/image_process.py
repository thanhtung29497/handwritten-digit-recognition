"""
    Module contains image processing function
"""
from skimage.io import imread
from skimage import measure
from skimage.filters import threshold_yen
from skimage.measure import regionprops
from skimage.transform import resize
from skimage import exposure
import matplotlib.patches as patches
from numpy import zeros

def load_image(path):
    """
        Read image from path
    """
    return imread(path, as_grey=True)


def to_binary_image(image):
    """
        convert image to binary image using custom threshold algorithm
    """
    threshold_value = threshold_yen(image)
    binary_car_image = image < threshold_value
    return binary_car_image


def resize_to_fixed_width(image, width):
    """
        resize image to fixed width but keep ratio between width and height
    """
    resized_ratio = image.shape[0] / width
    height = round(image.shape[1] / resized_ratio)
    return resize(image, (width, height))


def resize_to_fixed_height(image, height):
    """
        resize image to fixed height but keep ratio between width and height
    """
    resized_ratio = image.shape[1] / height
    width = round(image.shape[0] / resized_ratio)
    return resize(image, (width, height))


def make_border(image, size):
    """
        create borders of zeros around image to make image reach a fixed size
    """
    result = zeros(size)
    start_row = round((size[0] - image.shape[0]) / 2)
    start_col = round((size[1] - image.shape[1]) / 2)
    result[start_row:start_row + image.shape[0], start_col:
           start_col + image.shape[1]] = image
    return result


def localize_image(image):
    """
        detect connected regions of pixel in a binary image
    """
    label_image = measure.label(image)
    return regionprops(label_image)


def gamma_correction(image):
    """
        apply gamma correction to an image
    """
    return exposure.adjust_gamma(image, 2)


def log_correction(image):
    """
        apply logarithm correction to an image
    """
    return exposure.adjust_log(image, 1)


def highlight_region(region, axe):
    """
        draw a red rectangle around an region of image
    """
    min_row, min_col, max_row, max_col = region.bbox
    rect_border = patches.Rectangle(
        (min_col, min_row),
        max_col - min_col,
        max_row - min_row,
        edgecolor="red",
        linewidth=2,
        fill=False)
    axe.add_patch(rect_border)


def filter_digit_like(regions, image):
    """
        filter all regions which have area greater then 0.1% of the image
    """
    digit_like_region = []
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        area = (max_col - min_col) * (max_row - min_row)
        if area > image.shape[0] * image.shape[1] / 1000:
            digit_like_region.append(region)
    return digit_like_region
