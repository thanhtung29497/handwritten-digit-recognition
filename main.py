"""
    main module
"""

import sys
import math
from tensorflow import train, InteractiveSession
import image_processing as imp
import matplotlib.pyplot as plt
import numpy as np

sess = InteractiveSession()


def restore_model(model_name):
    """
        restore all variables and placeholders of model_name
    """
    model_path = "./trained_model/" + model_name + "/" + model_name + ".meta"
    saver = train.import_meta_graph(model_path)
    graph = sess.graph
    saver.restore(sess,
                  train.latest_checkpoint("./trained_model/" + model_name))
    return graph


def preprocess_image(image):
    # convert to binary image
    binary_image = imp.to_binary_image(image)

    # detect connected regions
    regions = imp.localize_image(binary_image)

    # choose the regions which is "digit-like"
    digit_like_region = imp.filter_digit_like(regions, image)

    # convert each region to a normalized digit image
    digit_images = []
    for region in digit_like_region:
        min_row, min_col, max_row, max_col = region.bbox

        # crop region from binary image
        digit_image = binary_image[min_row:max_row, min_col:max_col]

        # reshape digit image to normalized size (28x28)
        if digit_image.shape[0] > digit_image.shape[1]:
            digit_image = imp.make_border(
                imp.resize_to_fixed_width(digit_image, 20), (28, 28))
        else:
            digit_image = imp.make_border(
                imp.resize_to_fixed_height(digit_image, 16), (28, 28))
        digit_images.append(digit_image)

    return binary_image, digit_like_region, digit_images

def plot_images (image, binary_image, digit_like_regions, digit_images, predicted_numbers):
    """
        display image, binary_image, highlight digit_like_regions
        display digit_images and prediction
    """

    # display image, binary_image
    fig, axe = plt.subplots(1, 2)
    axe[0].imshow(image, cmap="gray")
    axe[1].imshow(binary_image, cmap="gray")
    plt.axis("off")

    # highlight digit_like_regions
    for region in digit_like_regions:
        imp.highlight_region(region, axe[0])

    # divide plot to multiple rows, each row has 10 subplots
    fig, axes = plt.subplots(
        math.floor((len(digit_like_regions) - 1) / 10) + 1,
        math.floor(min(10, len(digit_like_regions))))

    row = 0
    col = 0

    for index, digit_image in enumerate(digit_images):

        # find position of the subplot which will display the digit_image
        if (len(digit_images) > 10):
            axe = axes[row][col]
        elif len(digit_like_regions) > 1:
            axe = axes[col]
        else:
            axe = axes

        # display the digit_image and prediction
        axe.imshow(digit_image, cmap="gray")
        axe.axis("off")
        axe.text(0, 7, predicted_numbers[index], color="white")

        # calculate the row and col
        col = col + 1
        if (col == 10):
            col = 0
            row = row + 1

    plt.show()


def test(model_name, image_path):
    """
        predict all digits which appear in image using model_name
    """
    # restore retrained model
    graph = restore_model(model_name)
    inputs = graph.get_tensor_by_name("inputs:0")
    prediction = graph.get_tensor_by_name("prediction:0")

    # preprocess image to get all digit-like images which appear in image (and other properties to plot)
    test_image = imp.load_image(image_path)
    binary_test_image, digit_like_regions, digit_images = preprocess_image(
        test_image)

    # use model to recognize digits
    predicted_numbers = []
    for digit_image in digit_images:
        normalized_image = np.reshape(digit_image, (1, 784))
        predicted_number = sess.run(
            prediction, feed_dict={
                inputs: normalized_image
            })[0]
        predicted_numbers.append(predicted_number)

    # plot image and result of recognition
    plot_images(test_image, binary_test_image, digit_like_regions,
                digit_images, predicted_numbers)

if __name__ == "__main__":
    if sys.argv[1] == "train":
        __import__("trains." + sys.argv[2])
    elif sys.argv[1] == "test":
        test(sys.argv[2], sys.argv[3])
