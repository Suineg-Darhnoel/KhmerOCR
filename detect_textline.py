import cv2 as cv
# a library for image processing (available in CPP or Python)
from skimage.filters import threshold_otsu
# based on otsu's method for thresholding
import numpy as np
# fundamental package for scientfic computing in Python
import matplotlib.pyplot as plt
# a library for visualizations in Python


def on_trackbar(arg):
    """
    on_trackbar
    -----------
        Do something when trackbar is used
    """
    pass


def map_value(
            old_val,
            old_interval,
            new_interval
        ):
    """
    map_value
    ---------
        translates the old value in the old_interval
        to a corresponding one in the new_interval
        ex: 0.5 in [0, 1] becomes 5 in [0, 10]

    Parameters
    ----------
        old_val: old value input
        old_interval: an iterable object consisting of two components
        new_interval: an iterable object consisting of two components

    """
    old_min, old_max = old_interval
    new_min, new_max = new_interval

    assert (old_min <= old_val <= old_max), "{} does not belong to {}"\
        .format(old_val, old_interval)

    old_scale = old_max - old_min
    new_scale = new_max - new_min

    new_val = (old_val - old_min)*(new_scale / old_scale) + new_min
    return new_val


def preprocess(image, apply_blur=False):
    """
    preprocess
    ----------
        (1) converts the color image to a grayscale one.
            (optional 1) : apply GaussianBlur
        (2) automatically computes for threshold by applying otsu's method to
        distinguish texts from background.

    returns the final preprocessed 1-channel image
    Parameters:
    -----------
        image:
            Use BGR color image
        apply_blur:
            For experiment lets see the difference when use
            GaussianBlur and when no blur is applied
    """
    # convert BGR img to GRAYSCALE img
    text_gray = cv.cvtColor(text_img, cv.COLOR_BGR2GRAY)

    if apply_blur:
        text_gray = cv.GaussianBlur(text_gray, (5, 5), 0)

    # threshold for separating the text from its backgroud
    img_threshold = threshold_otsu(text_gray)
    thres_bool_img = text_gray <= img_threshold
    thres_img = thres_bool_img.astype(np.uint8)

    tmp = 255*np.ones(image.shape[:2])
    final_img = cv.bitwise_or(tmp, tmp, mask=thres_img)

    return final_img


def get_rotated_image(prep_image):
    """
    get_rotated_image
    -----------------
        returns an image where all texts perpendicular to the vertical frame

    Parameters:
    ----------
        prep_image:
            can be computed by preprocess(source_image)
    """
    pts = cv.findNonZero(prep_image)
    ret = cv.minAreaRect(pts)
    # get the locations of retangle points and the angle
    (cx, cy), (w, h), ang = ret

    if w > h:
        w, h = h, w
        ang += 90

    M = cv.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated_image = cv.warpAffine(
                            prep_image, M,
                            (prep_image.shape[1], prep_image.shape[0])
                        )
    return rotated_image


def get_upper_lower_poses(stats, image_shape, threshold):
    """
    get_upper_lower_poses
    ---------------------
        tells where every upper and bottom line of a text should be drawn

    Parameters:
    -----------
        stats:
            can be computed by get_stats(rotated_image)
                (see get_stats function)
        image_shape:
            shape of the grayscale image corresponding to the original one
        threshold:
            should be chosen from the interval [stats_min, stats_max]
            the default for this experiment is stats_avg
    """
    H, W = image_shape
    uppers = [
                i for i in range(H-1)
                if stats[i] <= threshold
                and stats[i+1] > threshold
            ]

    lowers = [
                i for i in range(H-1)
                if stats[i] > threshold
                and stats[i+1] <= threshold
            ]

    return uppers, lowers


def get_stats(rotated_image):
    """
    get_stats
    ---------
        tells how much pixels contain in each row of the preprocessed image

    Parameters:
    -----------
        rotated_image:
            can be computed by get_rotated_image(prep_image)

    """
    stats = cv.reduce(rotated_image, 1, cv.REDUCE_AVG).reshape(-1)
    return stats


def draw_lines_on(source_image, padding=0, threshold=None):
    """
    draw_lines_on
        by looking at the text image
        it does computation then outputs an image
        with upper and lower lines of the texts.

    Parameters:
    -----------
        source_image:
            Use BGR color image
        padding:
            After lines have been found, without padding drawn lines
            just fit to the top and bottom of the texts.
            Increasing padding will be useful for some writting system
            where superscript is used.

    """
    prep_image = preprocess(source_image)

    rotated_image = get_rotated_image(prep_image).astype(np.uint8)
    rotated_color_image = cv.cvtColor(rotated_image, cv.COLOR_GRAY2BGR)
    stats = get_stats(rotated_image)

    if threshold is None:
        threshold = np.average(stats)

    H, W = source_image.shape[:2]
    uppers, lowers = get_upper_lower_poses(stats, (H, W), threshold)

    # drawing process
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)

    for upper in uppers:
        cv.line(rotated_color_image,
                (0, upper-padding),
                (W, upper-padding),
                color_green, 1)

    for lower in lowers:
        cv.line(rotated_color_image,
                (0, lower-padding),
                (W, lower-padding),
                color_red, 1)

    # for readibility let's use the inverted version of the image
    inverted_image = cv.bitwise_not(source_image)

    # blend the original with the rotated_color_image
    alpha, beta = 1, 1
    blended_image = cv.addWeighted(
                inverted_image,
                alpha, rotated_color_image, beta, 0
            )

    return blended_image


if __name__ == "__main__":
    text_img = cv.imread("text_images/khmer_handwriting_1.jpg")
    init_stats = get_stats(get_rotated_image(preprocess(text_img)))

    # GUI
    # variable initialization
    stats_min = int(np.min(init_stats))
    stats_max = int(np.max(init_stats))

    stats_interval = (stats_min, stats_max)
    trackbar_interval = (0, 100)

    thres = np.average(init_stats)
    # naming
    window_name = "drawlines"
    trackbar_name = "my_trackbar"

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.createTrackbar(
                trackbar_name,
                window_name,
                *trackbar_interval,
                on_trackbar
            )

    while (True):
        final_image = draw_lines_on(text_img, padding=0, threshold=thres)
        trackbar_thres = cv.getTrackbarPos(trackbar_name, window_name)
        thres = map_value(trackbar_thres, trackbar_interval, stats_interval)
        cv.imshow(window_name, final_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
