"""
TODO:
    1. Make a method for checking for a single shape of all images in the list   
    2. Leave only one border width variable. No need on vertical and horizaontal one

"""


from glob import glob
import cv2
import numpy as np
import moviepy.editor as mpy


BORDER_COEFF = 0.15


def _get_shape(im_path_A, im_path_B):
    im_A = cv2.imread(im_path_A)
    im_B = cv2.imread(im_path_B)

    assert im_A.shape == im_B.shape, "Images from both must have equal shapes"

    return im_A.shape


def check_lists(lists):
    """
        Check whether list have equal number of elements
    """

    
    all_lenghts = [len(lst) for lst in lists]

    default_length = all_lenghts[0]
    check_array = [len(lst) == default_length for lst in lists]

    assert all(check_array) == True, f"All given lists must have equal number of images, but they have {all_lenghts} images"


def make_paired_gif(*args, save_path="./animation.gif", fps=25):
    """
        Method creates a gif-image with images from lists in args
        list_A, list_B - sorted lists with path to images
        
        :param args: variable number of lists with pathes to images
        :param save_path: path for saving gif-image with gif-name
        :param fps: frames-per-second for gif-image
        :return: 
    
    """
    image_lists = list(args)
    check_lists(image_lists)

    num = len(image_lists) # number of given image collections
    print(f"{num} image collections are processing")
    
    h, w, c = _get_shape(image_lists[0][0], image_lists[1][0])        

    vbw = int(w * BORDER_COEFF) # vertical border width
    hbv = int(h * BORDER_COEFF) # horizantal borders width

    # creating background
    width = int(num*w + (num + 1)*vbw)
    height = int(h + 2 * hbv)
    background = np.zeros((height, width, 3))


    im_sequence = []
    for im_A, im_B in zip(list_A, list_B):
        im_A = cv2.imread(im_A)
        im_B = cv2.imread(im_B)
        shape = im_A.shape

        assert im_A.shape == im_B.shape, "Images from both must have equal shapes"

        # putting images on the plane
        background[hbv:-hbv, vbw:vbw+w, :] = im_A[...,::-1]
        background[hbv:-hbv, (num_images + 1)*vbw + (num_images - 1)*w:-vbw, :] = im_B[...,::-1]
        
        im_sequence.append(background)

    clip = mpy.ImageSequenceClip(im_sequence, fps=fps)
    clip.write_gif(save_path, fps=fps)
    """


if __name__ == "__main__":
    RESULTS_DIR = "/home/dmitry.klimenkov/repos/vid2vid/results/pose2body_256p_g2_flatback_v2/test_latest/27/"
    IM_DIR = "/media/data/klimenkov/datasets/vid2vid_3/test_img/27/"


    list_A = sorted(glob(RESULTS_DIR + "/fake*.jpg"))
    list_B = sorted(glob(RESULTS_DIR + "/real*.jpg"))
    list_C = sorted(glob(IM_DIR + "/*.jpg"))

    #check_lists([list_A, list_B, list_C[2:]])

    make_paired_gif(list_A, list_B, list_C[2:], save_path=IM_DIR+"/animation.gif")

        