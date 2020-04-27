from motionflow import *
from os import listdir
import matplotlib.pyplot as plt

# Specify relevant file paths
verbose = True
base_path = "./BSDS300/images/train/" # directory with the images to blur
out_path = "./blurimages/train/" # directory to save blurred results to
img_paths = listdir(base_path)
total = len(img_paths)

for i, rel_path in enumerate(img_paths):
    if verbose:
        print("{0} of {1} done".format(i, total))
    
    # Blur
    img = plt.imread(base_path + rel_path)
    img_mild_blur = blur_image(img, sample_motion_flow(img, params=PARAMS_MILD_BLUR))
    img_heavy_blur = blur_image(img, sample_motion_flow(img, params=PARAMS_HEAVY_BLUR))
    
    # Renormalize
    img_mild_blur = 255 * (img_mild_blur - img_mild_blur.min()) / (img_mild_blur.max() - img_mild_blur.min())
    img_heavy_blur = 255 * (img_heavy_blur - img_heavy_blur.min()) / (img_heavy_blur.max() - img_heavy_blur.min())
    
    # Save
    plt.imsave(out_path + rel_path[:-4] + "_mild_blur.jpg", img_mild_blur.astype("uint8"))
    plt.imsave(out_path + rel_path[:-4] + "_heavy_blur.jpg", img_heavy_blur.astype("uint8"))

if verbose:
    print("{} of {} done".format(total, total))