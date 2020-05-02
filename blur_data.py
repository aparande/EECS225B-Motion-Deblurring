from motionflow import *
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Training Data Generation')
parser.add_argument("--data-dir", type=str, required=True, dest="output_dir")
parser.add_argument("-n", type=int, dest="n", required=True, help='How many blurred copies to produce per unblurred image')

args = parser.parse_args()

all_img_path = os.path.join(args.output_dir, "Images")
with open(os.path.join(args.output_dir, "ids.txt"), 'r') as id_file:
    ids = [img_id.replace("\n", "") for img_id in id_file.readlines()]

for img_id in tqdm(ids):
    img_dir = os.path.join(all_img_path, img_id)
    os.makedirs(img_dir, exist_ok=True)
    base_image_path = os.path.join(img_dir, img_id + ".jpg")
    
    tag_offset = 0
    if os.path.exists(base_image_path):
        tag_offset = len([name for name in os.listdir(img_dir) if os.path.isfile(name)]) - 1
    elif os.path.exists(os.path.join(all_img_path, img_id + ".jpg")):
        os.rename(os.path.join(all_img_path, img_id + ".jpg"), base_image_path)
    else:
        print("Error: Could not find unblurred image for {}".format(img_id))
        continue

    tag_offset += 1
    img = plt.imread(base_image_path)
    for i in range(args.n):
        img_mild_blur = blur_image(img, sample_motion_flow(img, params=PARAMS_MILD_BLUR))
        img_mild_blur = 255 * (img_mild_blur - img_mild_blur.min()) / (img_mild_blur.max() - img_mild_blur.min())

        blurred_path = os.path.join(img_dir, "{}_{}.jpg".format(img_id, i + tag_offset))
        plt.imsave(blurred_path, img_mild_blur.astype('uint8'))