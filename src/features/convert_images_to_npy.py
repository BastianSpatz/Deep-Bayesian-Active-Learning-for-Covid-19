import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

# Spline interpolated zoom (SIZ)
def change_depth_siz(volume, desired_depth):
    current_depth = volume.shape[0]
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    volume_new = zoom(volume, (depth_factor, 1, 1), mode='nearest')
    return volume_new


def check_images(volume_path, image_paths):
    # check if volume is 'appropriate'
    if len(image_paths) <= 3:
        return False
    
    random_image_path_idx = random.randint(0, len(image_paths))
    image = cv2.imread(os.path.join(volume_path, image_paths[random_image_path_idx]),  cv2.IMREAD_GRAYSCALE)
    if image.shape != (512, 512):
        Exception("image {} has wrong dimension: {}".format(image_paths[random_image_path_idx], image.shape))
        return False
    
    if image_paths[random_image_path_idx][-3:] != "png":
        Exception("image {} has wrong datatype: {}".format(image_paths[random_image_path_idx], image_paths[random_image_path_idx][-3:]))
        return False
    
    try:
        image_path = volume_path + "/" + image_paths[len(image_paths)//2]
        image = cv2.imread(image_path)
        unique, counts = np.unique(image, return_counts=True)
        mapColorCounts = dict(zip(unique, counts))  
        sum = 0
        for key in mapColorCounts.keys():
            if key != 0:
                sum += mapColorCounts[key]
        if mapColorCounts[0] >= sum:
            print("image {} has more black pixel than anything else".format(image_path))
            return False
    except Exception as e:
        print("could not check {}".format(image_path))
        print(e)
        return True
    
    return True


def convert_images_to_npy(args):
    path_to_folders = args.file_path
    df = pd.read_csv(args.file_names_csv)
    for _, row in df.iterrows():
        path_to_volume_folder = path_to_folders + \
            row["label"] + "/" + str(row["patient_id"]) + \
            "/" + str(row["scan_id"])
        if not os.path.exists(path_to_volume_folder):
            print("Could not find volume folder {}".format(path_to_volume_folder))
            continue
        if os.path.exists(args.save_path + "volumes/" + "vol_" + row["label"] + "_" + str(row["patient_id"]) + "_" + str(row["scan_id"]) + ".npy"):
            print("Volume already exists {}".format(path_to_volume_folder))
            continue
        
        # get volume and associated class
        images = [x for x in os.listdir(path_to_volume_folder)]
        print("Patient: {} scan id: {}".format(
            row["patient_id"], row["scan_id"]), end="\r")
        if row["label"] == "CP":
            y = 0
        elif row["label"] == "NCP":
            y = 1
        elif row["label"] == "Normal":
            y = 2

        
        # check if volume is 'appropriate'
        if len(images) < 10:
            continue

        image_path = path_to_volume_folder + "/" + images[0]
        image = cv2.imread(image_path)
        if image.shape != (512, 512, 3):
            print("image {} has wrong dimension: {}".format(image_path, image.shape))
            continue
        if image_path[-3:] != "png":
            print("image {} has wrong datatype: {}".format(image_path, image_path[-3:]))
            continue
        try:
            image_path = path_to_volume_folder + "/" + images[len(images)//2]
            image = cv2.imread(image_path)
            unique, counts = np.unique(image, return_counts=True)
            mapColorCounts = dict(zip(unique, counts))  
            sum = 0
            for key in mapColorCounts.keys():
                if key != 0:
                    sum += mapColorCounts[key]
            if mapColorCounts[0] >= sum:
                print("image {} has more black pixel than anything else".format(image_path))
                continue
        except Exception as e:
            print("could not check {}".format(image_path))
            print(e)

        # unique, counts = np.unique(image, return_counts=True)
        # mapColorCounts = dict(zip(unique, counts))
        # if mapColorCounts[0] > 512*512//2:
        #     continue
        volume = []
        try:
            for image in images:
                image_path = path_to_volume_folder + "/" + image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if any((np.array(image) == x).all() for x in volume):
                    print("image already in volume {}".format(path_to_volume_folder))
                    continue
                volume.append(np.array(image))
        except Exception as e:
            print("could not check for duplicates")
            print("skipping this volume {}".format(image_path))
            print(e)
            continue

        volume = change_depth_siz(np.array(volume), desired_depth=64)
        try:
            if not os.path.exists(args.save_path + "volumes"):
                # Create a new directory because it does not exist
                os.makedirs(args.save_path + "volumes")
            if not os.path.exists(args.save_path + "labels"):
                # Create a new directory because it does not exist
                os.makedirs(args.save_path + "labels")
            np.save(args.save_path + "volumes/" + "vol_" + row["label"] + "_" + str(
                row["patient_id"]) + "_" + str(row["scan_id"]), volume)
            np.save(args.save_path + "labels/" + "label_" + row["label"] + "_" + str(
                row["patient_id"]) + "_" + str(row["scan_id"]), np.array(y))
        except Exception as e:
            print(e)
            print("could not save image {} of size: {}".format(path_to_volume_folder, len(volume)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
	Utility script to convert commonvoice into wav and create the training and test json files for speechrecognition. """
                                     )
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to the extracted zipfiles')
    parser.add_argument('--file_names_csv', type=str, default=None, required=True,
                        help='path to the zip_file_names.csv file. Check http://ncov-ai.big.ac.cn/download?lang=en')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='path to the save location of the .npy files')
    args = parser.parse_args()

    convert_images_to_npy(args)
