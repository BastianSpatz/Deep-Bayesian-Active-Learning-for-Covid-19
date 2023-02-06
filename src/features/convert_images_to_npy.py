import argparse
import os

import cv2
import numpy as np
import pandas as pd


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

        images = [x for x in os.listdir(path_to_volume_folder)]
        print("Patient: {} scan id: {}".format(
            row["patient_id"], row["scan_id"]), end="\r")
        if row["label"] == "CP":
            y = 0
        elif row["label"] == "NCP":
            y = 1
        elif row["label"] == "Normal":
            y = 2

        volume = []
        if len(images) <= 3:
            continue
        image_path = path_to_volume_folder + "/" + images[0]
        image = cv2.imread(image_path)
        if image.shape != (512, 512, 3):
            print("image has wrong dimension: {}".format(image.shape))
            continue
        for image in images:
            image_path = path_to_volume_folder + "/" + image
            image = cv2.imread(image_path)
            volume.append(np.array(image))
        if len(images) < 64:
            num_extend = 64 - len(images)
            extend_list = [volume[-1] for _ in range(num_extend)]
            volume.extend(extend_list)
            print("extended volume {} by {} slices".format(
                image_path, num_extend))
        try:
            if not os.path.exists(args.save_path + "volumes"):
                # Create a new directory because it does not exist
                os.makedirs(args.save_path + "volumes")
            if not os.path.exists(args.save_path + "labels"):
                # Create a new directory because it does not exist
                os.makedirs(args.save_path + "labels")
            np.save(args.save_path + "volumes/" + "vol_" + row["label"] + "_" + str(
                row["patient_id"]) + "_" + str(row["scan_id"]), np.array(volume))
            np.save(args.save_path + "labels/" + "label_" + row["label"] + "_" + str(
                row["patient_id"]) + "_" + str(row["scan_id"]), np.array(y))
        except Exception as e:
            print(e)


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
