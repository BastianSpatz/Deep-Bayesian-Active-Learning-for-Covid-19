import argparse
import os
import random
import csv
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from src.features.utils import exterior_exclusion


_CLASS_MAP = {"CP": 0, "NCP": 1, "Normal": 2}
_LESION_FILE = "lesions_slices.csv"
_EXTRA_LESION_FILE = "extra_lesion_slices.csv"
_UNZIP_FILE = "unzip_filenames.csv"
_EXCLUDE_FILE = "exclude_list.txt"


# Cases accidentally included that are removed in v2+
_PATCH_CASES = [
    "NCP_328_1805",
    "CP_1781_3567",
    "CP_1769_3516",
    "NCP_1058_2635",
    "NCP_868_2395",
    "NCP_868_2396",
    "NCP_869_2397",
    "NCP_911_2453",
]


def process_cncb_data(
    root_dir, exclude_file=_EXCLUDE_FILE, extra_lesion_files=_EXTRA_LESION_FILE
):
    """Process slices for all included CNCB studies"""
    # Get file paths
    lesion_files = [os.path.join(root_dir, _LESION_FILE)]
    if extra_lesion_files is not None:
        lesion_files += [os.path.join(root_dir, extra_lesion_files)]
    unzip_file = os.path.join(root_dir, _UNZIP_FILE)
    exclude_file = os.path.join(root_dir, exclude_file)
    image_files, classes = _get_files(lesion_files, unzip_file, exclude_file, root_dir)
    # filenames = [os.path.basename(f) for f in image_files]

    return image_files, classes


def _get_lesion_files(lesion_files, exclude_list, root_dir):
    """Reads the lesion files to identify relevant
    slices and returns their paths"""
    files, classes = [], []
    for lesion_file in lesion_files:
        with open(lesion_file, "r") as f:
            f.readline()
            for line in f.readlines():
                cls, pid, sid = line.split("/")[:3]
                if pid not in exclude_list:
                    # Patch to remove a few erroneous files
                    case_str = "{}_{}_{}".format(cls, pid, sid)
                    if case_str in _PATCH_CASES:
                        continue
                    if os.path.exists(os.path.join(root_dir, line.strip("\n"))):
                        files.append(os.path.join(root_dir, line.strip("\n")))
                        classes.append(_CLASS_MAP[cls])
    return files, classes


def _get_files(lesion_files, unzip_file, exclude_file, root_dir):
    """Gets image file paths according to given lists"""
    excluded_pids = _get_excluded(exclude_file)
    files, classes = _get_lesion_files(
        lesion_files, excluded_pids["CP"] + excluded_pids["NCP"], root_dir
    )
    with open(unzip_file, "r") as f:
        reader = list(csv.DictReader(f, delimiter=",", quotechar="|"))
        for row in reader:
            if row["label"] == "Normal":
                pid = row["patient_id"]
                sid = row["scan_id"]
                if pid not in excluded_pids["Normal"]:
                    if os.path.exists(os.path.join(root_dir, "Normal", pid, sid)):
                        new_paths = [
                            os.path.join(root_dir, "Normal", pid, sid, x)
                            for x in os.listdir(
                                os.path.join(root_dir, "Normal", pid, sid)
                            )
                        ]
                        files += new_paths
                        classes += [_CLASS_MAP["Normal"] for _ in range(len(new_paths))]
    return files, classes


def _get_excluded(exclude_file):
    """Reads the exclusion list and returns a
    dict of lists of excluded patients"""
    excluded = {"NCP": [], "CP": [], "Normal": []}
    with open(exclude_file, "r") as f:
        for line in f.readlines():
            cls, pid = line.strip("\n").split()
            excluded[cls].append(pid)
    return excluded


# Spline interpolated zoom (SIZ)
def change_depth_siz(volume, desired_depth):
    current_depth = volume.shape[0]
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    volume_new = zoom(volume, (depth_factor, 1, 1), mode="nearest")
    return volume_new


def check_images(volume_path, image_paths):
    # check if volume is 'appropriate'
    if len(image_paths) <= 3:
        return False

    random_image_path_idx = random.randint(0, len(image_paths))
    image = cv2.imread(
        os.path.join(volume_path, image_paths[random_image_path_idx]),
        cv2.IMREAD_GRAYSCALE,
    )
    if image.shape != (512, 512):
        Exception(
            "image {} has wrong dimension: {}".format(
                image_paths[random_image_path_idx], image.shape
            )
        )
        return False

    if image_paths[random_image_path_idx][-3:] != "png":
        Exception(
            "image {} has wrong datatype: {}".format(
                image_paths[random_image_path_idx],
                image_paths[random_image_path_idx][-3:],
            )
        )
        return False

    try:
        image_path = volume_path + "/" + image_paths[len(image_paths) // 2]
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


def load_resize_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # resize image
    try:
        image = exterior_exclusion(image)
    except:
        print(image_path)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    return image


def create_npy_data(args):
    files, _ = process_cncb_data(root_dir=args.root_dir)
    unique_files = list(set(files))
    volume_paths = []
    for path in unique_files:
        root_path = os.path.join(args.root_dir, path)
        if os.path.exists(root_path):
            volume_paths.append(path)

    for path in tqdm(volume_paths):
        # get volume and associated class
        image_paths = [
            os.path.join(args.root_dir, path, x)
            for x in os.listdir(os.path.join(args.root_dir, path))
        ]
        if "NCP" in path:
            y = 1
        elif "CP" in path:
            y = 0
        elif "Normal" in path:
            y = 2
        volume = np.zeros(shape=(len(image_paths), 512, 512))
        for idx, image_path in enumerate(image_paths):
            image = load_resize_image(image_path)
            volume[idx] = image

        cls, patient_id, scan_id = path.split("\\")
        try:
            if not os.path.exists(os.path.join(args.save_path, "data")):
                # Create a new directory because it does not exist
                os.makedirs(os.path.join(args.save_path, "data"))
            np.savez_compressed(
                os.path.join(args.save_path, "data", f"{cls}_{patient_id}_{scan_id}"),
                x=volume,
                y=np.array(y),
            )
        except Exception as e:
            print(e)
            print("could not save image {} of size: {}".format(path, len(volume)))


def convert_images_to_npy(args):
    path_to_folders = args.file_path
    df = pd.read_csv(args.file_names_csv)
    for _, row in df.iterrows():
        path_to_volume_folder = (
            path_to_folders
            + row["label"]
            + "/"
            + str(row["patient_id"])
            + "/"
            + str(row["scan_id"])
        )
        if not os.path.exists(path_to_volume_folder):
            print("Could not find volume folder {}".format(path_to_volume_folder))
            continue
        if os.path.exists(
            args.save_path
            + "volumes/"
            + "vol_"
            + row["label"]
            + "_"
            + str(row["patient_id"])
            + "_"
            + str(row["scan_id"])
            + ".npy"
        ):
            print("Volume already exists {}".format(path_to_volume_folder))
            continue

        # get volume and associated class
        images = [x for x in os.listdir(path_to_volume_folder)]
        print(
            "Patient: {} scan id: {}".format(row["patient_id"], row["scan_id"]),
            end="\r",
        )
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
            image_path = path_to_volume_folder + "/" + images[len(images) // 2]
            image = cv2.imread(image_path)
            unique, counts = np.unique(image, return_counts=True)
            mapColorCounts = dict(zip(unique, counts))
            sum = 0
            for key in mapColorCounts.keys():
                if key != 0:
                    sum += mapColorCounts[key]
            if mapColorCounts[0] >= sum:
                print(
                    "image {} has more black pixel than anything else".format(
                        image_path
                    )
                )
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
            np.save(
                args.save_path
                + "volumes/"
                + "vol_"
                + row["label"]
                + "_"
                + str(row["patient_id"])
                + "_"
                + str(row["scan_id"]),
                volume,
            )
            np.save(
                args.save_path
                + "labels/"
                + "label_"
                + row["label"]
                + "_"
                + str(row["patient_id"])
                + "_"
                + str(row["scan_id"]),
                np.array(y),
            )
        except Exception as e:
            print(e)
            print(
                "could not save image {} of size: {}".format(
                    path_to_volume_folder, len(volume)
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
	Utility script to convert commonvoice into wav and create the training and test json files for speechrecognition. """
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        required=True,
        help="path to the extracted zipfiles",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=True,
        help="path to the save location of the .npy files",
    )
    args = parser.parse_args()

    create_npy_data(args)
