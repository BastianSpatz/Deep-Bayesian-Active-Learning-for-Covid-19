import os
import cv2
import argparse
import json
import numpy as np    
from PIL import Image 
from skimage import io
from sklearn.model_selection import train_test_split
from utils import create_annotation_info
import imageio.core.util

def ignore_warnings(*args, **kwargs):
    pass

def convert(args):
    info = {"year" : "2023",
                     "version" : "1.0",
                     "description" : "CT Lung Scans",
                     "contributor" : "Bastian Spatz",
                     "url" : "http://cocodataset.org",
                     "date_created" : "2023"
                    }
    licenses = [{"id": 1,
                        "name": "Attribution-NonCommercial",
                        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                        }]
    categories = [{"id": 1, "name": "lung", "supercategory": "lung"}]
    images_radiopedia = np.load(os.path.join(args.npy_file_location, 'images_radiopedia.npy'))
    masks_radiopedia = np.load(os.path.join(args.npy_file_location, 'masks_radiopedia.npy'))
    images_medseg = np.load(os.path.join(args.npy_file_location, 'images_medseg.npy'))
    masks_medseg = np.load(os.path.join(args.npy_file_location, 'masks_medseg.npy'))
    images_ct_lesion = np.load(os.path.join(args.npy_file_location, 'images_ct_lesion_seg.npy'))
    masks_ct_lesion = np.load(os.path.join(args.npy_file_location, 'masks_ct_lesion_seg.npy'))
    # test_images_medseg = np.load(os.path.join(args.file_path, 'test_images_medseg.npy')).astype(np.float32)

    med_seg_images = np.concatenate((images_radiopedia, images_medseg), axis=0)
    med_seg_images = np.delete(med_seg_images, 155, axis=0)
    med_seg_masks = np.concatenate((masks_radiopedia, masks_medseg), axis=0)
    med_seg_masks = np.delete(med_seg_masks, 155, axis=0)
    med_seg_masks = 1 - med_seg_masks[:, :, :, 3]
    print(med_seg_masks.shape)
    print(masks_ct_lesion.shape)
    print(med_seg_images.shape)
    print(np.expand_dims(images_ct_lesion, axis=-1).shape)
    masks = np.concatenate((med_seg_masks, masks_ct_lesion), axis=0)
    images = np.concatenate((med_seg_images, np.expand_dims(images_ct_lesion, axis=-1)), axis=0)

    lenght, width, height = masks.shape

    # width = int(width/2)
    # height = int(height/2)

    train_idx, test_idx = train_test_split(range(lenght), test_size=0.1)
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5)

    annotations_train = []
    annotations_test = []
    annotations_val = []
    images_train = []
    images_test = []
    images_val = []
    dirs = ["train", "test", "val"]
    sub_dirs = ["images", "masks"]
    print("removing existing images")
    # This looks like hot garbage; FIX IT
    for dir in dirs:
        if not os.path.exists(os.path.join(args.save_file_loaction, "segmentation", dir)):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(args.save_file_loaction, "segmentation", dir))
        for sub_dir in sub_dirs:
            if not os.path.exists(os.path.join(args.save_file_loaction, "segmentation", dir, sub_dir)):
                # Create a new directory because it does not exist
                os.makedirs(os.path.join(args.save_file_loaction, "segmentation", dir, sub_dir))

    for dir in dirs:
        for sub_dir in sub_dirs:
            for f in os.listdir(args.save_file_loaction + dir + "/" + sub_dir):
                os.remove(os.path.join(args.save_file_loaction + dir + "/" + sub_dir, f))

    print("all images removed")

    for idx in range(lenght):
        if np.all(masks[idx] == 0):
                continue
        print("converting file " + str(idx) + "/" + str(lenght), end="\r")
        file_name = str(idx).zfill(len(str(lenght))) + ".png"

        image = images[idx][:, :, 0]
        mask = masks[idx]
        annotation = create_annotation_info(1, 1, 1, mask)
        if idx in train_idx:
            images_train.append({"date_captured" : "2022",
                            "file_name" : args.save_file_loaction + "segmentation_data/train/images/image_" + file_name, # remove "/"
                            "id" : idx,
                            "license" : 1,
                            "url" : "",
                            "height" : height,
                            "width" : width})
            annotations_train.append({"segmentation" : annotation["segmentation"],
                                "area" : annotation["area"],
                                "iscrowd" : 0,
                                "image_id" : idx,
                                "bbox" : annotation["bbox"],
                                "category_id" : 1,
                                "id": idx})
            cv2.imwrite(args.save_file_loaction + "segmentation_data/train/images/image_" + file_name, image)
            cv2.imwrite(args.save_file_loaction + "segmentation_data/train/masks/mask_" + file_name, mask)
        elif idx in test_idx:
            images_test.append({"date_captured" : "2022",
                            "file_name" : args.save_file_loaction + "segmentation_data/test/images/image_" + file_name, # remove "/"
                            "id" : idx,
                            "license" : 1,
                            "url" : "",
                            "height" : height,
                            "width" : width})
            annotations_test.append({"segmentation" : annotation["segmentation"],
                                "area" : annotation["area"],
                                "iscrowd" : 0,
                                "image_id" : idx,
                                "bbox" : annotation["bbox"],
                                "category_id" : 1,
                                "id": idx})
            cv2.imwrite(args.save_file_loaction + "segmentation_data/test/images/image_" + file_name, image)
            cv2.imwrite(args.save_file_loaction  + "segmentation_data/test/masks/mask_" + file_name, mask)
        elif idx in val_idx:
            images_val.append({"date_captured" : "2022",
                            "file_name" : args.save_file_location + "segmentation_data/val/images/image_" + file_name, # remove "/"
                            "id" : idx,
                            "license" : 1,
                            "url" : "",
                            "height" : height,
                            "width" : width})
            annotations_val.append({"segmentation" : annotation["segmentation"],
                                "area" : annotation["area"],
                                "iscrowd" : 0,
                                "image_id" : idx,
                                "bbox" : annotation["bbox"],
                                "category_id" : 1,
                                "id": idx})
            cv2.imwrite(args.save_file_location + "segmentation_data/val/images/image_" + file_name, image)
            cv2.imwrite(args.save_file_location + "segmentation_data/val/masks/mask_" + file_name, mask)

    
    json_data_train = {"info" : info,
                    "images" : images_train,
                    "licenses" : licenses,
                    "annotations" : annotations_train,
                    "categories" : categories}
    json_data_test = {"info" : info,
                    "images" : images_test,
                    "licenses" : licenses,
                    "annotations" : annotations_test,
                    "categories" : categories}
    json_data_val = {"info" : info,
                    "images" : images_val,
                    "licenses" : licenses,
                    "annotations" : annotations_val,
                    "categories" : categories}

    print("creating JSON's")
    with open(os.path.join(args.save_file_location, "segmentation_data/train.json"), "w") as jsonfile:
        json.dump(json_data_train, jsonfile, sort_keys=True, indent=4)
    with open(os.path.join(args.save_file_location, "segmentation_data/test.json"), "w") as jsonfile:
        json.dump(json_data_test, jsonfile, sort_keys=True, indent=4)
    with open(os.path.join(args.save_file_location, "segmentation_data/val.json"), "w") as jsonfile:
        json.dump(json_data_val, jsonfile, sort_keys=True, indent=4)
    print("Done!")


def create_coco_entry():
    pass
def npy_to_coco_file(npy_file_location, save_coco_file_location):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    Utility script to convert .npy segmentation data to the coco format. """
    )
    parser.add_argument('--npy_file_location', type=str, default=None, required=True,
                        help='path to the .npy files for segmentation')  
    parser.add_argument('--save_file_location', type=str, default=None, required=True,
                        help='path to the .npy files for segmentation')                 

    convert()