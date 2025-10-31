import json

import os

def prepare_json():
    with open(
        "/Datasets/BDD100k_Anomaly/validation_list.json", "r"
    ) as json_file:
        val_list = json.load(json_file)

    val_list = [i.split("/")[-1].split(".")[0] for i in val_list]
    image_dir = "/Datasets/BDD100k_Anomaly/images/10k/val"
    target_dir = (
        "/Datasets/BDD100k_Anomaly/labels/pan_seg/anom_bitmasks/train"
    )
    # take all elements in image dir that are not in val_list and put them into a json file anom_files_val_list.json

    os.chdir("/Datasets/BDD100k_Anomaly/images/10k/val")
    val_images = os.listdir()
    val_anom_list = [
        i.split(".")[0] for i in val_images if i.split(".")[0] not in val_list
    ]
    with open(
        "/Datasets/BDD100k_Anomaly/anom_files_val_list.json", "w"
    ) as json_file:
        json.dump(val_anom_list, json_file)

if __name__ == "__main__":
    import json
    prepare_json()