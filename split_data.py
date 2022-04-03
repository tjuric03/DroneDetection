import pandas as pd
import numpy as np
from glob import glob
import os
import shutil
from sklearn.model_selection import train_test_split


DATA_PATH = "./Data"
TEST_SIZE = 0.1

# Stratified splitting of the data
def split_data():
    images = pd.DataFrame(glob(f"{DATA_PATH}/*/*.JPEG"),columns=["image_path"])
    images["image_path"] = images["image_path"].str.replace("\\","/")
    images["type"] = images.apply(lambda row: row[0].split("/")[-2],axis=1)

    train, test = train_test_split(images,test_size=TEST_SIZE,stratify=images["type"])

    train["image_path"].to_csv(f"{DATA_PATH}/train.txt",index=False,header=False)
    test["image_path"].to_csv(f"{DATA_PATH}/test.txt",index=False,header=False)


# Type can be either "train" or "test"
def fill_folder(type="train"):
    paths = np.genfromtxt(f"{DATA_PATH}/{type}.txt", delimiter="\n",dtype=np.dtype(str))

    destination_path = f"{DATA_PATH}/{type.capitalize()}"
    os.mkdir(destination_path)

    for image_path in paths:
        xml_path = os.path.dirname(image_path) + "/XML/" + os.path.basename(image_path).split(".")[0] + ".xml"
        shutil.copy2(image_path, destination_path)
        shutil.copy2(xml_path, destination_path)

def create_train_test_folders():

    if os.path.exists(f"{DATA_PATH}/Train"):
        print("Skipping... Train folder already exists")
    else:
        fill_folder("train")

    if os.path.exists(f"{DATA_PATH}/Test"):
        print("Skipping... Test folder already exists")
    else:
        fill_folder("test")

if __name__ == "__main__":
    if os.path.exists(f"{DATA_PATH}/train.txt") or os.path.exists(f"{DATA_PATH}/test.txt"):
        print("Skipping... Data has already been split to train and test set!")
    else:
        split_data()

    create_train_test_folders()