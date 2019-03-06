import os
import random
import json
import lycon
import numpy as np


SUBSET = ["PUC", "UFPR04", "UFPR05"][2]
ROOT_DB = "/media/scott/1056239B56238098/Datasets/PKLot/PKLotSegmented"
ROOT = "/media/scott/cvmdata/Projects/Thesis"
ROOT_SUBSET = f"{ROOT_DB}/{SUBSET}"
TXT_FILES = f"{ROOT}/training_files/{SUBSET}"


class Main(object):
    def __init__(self,
                 subset,
                 root=ROOT,
                 root_db=ROOT_DB,
                 root_subset=ROOT_SUBSET,
                 all_txt=f"{TXT_FILES}/all.txt",
                 train_txt=f"{TXT_FILES}/train.txt",
                 test_txt=f"{TXT_FILES}/test.txt",
                 cfg_json=f"{TXT_FILES}/cfg.json",
                 d_set_division=0.9):
        self.subset = subset
        self.root = root
        self.root_db = root_db
        self.root_subset = root_subset
        self.all_txt = all_txt
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.cfg_json = cfg_json
        self.d_set_division = d_set_division
        self.all_images_as_string = None
        self.create_folder(TXT_FILES)

        if not os.path.isfile(self.all_txt) or not os.path.isfile(self.train_txt) or not os.path.isfile(self.test_txt) \
                or not os.path.isfile(self.cfg_json):
            self.create_files()

    # Create folder if not exists
    @staticmethod
    def create_folder(folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    # Read all image names from folders
    def read_img_names_from_dir(self):
        images = []
        for root, dirs, files in os.walk(self.root_subset):
            for name in files:
                if name.endswith(".jpg"):
                    images.append(root + "/" + name + " 0") if os.path.basename(root) == "Empty" else images.append(root + "/" + name + " 1")
        self.all_images_as_string = images

    # Read images from txt file
    def read_img_names_from_txt(self):
        result = []
        with open(self.all_txt, 'r') as f:
            for line in f:
                result.append(line.split(" ")[0])
        self.all_images_as_string = result

    # Create txt file from strings of images
    @staticmethod
    def create_txt(arr, filename):
        if os.path.isfile(filename):
            open(filename, 'w').close()

        for rows in arr:
            with open(filename, "a+") as f:
                f.write(rows+"\n")

    # Create config file
    def create_cfg(self):
        img_height, img_width = self.get_average_dimensions_of_images()
        if os.path.isfile(self.cfg_json):
            open(self.cfg_json, 'w').close()

        with open(self.cfg_json, 'a+') as json_file:
            data = {
                'input_shape': {
                    'height': img_height,
                    'width': img_width,
                }
            }
            json.dump(data, json_file)

    # Define average dimension of all images
    def get_average_dimensions_of_images(self):
        heights, widths = [], []
        a = 0
        with open(self.all_txt, 'r') as f:
            for line in f:
                img = lycon.load(line.split(" ")[0])
                heights.append(img.shape[0])
                widths.append(img.shape[1])
                a += 1
                print(a)
        return int(sum(heights) / len(heights)), int(sum(widths) / len(widths))

    # Read config file
    def read_from_json(self):
        with open(self.cfg_json) as json_file:
            data = json.load(json_file)
        return data

    # Get input_shape of image
    def get_input_shape(self):
        conf = self.read_from_json()
        return conf['input_shape']['height'], conf['input_shape']['width'], conf['input_shape']['channels']

    # Split string of images for training and testing
    def split_images(self):
        random.shuffle(self.all_images_as_string)
        length = int(len(self.all_images_as_string) * self.d_set_division)
        train_data = self.all_images_as_string[:length]
        test_data = self.all_images_as_string[length:]
        return train_data, test_data

    # Create all necessary txt files
    def create_files(self):
        train, test = self.split_images()
        self.create_txt(self.all_images_as_string, self.all_txt)
        print("All.txt is created")
        self.create_txt(train, self.train_txt)
        print("Train.txt is created")
        self.create_txt(test, self.test_txt)
        print("Test.txt is created")
        self.create_cfg()
        print("cfg.json is created")

    def convert_string_to_np_array(self, attr):
        types = ['train', 'test']
        if attr not in types:
            raise ValueError("Invalid input. Expected one of: %s" % types)
        filename = self.train_txt if attr is 'train' else self.test_txt
        train_data, labels = [], []
        height, width, channels = self.get_input_shape()
        with open(filename, 'r') as f:
            for line in f:
                image = lycon.resize(lycon.load(line.split(" ")[0]), width=width, height=height,
                                     interpolation=lycon.Interpolation.CUBIC)
                train_data.append(image)
                labels.append(line.split(" ")[1])
        return np.array(train_data), np.array(labels)


pklot = Main(SUBSET)
train_x, train_y = pklot.convert_string_to_np_array('train')
