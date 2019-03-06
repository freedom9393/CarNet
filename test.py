import os
import os.path
import random
import numpy as np
import time
import lycon
from keras.models import model_from_json
random.seed(0)
import shutil
import cv2
start_time = time.time()
from functions import Main

SUBSET = ["PUC", "UFPR04", "UFPR05"][1]
TESTING_SUBSET = ["PUC", "UFPR04", "UFPR05"][0]
ROOT_DB = "/media/scott/1056239B56238098/Datasets/PKLot/PKLotSegmented"
ROOT = "/media/scott/cvmdata/Projects/Thesis"
TESTING_FILE = f"{ROOT}/training_files/{TESTING_SUBSET}"
PUC = f"{ROOT_DB}/PUC"
UFPR04 = f"{ROOT_DB}/UFPR04"
UFPR05 = f"{ROOT_DB}/UFPR05"
FAILED_IMAGES_DIR = f"{ROOT}/misclassified_images"
TRAIN_AND_TEST_DIR = f"{ROOT}/training_files/{SUBSET}"
# LOGS_FILE = f"{ROOT}/logs/{SUBSET}"
CHECKPOINT_FILES = f"{ROOT}/checkpoints/{SUBSET}"
# PERCENTAGE_OF_TRAIN_IMAGES = 0.9
SAVED_MODEL = f"{ROOT}/saved_models/{SUBSET}"
# SAVED_MODEL = f"{ROOT}/saved_models/Multiple"
# EPOCHS = 500
# BATCH_SIZE = 64
print("Getting dimension")
height, width, channel = 89, 50, 3


def images_as_string(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            result.append(line)
    return result


def convert_string_to_np_array(filename, img_height, img_width):
    train_data, labels = [], []
    with open(filename, 'r') as n:
        for w, e in enumerate(n):
            pass
    data = np.empty((w+1, height, width, channel), dtype=np.uint8)
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            image = lycon.resize(lycon.load(line.split(" ")[0]), width=img_width, height=img_height,
                                        interpolation=lycon.Interpolation.CUBIC)
            # print(image.shape)
            # zzz = np.rot90(image, 2)
            # print(zzz.shape)
            # exit()
            # data[i, ...] = lycon.resize(lycon.load(line.split(" ")[0]), width=img_width, height=img_height,
            #                             interpolation=lycon.Interpolation.CUBIC)
            data[i, ...] = image
            labels.append(line.split(" ")[1])
            print(i)
    return data, np.array(labels)


print("Reading test.txt")
test_x, test_y = convert_string_to_np_array(f"{TESTING_FILE}/test.txt", height, width)
# print("--- %s seconds ---" % (time.time() - start_time))
print("Preprocess data")
test_x = test_x / 255.0

json_file = open(f"{SAVED_MODEL}/{SUBSET}_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(f"{SAVED_MODEL}/{SUBSET}_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(test_x, test_y, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
