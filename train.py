import os
import os.path
import random
import numpy as np
import lycon
from keras.models import Sequential
# from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
random.seed(0)

SUBSET = "UFPR05"
ROOT_DB = "/media/scott/1056239B56238098/Datasets/PKLot/PKLotSegmented"
ROOT = "/media/scott/cvmdata/Projects/Thesis"
ROOT_SUBSET = f"{ROOT_DB}/{SUBSET}"
TRAIN_AND_TEST_DIR = f"{ROOT}/training_files/{SUBSET}"
LOGS_FILE = f"./logs/{SUBSET}"
CHECKPOINT_FILES = f"/media/scott/cvmdata/checkpoints/Thesis/{SUBSET}"
PERCENTAGE_OF_TRAIN_IMAGES = 0.9
SAVED_MODEL = f"{ROOT}/saved_models/{SUBSET}"
EPOCHS = 500
BATCH_SIZE = 32


def image_names_to_string(root_folder):
    images = []
    for root, dirs, files in os.walk(root_folder):
        for name in files:
            if name.endswith(".jpg"):
                if os.path.basename(root) == "Empty":
                    images.append(root+"/"+name+" 0")
                else:
                    images.append(root + "/" + name+" 1")
    return images


def create_txt(arr, filename):
    if os.path.isfile(filename):
        open(filename, 'w').close()

    for rows in arr:
        with open(filename, "a+") as f:
            f.write(rows+"\n")


def split_data_for_train_and_test(arr, split_percentage=PERCENTAGE_OF_TRAIN_IMAGES):
    random.shuffle(arr)
    length = int(len(arr) * split_percentage)
    train_data = arr[:length]
    test_data = arr[length:]
    return train_data, test_data


def get_average_dimensions_of_images(filename):
    heights, widths = [], []
    a = 0
    with open(filename, 'r') as f:
        for line in f:
            img = lycon.load(line.split(" ")[0])
            heights.append(img.shape[0])
            widths.append(img.shape[1])
            a += 1
            print(a)
    return int(sum(heights) / len(heights)), int(sum(widths) / len(widths))


def convert_string_to_np_array(filename, img_height, img_width):
    train_data, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            image = lycon.resize(lycon.load(line.split(" ")[0]), width=img_width, height=img_height,
                                 interpolation=lycon.Interpolation.CUBIC)
            train_data.append(image)
            labels.append(line.split(" ")[1])
    return np.array(train_data), np.array(labels)


# Create files for train/test
def create_files():
    zet = image_names_to_string(ROOT_SUBSET)
    print("Images converted to string")
    if not os.path.exists(TRAIN_AND_TEST_DIR):
        os.makedirs(TRAIN_AND_TEST_DIR)
    create_txt(zet, f"{TRAIN_AND_TEST_DIR}/all.txt")
    train, test = split_data_for_train_and_test(zet)
    print("Data is split to train and test")
    create_txt(train, f"{TRAIN_AND_TEST_DIR}/train.txt")
    print("Train file is created")
    create_txt(test, f"{TRAIN_AND_TEST_DIR}/test.txt")
    print("Test file is created")


# Determine input shape
def get_dimension():
    img_height, img_width = get_average_dimensions_of_images(f"{TRAIN_AND_TEST_DIR}/all.txt")
    return img_height, img_width, 3


print("Getting dimension")
height, width, channel = get_dimension()
# height, width, channel = (89, 50, 3)
print("Reading train.txt")
train_x, train_y = convert_string_to_np_array(f"{TRAIN_AND_TEST_DIR}/train.txt", height, width)
print("Reading test.txt")
test_x, test_y = convert_string_to_np_array(f"{TRAIN_AND_TEST_DIR}/test.txt", height, width)
print("Preprocessing data")
train_x = train_x / 255.0
test_x = test_x / 255.0
print("train_x shape: ", train_x.shape)
print("train_y shape: ", train_y.shape)
print("test_x shape: ", test_x.shape)
print("test_y shape: ", test_y.shape)


model = Sequential()
model.add(Conv2D(96, (11, 11,), padding='valid', strides=(1, 1), dilation_rate=(2, 2),
                 input_shape=(height, width, channel), name='conv1'))
model.add(Activation('relu', name='relu1'))
model.add(MaxPooling2D((2, 2), strides=(3, 3), padding='same', name='pool1'))

model.add(Conv2D(192, (11, 11), padding='same', name='conv2', strides=(1, 1), dilation_rate=(2, 2),))
model.add(Activation('relu', name='relu2'))
model.add(MaxPooling2D((2, 2), padding='same', name='pool2'))

model.add(Conv2D(384, (11, 11), padding='same', name='conv3', strides=(1, 1), dilation_rate=(2, 2),))
model.add(Activation('relu', name='relu3'))
model.add(MaxPooling2D((2, 2), padding='same', name='pool3'))

model.add(Flatten())
model.add(Dropout(0.8, name='dropout6'))
model.add(Dense(4096, activation='relu', name='fc6'))
model.add(Dropout(0.8, name='dropout7'))
model.add(Dense(4096, activation='relu', name='fc7'))
model.add(Dropout(0.8, name='dropout8'))
model.add(Dense(1, activation='sigmoid', name='predictions'))

model.summary()

learning_rate = 0.00001
weight_decay = 0.0005
nesterov = True
momentum = 0.99

sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=nesterov)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
if not os.path.exists(CHECKPOINT_FILES):
    os.makedirs(CHECKPOINT_FILES)
checkpoint = ModelCheckpoint(CHECKPOINT_FILES+"/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5",
                             verbose=1, monitor='val_loss', save_best_only=False)

if not os.path.exists(LOGS_FILE):
    os.makedirs(LOGS_FILE)
tensorboard = TensorBoard(log_dir=LOGS_FILE, histogram_freq=0, write_graph=True, write_images=False)

model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[checkpoint, tensorboard],
          validation_split=0.2)
scores = model.evaluate(train_x, train_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model_json = model.to_json()
if not os.path.exists(SAVED_MODEL):
    os.makedirs(SAVED_MODEL)
with open(f"{SAVED_MODEL}/{SUBSET}_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(f"{SAVED_MODEL}/{SUBSET}_model.h5")
print("Saved model to disk")
