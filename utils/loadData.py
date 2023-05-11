import os
import csv
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import hyperParameters
from utils.custom_datagenerator import SirtaDataGenerator

constants = hyperParameters()

SEED = constants.SEED
BATCH_SIZE = constants.BATCH_SIZE
INPUT_SHAPE = constants.input_shape
ROOT_DATA = constants.data_root


def loadCSVerror(path, rootpath):
    filecsv = open(path)
    csvreader = csv.reader(filecsv)
    rows = []

    header = []
    header = next(csvreader)

    for row in csvreader:
        try:

            impath = "{}/{}".format(rootpath, row[0])

            new_row = {'colorfulness': float(row[1]),
                       'blur': float(row[2]),
                       'y-mean': float(row[3]),
                       'y-variance': float(row[4]),
                       'y-max': float(row[5]),
                       'error': float(row[6]),
                       'impath': impath}

            rows.append(new_row)

        except Exception as e:
            print(e)

    return rows


def loadCSVangle(path, rootpath):

    filecsv = open(path)
    csvreader = csv.reader(filecsv)
    rows = []

    # train_list와 val_list같은 경우 header가 없음.
    # header = []
    # header = next(csvreader)

    for row in csvreader:
        try:
            impath = "{}/{}jpg".format(rootpath, row[0])

            new_row = {'elevation': float(row[1]),
                       'azimuth': float(row[2]),
                       'impath': impath}

            rows.append(new_row)

        except Exception as e:
            print(e)

    return rows


def loadCSVsirta(csvpath, high_imgdir, low_imgdir):
    filecsv = open(csvpath)
    csvreader = csv.reader(filecsv)
    rows = []

    header = []
    header = next(csvreader)

    for row in csvreader:
        try:
            impath_high = "{}/{}.jpg".format(high_imgdir, row[0])
            impath_low = "{}/{}.jpg".format(low_imgdir, row[0])

            if not os.path.isfile(impath_high) or not os.path.isfile(impath_low):
                continue

            new_row = {'elevation': float(row[1]),
                       'azimuth': float(row[2]),
                       'impath_high': impath_high,
                       'impath_low': impath_low}

            rows.append(new_row)

        except Exception as e:
            print(e)

    print("number of rows:", len(rows))

    return rows


def loadSirta(csvpath, high_imgdir, low_imgdir, batch_size, image_size, seed=2, shuffle=False, val_split=0.2):
    all_rows = loadCSVsirta(csvpath, high_imgdir, low_imgdir)
    if shuffle:
        random.seed(seed)
        random.shuffle(all_rows)

    n = len(all_rows)
    test_rows = all_rows[:int(val_split*n)]
    train_rows = all_rows[int(val_split*n):]
    print(f"total dataset:{n}, train:{len(train_rows)}, test:{len(test_rows)}")

    train_generator = SirtaDataGenerator(
        train_rows,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    test_generator = SirtaDataGenerator(
        test_rows,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_generator, test_generator


def imageDataset(datapath, batch_size, val_split = 0.2):
    # ldr image를 load해서 리턴

    subset = "training"
    if val_split == 0:
        subset = None
    train_images = tf.keras.utils.image_dataset_from_directory(
        datapath,
        seed=SEED,
        labels=None,
        class_names=None,
        image_size=INPUT_SHAPE[:2],
        batch_size=batch_size,
        validation_split=val_split,
        subset=subset
    )

    subset = "validation"
    if val_split == 0:
        subset = None
    val_images = tf.keras.utils.image_dataset_from_directory(
        datapath,
        seed=SEED,
        labels=None,
        class_names=None,
        image_size=INPUT_SHAPE[:2],
        batch_size=batch_size,
        validation_split=val_split,
        subset=subset
    )

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_images = train_images.map(lambda x: normalization_layer(x))
    val_images = val_images.map(lambda x: normalization_layer(x))
    return train_images, val_images


def errorBin(train_csvpath, train_datapath):

    # train_rows = loadCSVerror(train_csvpath, train_datapath)
    rows = np.loadtxt(train_csvpath, dtype='str', delimiter=',')
    rows = rows[rows[:, 6].argsort()]

    n = len(rows)
    bin = 5
    split_pos = int(n / 5)
    c_class = 1
    added = []
    for i in range(1, n):
        if(i % split_pos == 0):
            c_class = c_class + 1

        row = rows[i]
        a_row = row.tolist()
        a_row.append(c_class)
        added.append(a_row)
        # index = (n / 5) * (i + 1)
        # print(train_rows[index])

    added = np.array(added)
    np.savetxt("excluded_added.csv", added, delimiter=',', fmt='%s', header="FileName,Colorfulness,Blur,Y-Mean,Y-Variance,Y-Max,Error,class")

def errorDataLoader(csvpath ="train_listed.csv", datapath="ldr_train/", x_col ='impath', y_col ='error', val=True):
    validation_split = 0
    if val:
        validation_split = 0.2

    train_rows = loadCSVerror(csvpath, datapath)
    print("train_rows:", len(train_rows))

    train_df = pd.DataFrame.from_records(train_rows)
    train_df = train_df.sample(frac=1.0, random_state=1)
    train_df = train_df.reset_index(drop=True)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=validation_split
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col=x_col,
        y_col=y_col,
        target_size=INPUT_SHAPE[:2],
        color_mode='rgb',
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        subset='training'
    )

    if not val:
        return train_images

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col=x_col,
        y_col=y_col,
        target_size=INPUT_SHAPE[:2],
        color_mode='rgb',
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        subset='validation'
    )

    return train_images, val_images

def angleDataLoader(excluded=False ,val=True):
    validation_split = 0
    if val:
        validation_split = 0.2

    train_csvpath = os.path.join(ROOT_DATA, "val_listed.csv")
    train_datapath = os.path.join(ROOT_DATA, "ldr_val/")
    if excluded:
        train_datapath = os.path.join(ROOT_DATA, "ldr_val_excluded/")

    test_rows = loadCSVangle(train_csvpath, train_datapath)
    print("test_rows:", len(test_rows))
    test_df = pd.DataFrame.from_records(test_rows)
    # test_df = test_df.sample(frac=1.0,  random_state=1)
    # test_df = test_df.reset_index(drop=True)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=validation_split
    )
    train_images = train_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='impath',
        y_col=['elevation', 'azimuth'],
        target_size=INPUT_SHAPE[:2],
        color_mode='rgb',
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='impath',
        y_col=['elevation', 'azimuth'],
        target_size=INPUT_SHAPE[:2],
        color_mode='rgb',
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        subset='validation'
    )

    return train_images, val_images


