import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from datetime import datetime

sys.path.append("..")
import tensorflow as tf
from utils.loadData import errorDataLoader, loadSirta, loadCSVsirta
from gan import GAN
from train import Train


def plot_reconstructed(model, images, suptitle, save_dir):
    if model is None:
        reconstructed_images = images
    else:
        reconstructed_images = model.reconstruct(images=images, type='high')
        generated_images = model.generate_high2low(images=images)

        plt.figure(figsize=(16, 8))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(generated_images[i])
            plt.title(f"{i + 1}th image,\nm:{np.min(generated_images[i])}, \nM:{np.max(reconstructed_images[i])}")
        plt.suptitle(suptitle)

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{suptitle}_gen.png")
        plt.close()

        plt.figure(figsize=(16, 8))
        plt.imshow(generated_images[0])
        plt.title(f"1st image,\nm:{np.min(generated_images[0])}, \nM:{np.max(generated_images[0])}")
        plt.savefig(f"{save_dir}/{suptitle}_gen_bigger.png")
        plt.close()

    plt.figure(figsize=(16, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(reconstructed_images[i])
        plt.title(f"{i + 1}th image,\nm:{np.min(reconstructed_images[i])}, \nM:{np.max(reconstructed_images[i])}")
    plt.suptitle(suptitle)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{suptitle}_reconst.png")
    plt.close()

    plt.figure(figsize=(16, 8))
    plt.imshow(reconstructed_images[0])
    plt.title(f"1st image,\nm:{np.min(reconstructed_images[0])}, \nM:{np.max(reconstructed_images[0])}")
    plt.savefig(f"{save_dir}/{suptitle}_reconst_bigger.png")
    plt.close()


def scheduler(self, epoch, lr):
    if epoch != 0 and epoch % 5 == 0 or epoch > 20:
        return lr * tf.math.exp(-0.1)
    else:
        return lr


class CustomCheckpointCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor, save_best_only, save_weights_only, mode, verbose, continue_learning=False,
                 trained_epoch=0):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.verbose = verbose
        self.continue_learning = continue_learning
        self.trained_epoch = trained_epoch
        super().__init__(
            filepath=self.filepath,
            moniter=self.monitor,
            save_best_only=self.save_best_only,
            save_weights_only=self.save_weights_only,
            mode=self.mode,
            verbose=self.verbose
        )

    def on_epoch_end(self, epoch, logs=None):
        if self.continue_learning:
            epoch = epoch + self.trained_epoch
        save_path = self.filepath + "/{}".format(epoch + 1)
        self.model.save_weights(save_path)


class CustomPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_images, save_dir, continue_learning=False, trained_epoch=0):
        self.test_images = test_images  # image_generator
        self.save_path = f"{save_dir}/plot"
        self.continue_learning = continue_learning
        self.trained_epoch = trained_epoch

        os.makedirs(self.save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.continue_learning:
            epoch = epoch + self.trained_epoch

        plot_reconstructed(
            self.model,
            self.test_images,
            suptitle=f'Prediction Visualization Keras Callback - Epoch: {epoch + 1}',
            save_dir=self.save_path
        )


def preprocessing(img, image_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[1], image_size[0]), cv2.INTER_CUBIC)
    return img / 255.


def yield_sirta(
        csvpath="/home/cvnar1/Desktop/Data/05.SIRTA/metafiles/20230314163052_labels_norm.csv",
        high_imgdir="/home/cvnar1/Desktop/Data/05.SIRTA/2017_single_equi_final_01_buffer",
        low_imgdir="/home/cvnar1/Desktop/Data/05.SIRTA/2017_single_equi_final_03_buffer",
):

    with open(csvpath) as f:
        reader = csv.reader(f)
        for row in reader:
            impath_high = "{}/{}.jpg".format(high_imgdir, row[0])
            impath_low = "{}/{}.jpg".format(low_imgdir, row[0])
            elevation = float(row[1])
            azimuth = float(row[2])

            if not os.path.isfile(impath_high) or not os.path.isfile(impath_low):
                continue

            high_exposure = cv2.imread(impath_high)
            low_exposure = cv2.imread(impath_low)
            infos = [elevation, azimuth]
            image_size = high_exposure.shape

            high_exposure = preprocessing(high_exposure, image_size)
            low_exposure = preprocessing(low_exposure, image_size)

            yield [high_exposure, low_exposure], infos


if __name__ == '__main__':

    strategy = tf.distribute.MirroredStrategy()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    print('Number of divices: {}'.format(strategy.num_replicas_in_sync))

    IMG_SIZE = (256, 1024, 3)
    BATCH_SIZE = 32
    EPOCHS = 100
    learning_rate = 0.001
    tf_function = True
    SEED = 2

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    reconstruction_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    metrics = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

    train_mode = 'custom_loop'
    continue_learning = False
    trained_epoch = 0

    save_name = "GAN_conv"
    TYPE = "conv"
    filters = [8, 16, 16, 16, 32, 32, 32]
    num_decoder = 1
    current_time = str(datetime.now())
    save_dir = f"./save/{save_name}/{current_time}"

    train_generator, test_generator = loadSirta(
        csvpath="/home/cvnar1/Desktop/Data/05.SIRTA/metafiles/20230314163052_labels_norm.csv",
        high_imgdir="/home/cvnar1/Desktop/Data/05.SIRTA/2017_single_equi_final_01_buffer",
        low_imgdir="/home/cvnar1/Desktop/Data/05.SIRTA/2017_single_equi_final_03_buffer",
        batch_size=1,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED,
        val_split=0.2)

    train_dataset = tf.data.Dataset.from_generator(
        generator=train_generator,
        output_signature=(tf.TensorSpec(shape=(256, 1024, 3), dtype=tf.float32),
                          tf.TensorSpec(shape=(256, 1024, 3), dtype=tf.float32))
    )
    test_dataset = tf.data.Dataset.from_generator(
        generator=test_generator,
        output_signature=(tf.TensorSpec(shape=(256, 1024, 3), dtype=tf.float32),
                          tf.TensorSpec(shape=(256, 1024, 3), dtype=tf.float32))
    )

    train_dataset = train_dataset.with_options(options).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.with_options(options).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_dist_ds = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_ds = strategy.experimental_distribute_dataset(test_dataset)

    with strategy.scope():
        model = GAN(IMG_SIZE, filters=filters, type=TYPE, num_decoder=num_decoder)

    model.compile(
        optimizer=optimizer,
        reconstruction_loss=reconstruction_loss,
        metrics=metrics,
    )
    model.build(input_shape=(BATCH_SIZE, 256, 1024, 3))
    # model.summary()

    print('Check Before Training...')
    high_images, low_images = next(iter(test_dataset))
    # model.evaluate(test_ds, verbose=1)
    plot_reconstructed(None, low_images, "Low exposure images", save_dir)
    plot_reconstructed(None, high_images, "Input high exposure images", save_dir)
    plot_reconstructed(model, high_images, "Before training", save_dir)

    checkpoint_filepath = f"{save_dir}/checkpoint"
    checkpoint_callback = CustomCheckpointCallback(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True,
        mode='min',
        verbose=1,
        continue_learning=continue_learning,
        trained_epoch=trained_epoch,
    )

    log_dir = "./save/{}/{}/tensorboard/".format(save_name, current_time)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(
        schedule=scheduler,
        verbose=1
    )

    plotting_callback = CustomPlotCallback(
        high_images,
        save_dir=save_dir,
        continue_learning=continue_learning,
        trained_epoch=trained_epoch)

    _callbacks = [checkpoint_callback, tensorboard_callback, plotting_callback]
    callbacks = tf.keras.callbacks.CallbackList(
        _callbacks, add_history=False, model=model
    )

    trainer = Train(epochs=EPOCHS,
                    model=model,
                    batch_size=BATCH_SIZE,
                    strategy=strategy,
                    enable_function=tf_function,
                    optimizer=optimizer,
                    )

    try:
        print('Training...')
        if train_mode == 'custom_loop':
            trainer.custom_loop(train_dist_ds,
                                test_dist_ds,
                                strategy,
                                callbacks)
        elif train_mode == 'keras_fit':
            raise ValueError(
                '`tf.distribute.Strategy` does not support subclassed models yet.')
        else:
            raise ValueError(
                'Please enter either "keras_fit" or "custom_loop" as the argument.')

    except KeyboardInterrupt:
        print('[Info] End training due to keyboard interruption')
        plot_reconstructed(model, high_images, "After training", save_dir)
    print('Evaluation result:', model.evaluate(test_generator))
