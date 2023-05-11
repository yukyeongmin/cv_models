from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from utils.utils import class_to_startpoint
from utils.utils import normalization_to_pixel

class CustomLearningRateCallback(Callback):
    # cf. book "typing is beliving" 
    # cf. https://littlefoxdiary.tistory.com/87

    '''
    1. 특정 epoch가 지나면 learning rate를 낮춘다.

    2. 특정 epoch가 되면 learning rate가 바뀌도록 한다. # TODO
        schedule은 dictionary
        {'epoch':'바뀔 learning rate'}
        
    3. loss 변화가 적어지면 learning rate를 바꾼다. # TODO 
    '''
    def __init__(self, schedule):
        self.schedule = schedule

    # reduce lr
    def down_lr(self, current_lr, ratio=0.5):
        return current_lr*ratio

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        if(epoch > 1):
            # set reduce point
            if(epoch%50==0):
                current_lr = self.down_lr(current_lr)

                K.set_value(self.model.optimizer.learning_rate, current_lr)
                print('\n Epoch %03d: learning rate changed! %s.' %(epoch+1, current_lr.numpy()))

    def on_epoch_end(self, epoch, logs=None):
        pass 


class CustomPlotCallback(Callback):
    # 참고 https://stackabuse.com/custom-keras-callback-for-saving-prediction-on-each-epoch-with-visualizations/
    def __init__(self, test_images, test_labels, modelname, continue_learning=False, trained_epoch=0):
        self.test_images = test_images # image_generator
        self.test_labels = test_labels # np
        self.modelname = modelname
        self.continue_learning = continue_learning
        self.trained_epoch = trained_epoch

        os.makedirs("./plot_results/{}".format(self.modelname), exist_ok=True)

    def on_train_batch_end(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # get model prediction
        y_pred = np.squeeze(self.model.predict(self.test_images))

        if self.continue_learning:
            epoch = epoch + self.trained_epoch
        
        # plot the predicted theta, phi and save
        fig, ax = plt.subplots(figsize=(8,8))
        gt = ax.scatter(self.test_labels[:,1], self.test_labels[:,0],s=5)
        pred = ax.scatter(y_pred[:,1], y_pred[:,0],s=5)
        ax.legend((gt, pred),('ground truth', 'pred'))
        plt.title(f'Prediction Visualization Keras Callback - Epoch: {epoch+1}')
        plt.savefig('./save/{}/plot_results/epoch_{}.png'.format(self.modelname, epoch+1))
        plt.xlabel('altitude')
        plt.ylabel('azimuth')
        plt.close()

class CustomPlotClassCallback(Callback):
    def __init__(self, test_images, image_shape, modelname, patch_size, continue_learning=False, trained_epoch=0):
        self.test_images = test_images
        self.image_shape = image_shape #(height, width, channel)
        self.modelname = modelname
        self.patch_size = patch_size
        self.continue_learning = continue_learning
        self.trained_epoch = trained_epoch

        os.makedirs("./save/{}/plot_results".format(self.modelname), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.continue_learning:
            epoch = epoch + self.trained_epoch

        fig, ax = plt.subplots(figsize=(8,4))
        for i in range(9):
            images = next(self.test_images) # (image(batch, height, width, channel), label)
            test_labels = images[1]
            test_labels = normalization_to_pixel(test_labels)
            test_images = images[0]
            y_pred = self.model.predict(test_images) # (batch, patch_num)

            ax = plt.subplot(3,3, i+1)
            plt.imshow(test_images[i])
            plt.scatter(test_labels[i,0],test_labels[i,1],s=5,c='red')
            start_point, label = class_to_startpoint(self.image_shape,self.patch_size,y_pred[i])
            rect = patches.Rectangle(start_point,self.patch_size, self.patch_size, linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)
            plt.title("predicted patch: {}".format(label))
            plt.axis("off")
        plt.savefig("./save/{}/plot_results/epoch_{}.png".format(self.modelname, epoch+1))

class CustomCheckpointCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor, save_best_only, save_weights_only, mode, verbose, continue_learning=False, trained_epoch=0):
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
            epoch = epoch+self.trained_epoch
        save_path = self.filepath+"/epoch_{}".format(epoch+1)
        self.save_weights(save_path)