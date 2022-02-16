import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, Lambda
from keras.models import Model, Sequential


def LRN(image):
    import tensorflow as tf
    
    return tf.nn.local_response_normalization(image)
    
    
    
class AlexNet(Model):
    def __init__(self):
        super(AlexNet,self).__init__()
        
        self.preprocessing = keras.layers.experimental.preprocessing.Resizing(227,227, interpolation="bilinear", input_shape=(32,32,3))
        self.conv1 = Sequential([Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding="valid"),
                                Lambda(LRN),
                                Activation("relu"),
                                MaxPool2D(pool_size=(3,3), strides=(2,2))])
        self.conv2 = Sequential([Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same"),
                                Lambda(LRN),
                                Activation("relu"),
                                MaxPool2D(pool_size=(3,3), strides=(2,2))])
        self.conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.conv4 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.conv5 = Sequential([Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
                                 MaxPool2D(pool_size=(3,3), strides=(2,2))])
        
        self.dense1 = Sequential([Flatten(),
                                  Dense(2048, activation='relu'),
                                  Dropout(0.5)])
        self.dense2 = Sequential([Dense(2048, activation='relu'),
                                  Dropout(0.5)])
        self.dense3 = Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.preprocessing(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x

            
