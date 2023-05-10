import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, Dense, ZeroPadding2D, AveragePooling2D, Flatten, Concatenate 

class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()

        self. mapping = [[0,1,2],
                        [1,2,3],
                        [2,3,4],
                        [3,4,5],
                        [4,5,0],
                        [5,0,1],
                        [0,1,2,3],
                        [1,2,3,4],
                        [2,3,4,5],
                        [3,4,5,0],
                        [4,5,0,1],
                        [5,0,1,2],
                        [0,1,3,4],
                        [1,2,4,5],
                        [0,2,4,5],
                        [0,1,2,3,4,5]]

        # feature extractor 
        self.zero_padding = ZeroPadding2D(padding=2)
        self.conv1 = Conv2D(filters=6, kernel_size=5, padding="valid", strides=1, activation="tanh")
        self.pool2 = AveragePooling2D(pool_size=2, strides=2)
        self.conv3 = mapping_layer(self.mapping) 
        self.pool4 = AveragePooling2D(pool_size=2, strides=2)


        # classifier
        self.flatten = Flatten()
        self.dense1 = Dense(units=120, activation="tanh")
        self.dense2 = Dense(units=84, activation="tanh")
        self.dense3 = Dense(units=10, activation="softmax")

    def call(self, x):
        x = self.zero_padding(x)
        x = self.conv1(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool4(x)
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class trainable_pooling(Layer):
  def __init__(self):
    super(trainable_pooling, self).__init__()
    self.pool = AveragePooling2D(pool_size=2, strides=2)
    
  def build(self, input_shape):
    init = tf.random_normal_initializer()
    self.W = tf.Variable(name = "coefficient",
                            initial_value = init(shape = (input_shape[-1],)), 
                            dtype = "float32",
                            trainable = "True")
    self.b = tf.Variable(name = "bias",
                            initial_value = init(shape = (input_shape[-1],)),
                            dtype = "float32",
                            trainable = "True")
    
  def call(self, x):
    x = self.pool(x)
    return x*self.W+self.b
    

class mapping_layer(Layer):
     
  def __init__(self, mapping):
    super(mapping_layer, self).__init__()

    self.mapping = [] # tf.gather_nd를 사용하기 위해 형태 맞춤
    for map in mapping:
        temp = []
        for m in map:
            temp.append([m])
        self.mapping.append(temp)

    self.conv0 = Conv2D(filters=1, kernel_size=(5,5), strides=(1,1), padding='valid', activation='tanh')
    self.conv1 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv2 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv3 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv4 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv5 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv6 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv7 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv8 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv9 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv10 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv11 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv12 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv13 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv14 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    self.conv15 = Conv2D(filters=1, kernel_size=(5,5), strides=1, padding='valid', activation='tanh')
    
    self.concat = Concatenate(axis = -1)
    

  def call(self, input): # (128, 14, 14, 6)
    input = tf.transpose(input, perm=[3,0,1,2]) # (6, 128, 14, 14)
    
    # for index, map in enumerate(self.mapping):
    #     exec(name+"_"+str(index)+"= tf.gather_nd(input, map)", globals()) # (3, 128, 14, 14)
    #     s = "temp = self.conv"+str(index)+"(col_"+str(index)+")"
    #     exec(s)
    
    #     output = self.concat([output, temp])
    
    col_0 = tf.gather_nd(input, self.mapping[0])
    col_1 = tf.gather_nd(input, self.mapping[1])
    col_2 = tf.gather_nd(input, self.mapping[2])
    col_3 = tf.gather_nd(input, self.mapping[3])
    col_4 = tf.gather_nd(input, self.mapping[4])
    col_5 = tf.gather_nd(input, self.mapping[5])
    col_6 = tf.gather_nd(input, self.mapping[6])
    col_7 = tf.gather_nd(input, self.mapping[7])
    col_8 = tf.gather_nd(input, self.mapping[8])
    col_9 = tf.gather_nd(input, self.mapping[9])
    col_10 = tf.gather_nd(input, self.mapping[10])
    col_11 = tf.gather_nd(input, self.mapping[11])
    col_12 = tf.gather_nd(input, self.mapping[12])
    col_13 = tf.gather_nd(input, self.mapping[13])
    col_14 = tf.gather_nd(input, self.mapping[14])
    col_15 = tf.gather_nd(input, self.mapping[15])
    
    block0 = self.conv0(tf.transpose(col_0,[1,2,3,0]))
    block1 = self.conv1(tf.transpose(col_1,[1,2,3,0]))
    block2 = self.conv2(tf.transpose(col_2,[1,2,3,0]))
    block3 = self.conv3(tf.transpose(col_3,[1,2,3,0]))
    block4 = self.conv4(tf.transpose(col_4,[1,2,3,0]))
    block5 = self.conv5(tf.transpose(col_5,[1,2,3,0]))
    block6 = self.conv6(tf.transpose(col_6,[1,2,3,0]))
    block7 = self.conv7(tf.transpose(col_7,[1,2,3,0]))
    block8 = self.conv8(tf.transpose(col_8,[1,2,3,0]))
    block9 = self.conv9(tf.transpose(col_9,[1,2,3,0]))
    block10 = self.conv10(tf.transpose(col_10,[1,2,3,0]))
    block11 = self.conv11(tf.transpose(col_11,[1,2,3,0]))
    block12 = self.conv12(tf.transpose(col_12,[1,2,3,0]))
    block13 = self.conv13(tf.transpose(col_13,[1,2,3,0]))
    block14 = self.conv14(tf.transpose(col_14,[1,2,3,0]))
    block15 = self.conv15(tf.transpose(col_15,[1,2,3,0]))
    
    output = self.concat([block0,block1,block2,block3,block4,block5,block6,block7,block8,block9,block10,block11,block12,block13,block14,block15])
      
    return output
