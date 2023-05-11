from tensorflow.keras import layers
import keras
import tensorflow as tf


def discriminator_loss(disc_real_output, disc_generated_output, from_logits=False,
                       reduction=tf.keras.losses.Reduction.SUM):
    '''
    generated image 와 gt image에 대한 discriminator의 결과를 입력 받아 loss(binary cross entropy)를 출력한다.
    cross entropy with logits https://stats.stackexchange.com/questions/242907/why-use-binary-cross-entropy-for-generator-in-adversarial-networks
    '''

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,
                                                     reduction=reduction)
    if disc_real_output.shape != disc_generated_output.shape:
        n = min(len(disc_real_output), len(disc_generated_output))
        disc_real_output = disc_real_output[:n]
        disc_generated_output = disc_generated_output[:n]

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    if from_logits:
        total_disc_loss += tf.squeeze(disc_generated_output)

    # tf.distribute.Strategy 사용시 reduce_mean 사용 자제
    # 손실의 형상을 확인 할 것!!
    return total_disc_loss/2


def generator_loss(disc_generated_output, from_logits=False, reduction=tf.keras.losses.Reduction.SUM):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,
                                                     reduction=reduction)
    generated_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    return generated_loss


class NormLayer(keras.layers.Layer):
    def __init__(self, type):
        super(NormLayer, self).__init__()
        self.layer = None
        self.type = type

    def build(self, input_shape):
        _, h, w, c = input_shape
        if self.type == 'batch':
            self.layer = tf.keras.layers.BatchNormalization(

            )
        elif self.type == 'layer':
            self.layer = tf.keras.layers.LayerNormalization(

            )
        elif self.type == 'group':
            self.layer = tf.keras.layers.GroupNormalization(

            )
        elif self.type == 'instance':
            self.layer = tf.keras.layers.GroupNormalization(
                groups=c,
            )
        else:
            raise Exception("unknown type for normalization")

    def call(self, x):
        return self.layer(x)


def get_discriminator(input_shape, n_blocks, n_filters, norm_type='batch', activation_type='relu'):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(n_blocks):
        model.add(
            layers.Conv2D(filters=n_filters * (i + 1),
                          kernel_size=3,
                          strides=2,
                          padding='valid',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          ))
        if i == 0:
            model.add(NormLayer('layer'))
        else:
            model.add(NormLayer(norm_type))
        model.add(layers.Activation(activation_type))
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    return model

# class PatchDiscriminator(keras.Model):
#     def __init__(self, img_size, patch_size, n_blocks, n_filters, norm_type='batch', activation_type='relu'):
#         super(PatchDiscriminator, self).__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.n_layers = n_blocks
#         self.n_filters = n_filters
#         self.norm_type = norm_type
#         self.activation_type = activation_type
#         self.model = None
#
#     def build(self, input_shape):
#         """이미지를 입력받은뒤 realness에 대한 평가를 하여 0-1사이 값을 출"""
#
#         self.model = keras.Sequential()
#         self.model.add(layers.Input(shape=input_shape))
#         for i in range(self.n_layers):
#             self.model.add(
#                 layers.Conv2D(filters=self.n_filters * (i + 1),
#                               kernel_size=3,
#                               strides=2,
#                               padding='valid',
#                               use_bias=True,
#                               kernel_initializer='glorot_uniform',
#                               bias_initializer='zeros',
#                               ))
#             self.model.add(NormLayer(self.norm_type))
#             self.model.add(layers.Activation(self.activation_type))
#             self.model.add(layers.Dropout(0.3))
#
#         self.model.add(layers.Flatten())
#         self.model.add(layers.Dense(1))
#         self.model.add(layers.Activation('sigmoid'))
#
#     def call(self, inputs, training=None, mask=None):
#         return self.model(inputs, )
