from tensorflow.keras import layers
import keras
import tensorflow as tf
from tensorflow_graphics.image import pyramid
from vgg16 import Vgg16


class Unet_expo(keras.Model):
    def __init__(self, img_size, filters=[8, 16, 16, 16, 32, 32, 32], *args, **kwargs):
        super(Unet_expo, self).__init__(*args, **kwargs)
        self.filters = filters
        self.model = self.build_unet(img_size)
        self.w = 10  # for l1, mse
        self.u = 1  # for inner pyramid
        self.vgg = Vgg16()
        self.vgg2 = Vgg16()

        # self.v = 20     # for regress
        # self.patch_size = patch_size
        # self.error_regressor = self.build_regressor(128)
        # self.i = None

    def call(self, x):
        output_images, embedding_vectors, down_samples, up_samples = self.model(x)
        return output_images

    def compile(self,
                optimizer,
                reconstruction_loss,
                metrics
                ):
        super(Unet_expo, self).compile()
        self.optimizer = optimizer
        self.reconstruction_loss = reconstruction_loss
        self.metrics_fn = metrics

    def train_step(self, input_batch):
        high_expos, low_expos = input_batch
        with tf.GradientTape() as tape:
            output_images, embedding_vectors, down_samples, up_samples = self.model(low_expos, training=True)
            L1_loss = self.reconstruction_loss(high_expos, output_images)

            # vgg_pool1, vgg_pool2, vgg_pool3 = self.vgg(input_images, training=False)
            # vgg2_pool1, vgg2_pool2, vgg2_pool3 = self.vgg2(output_images, training=False)
            # perceptual_loss = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)))
            # perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)))
            # perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)))
            #
            # pred_error = self.error_regressor(embedding_vectors)
            # regress_loss = self.reconstruction_loss(errors, pred_error) # l1
            #
            pyramid_loss = self.inner_pyramid_loss(down_samples, up_samples)

            total_loss = L1_loss * self.w + pyramid_loss * self.u  # + perceptual_loss + regress_loss * self.v

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return {'l1': L1_loss,
                # 'perceptual': perceptual_loss,
                # 'regress': regress_loss,
                'pyramid': pyramid_loss,
                'total': total_loss}

    def test_step(self, data):
        high_expos, low_expos  = data
        output_images, embedding_vectors, down_samples, up_samples = self.model(high_expos, training=True)
        # 테스트 데이터세트에서 모델을 실행하는 동안 배치 통계를 원하기 때문에 여기서 training=True는 의도적
        # training=False를 사용하면 훈련 데이터세트에서 학습된 누적 통계를 얻게됨.

        L1_loss = self.reconstruction_loss(low_expos, output_images)

        # vgg_pool1, vgg_pool2, vgg_pool3 = self.vgg(images, training=False)
        # vgg2_pool1, vgg2_pool2, vgg2_pool3 = self.vgg2(output_images, training=False)
        # perceptual_loss = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)))
        # perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)))
        # perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)))
        #
        mse_loss = self.metrics_fn(high_expos, output_images)
        # pred_error = self.error_regressor(embedding_vectors)
        # regress_loss = self.reconstruction_loss(errors, pred_error)  # l1
        #
        # self.i = tf.random.uniform(shape=(), minval=0, maxval=16, dtype=tf.int32)
        pyramid_loss = self.inner_pyramid_loss(down_samples, up_samples)

        total_loss = L1_loss * self.w + pyramid_loss * self.u  # + perceptual_loss + regress_loss * self.v

        psnr, ssim = self.psnr_ssim(high_expos, output_images)
        psnr = tf.math.reduce_mean(psnr)
        ssim = tf.math.reduce_mean(ssim)

        return {'l1': L1_loss,
                'mse': mse_loss,
                # 'perceptual': perceptual_loss,
                # 'regress': regress_loss,
                'pyramid': pyramid_loss,
                'total': total_loss,
                'psnr': psnr,
                'ssim': ssim}

    def psnr_ssim(self, ori, pred):
        psnr = tf.image.psnr(ori, pred, max_val=255)
        ssim = tf.image.ssim(ori, pred, max_val=255)

        return psnr, ssim

    # def get_patches(self, ori, pred, random=True):
    #
    #     ori_patches = tf.image.extract_patches(images=ori,
    #                                            sizes=[1, self.patch_size, self.patch_size, 1],
    #                                            strides=[1, self.patch_size, self.patch_size, 1],
    #                                            rates=[1, 1, 1, 1],
    #                                            padding='VALID')
    #     pred_patches = tf.image.extract_patches(images=pred,
    #                                             sizes=[1, self.patch_size, self.patch_size, 1],
    #                                             strides=[1, self.patch_size, self.patch_size, 1],
    #                                             rates=[1, 1, 1, 1],
    #                                             padding='VALID')
    #
    #     # 패치화 이후 형태 - batch, 2, 8, 64x64x3
    #     if random:
    #         n = int(1024//self.patch_size)
    #         index = [self.i.numpy() // n, self.i.numpy() % n]
    #         random_ori = ori_patches[:, index[0], index[1], :]
    #         random_ori = tf.squeeze(random_ori)
    #         random_ori = tf.reshape(random_ori, [-1, self.patch_size, self.patch_size, 3])
    #
    #         random_pred = pred_patches[:, index[0], index[1], :]
    #         random_pred = tf.squeeze(random_pred)
    #         random_pred = tf.reshape(random_pred, [-1, self.patch_size, self.patch_size, 3])
    #         return random_ori, random_pred
    #
    #     return ori_patches, pred_patches
    #
    # def get_high_freq(self, images, level=2):
    #     images = images*255
    #     upsample = pyramid.upsample(images, num_levels=level)
    #     downsample = pyramid.downsample(upsample[level], num_levels=1)
    #
    #     return upsample[level-1]-downsample[-1]
    #
    # def pyramid_loss(self, ori, pred):
    #     ori_patches, pred_patches = self.get_patches(ori, pred)
    #     ori_high = self.get_high_freq(ori_patches)
    #     pred_high = self.get_high_freq(pred_patches)
    #     pyramid_loss = self.reconstruction_loss(ori_high, pred_high)
    #
    #     return pyramid_loss, ori_high, pred_high

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None,
             save_traces=True):
        self.model.save(filepath)

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     options=None):
        self.model.save_weights(filepath)

    def load_weights(self,
                     filepath,
                     by_name=False,
                     skip_mismatch=False,
                     options=None):
        self.model.load_weights(filepath)

    def build_unet(self, img_size):
        inputs = keras.Input(shape=img_size + (3,))
        n = len(self.filters)

        ### [First half of the network: downsampling inputs] ###
        x = layers.Conv2D(8, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual
        down_samples = []
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, filter in enumerate(self.filters):
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filter, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
            down_samples.append(x)

        embedding_vector = previous_block_activation
        embedding_vector = layers.Flatten()(embedding_vector)

        ### [Second half of the network: upsampling inputs] ###
        self.filters.reverse()
        up_samples = []
        for i, filter in enumerate(self.filters):
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            # symmetric skip connection
            # x = x + self.down_samples[n-i-1]
            up_samples.append(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            # image size * 2
            # x = layers.UpSampling2D(2)(x)
            _, h, w, c = x.shape
            x = tf.image.resize_with_pad(x, target_height=h * 2, target_width=w * 2, method='bilinear', antialias=True)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filter, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # adjust the number of channel
        x = layers.Conv2DTranspose(8, 4, strides=2, padding="same")(x)
        outputs = layers.Conv2D(3, 3, activation='sigmoid', padding="same")(x)

        # Define the model
        model = keras.Model(inputs=inputs, outputs=[outputs, embedding_vector, down_samples, up_samples])
        return model

    def get_high_freq(self, images, level=2):
        ''' for unet_expo in unet.py '''
        images = images * 255
        upsample = pyramid.upsample(images, num_levels=level)
        downsample = pyramid.downsample(upsample[level], num_levels=1)

        return upsample[level - 1] - downsample[-1]

    def inner_pyramid_loss(self, down_samples, up_samples):
        n = len(down_samples)
        pyramid_loss = 0
        for i in range(n):
            pyramid_loss += self.reconstruction_loss(down_samples[i], up_samples[n - i - 1])

        return pyramid_loss

    # def build_regressor(self, input_shape):
    #     dims = [64, 32, 8]
    #
    #     inputs = keras.Input(shape=(input_shape,))
    #     x = inputs
    #     for dim in dims:
    #         x = layers.Dense(dim)(x)
    #         x = layers.Activation('relu')(x)
    #     x = layers.Dense(1, activation='linear')(x)
    #     outputs = x
    #
    #     model = keras.Model(inputs=inputs, outputs=outputs)
    #     return model
    #
    # def predict_error(self, images):
    #     reconstructed_images, embedding = self.model(images)
    #     pred_error = self.error_regressor(embedding)
    #
    #     return pred_error


class Unet_reconst(keras.Model):

    def __init__(self, img_size, patch_size, filters=[8, 16, 16, 16, 32, 32, 32], *args, **kwargs):
        super(Unet_reconst, self).__init__(*args, **kwargs)
        self.filters = filters
        self.model = self.build_unet(img_size)
        self.patch_size = patch_size
        self.w = 100  # for l1, mse
        self.v = 20  # for regress
        self.u = 10  # for pyramid
        self.vgg = Vgg16()
        self.vgg2 = Vgg16()
        self.error_regressor = self.build_regressor(128)
        self.i = None

    def call(self, x):
        reconstructed_images, embedding_vectors = self.model(x)
        return reconstructed_images

    def compile(self,
                optimizer,
                reconstruction_loss,
                metrics
                ):
        super(Unet_reconst, self).compile()
        self.optimizer = optimizer
        self.reconstruction_loss = reconstruction_loss
        self.metrics_fn = metrics

    def train_step(self, input_batch):
        input_images, errors = input_batch
        with tf.GradientTape() as tape:
            reconstructed_images, embedding_vectors = self.model(input_images, training=True)
            L1_loss = self.reconstruction_loss(input_images, reconstructed_images)

            vgg_pool1, vgg_pool2, vgg_pool3 = self.vgg(input_images, training=False)
            vgg2_pool1, vgg2_pool2, vgg2_pool3 = self.vgg2(reconstructed_images, training=False)
            perceptual_loss = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)))
            perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)))
            perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)))

            pred_error = self.error_regressor(embedding_vectors)
            regress_loss = self.reconstruction_loss(errors, pred_error)  # l1

            pyramid_loss, _, _ = self.pyramid_loss(input_images, reconstructed_images)

            total_loss = L1_loss * self.w + regress_loss * self.v + pyramid_loss * self.u + perceptual_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return {'l1': L1_loss,
                'perceptual': perceptual_loss,
                'regress': regress_loss,
                'pyramid': pyramid_loss,
                'total': total_loss}

    def test_step(self, data):
        images, errors = data
        reconstructed_images, embedding_vectors = self.model(images, training=True)
        L1_loss = self.reconstruction_loss(images, reconstructed_images)

        vgg_pool1, vgg_pool2, vgg_pool3 = self.vgg(images, training=False)
        vgg2_pool1, vgg2_pool2, vgg2_pool3 = self.vgg2(reconstructed_images, training=False)
        perceptual_loss = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)))
        perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)))
        perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)))

        mse_loss = self.metrics_fn(images, reconstructed_images)
        pred_error = self.error_regressor(embedding_vectors)
        regress_loss = self.reconstruction_loss(errors, pred_error)  # l1

        self.i = tf.random.uniform(shape=(), minval=0, maxval=16, dtype=tf.int32)
        pyramid_loss, _, _ = self.pyramid_loss(images, reconstructed_images)

        total_loss = L1_loss * self.w + regress_loss * self.v + pyramid_loss * self.u + perceptual_loss

        psnr, ssim = self.psnr_ssim(images, reconstructed_images)
        psnr = tf.math.reduce_mean(psnr)
        ssim = tf.math.reduce_mean(ssim)

        return {'l1': L1_loss,
                'mse': mse_loss,
                'perceptual': perceptual_loss,
                'regress': regress_loss,
                'pyramid': pyramid_loss,
                'total': total_loss,
                'psnr': psnr,
                'ssim': ssim}

    def psnr_ssim(self, ori, pred):
        psnr = tf.image.psnr(ori, pred, max_val=255)
        ssim = tf.image.ssim(ori, pred, max_val=255)

        return psnr, ssim

    def get_patches(self, ori, pred, random=True):

        ori_patches = tf.image.extract_patches(images=ori,
                                               sizes=[1, self.patch_size, self.patch_size, 1],
                                               strides=[1, self.patch_size, self.patch_size, 1],
                                               rates=[1, 1, 1, 1],
                                               padding='VALID')
        pred_patches = tf.image.extract_patches(images=pred,
                                                sizes=[1, self.patch_size, self.patch_size, 1],
                                                strides=[1, self.patch_size, self.patch_size, 1],
                                                rates=[1, 1, 1, 1],
                                                padding='VALID')

        # 패치화 이후 형태 - batch, 2, 8, 64x64x3
        if random:
            n = int(1024 // self.patch_size)
            index = [self.i.numpy() // n, self.i.numpy() % n]
            random_ori = ori_patches[:, index[0], index[1], :]
            random_ori = tf.squeeze(random_ori)
            random_ori = tf.reshape(random_ori, [-1, self.patch_size, self.patch_size, 3])

            random_pred = pred_patches[:, index[0], index[1], :]
            random_pred = tf.squeeze(random_pred)
            random_pred = tf.reshape(random_pred, [-1, self.patch_size, self.patch_size, 3])
            return random_ori, random_pred

        return ori_patches, pred_patches

    def get_high_freq(self, images, level=2):
        images = images * 255
        upsample = pyramid.upsample(images, num_levels=level)
        downsample = pyramid.downsample(upsample[level], num_levels=1)

        return upsample[level - 1] - downsample[-1]

    def pyramid_loss(self, ori, pred):
        ori_patches, pred_patches = self.get_patches(ori, pred)
        ori_high = self.get_high_freq(ori_patches)
        pred_high = self.get_high_freq(pred_patches)
        pyramid_loss = self.reconstruction_loss(ori_high, pred_high)

        return pyramid_loss, ori_high, pred_high

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None,
             save_traces=True):
        self.model.save(filepath)

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     options=None):
        self.model.save_weights(filepath)

    def load_weights(self,
                     filepath,
                     by_name=False,
                     skip_mismatch=False,
                     options=None):
        self.model.load_weights(filepath)

    def build_unet(self, img_size):
        inputs = keras.Input(shape=img_size + (3,))
        n = len(self.filters)

        ### [First half of the network: downsampling inputs] ###
        # Entry block
        x = layers.Conv2D(8, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual
        intermediate_results = []
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, filter in enumerate(self.filters):
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filter, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
            intermediate_results.append(x)

        embedding_vector = previous_block_activation
        embedding_vector = layers.Flatten()(embedding_vector)

        ### [Second half of the network: upsampling inputs] ###
        self.filters.reverse()
        for i, filter in enumerate(self.filters):
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            # symmetric skip connection
            x = x + intermediate_results[n - i - 1]

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filter, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # adjust the number of channel
        x = layers.Conv2DTranspose(8, 3, strides=2, padding="same")(x)
        outputs = layers.Conv2D(3, 3, activation='sigmoid', padding="same")(x)

        # Define the model
        model = keras.Model(inputs=inputs, outputs=[outputs, embedding_vector])
        return model

    def build_regressor(self, input_shape):
        dims = [64, 32, 8]

        inputs = keras.Input(shape=(input_shape,))
        x = inputs
        for dim in dims:
            x = layers.Dense(dim)(x)
            x = layers.Activation('relu')(x)
        x = layers.Dense(1, activation='linear')(x)
        outputs = x

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def predict_error(self, images):
        reconstructed_images, embedding = self.model(images)
        pred_error = self.error_regressor(embedding)

        return pred_error
