import tensorflow as tf
import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
from vgg16 import Vgg16

from discriminators import get_discriminator, discriminator_loss, generator_loss


class NormLayer(keras.layers.Layer):
    def __init__(self, type):
        super(NormLayer, self).__init__()
        self.layer = None
        self.type = type

    def build(self, input_shape):
        _, h, w, c = input_shape
        if self.type == 'batch':
            self.layer = layers.BatchNormalization(

            )
        elif self.type == 'layer':
            self.layer = layers.LayerNormalization(

            )
        elif self.type == 'group':
            self.layer = layers.GroupNormalization(

            )
        elif self.type == 'instance':
            self.layer = layers.GroupNormalization(
                groups=c,
            )
        else:
            raise Exception("unknown type for normalization")

    def call(self, x):
        return self.layer(x)


class encodingBlock(layers.Layer):
    def __init__(self, filter):
        # 필요한 가중치는 tf.Variable로 만들어야 반복적인 사용 가능
        super(encodingBlock, self).__init__()
        self.filter = filter

        self.activation = layers.Activation("relu")
        self.conv1 = layers.SeparableConv2D(self.filter, 3, padding="same")
        self.batch_norm1 = layers.BatchNormalization()

        self.conv2 = layers.SeparableConv2D(self.filter, 3, padding="same")
        self.batch_norm2 = layers.BatchNormalization()

        self.pooling = layers.MaxPooling2D(4, strides=2, padding="same")

        # for residule
        self.conv3 = layers.Conv2D(self.filter, 1, strides=2, padding="same")

    def call(self, x, previous_block_activation):
        x = self.activation(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)

        x = self.activation(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)

        x = self.pooling(x)

        residual = self.conv3(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        return x


class Encoder(layers.Layer):
    def __init__(self, filters, name, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.conv = layers.Conv2D(8, 4, strides=2, padding="same")
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation("relu")
        self.blocks = self.get_blocks()

    def get_blocks(self):
        blocks = []
        for filter in self.filters:
            blocks.append(encodingBlock(filter))
        return blocks

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)

        previous_block_activation = x  # Set aside residual
        intermediate_results = []

        for encoding_block in self.blocks:
            x = encoding_block(x, previous_block_activation)
            previous_block_activation = x
            intermediate_results.append(x)

        embedding_vector = previous_block_activation
        embedding_vector = layers.Flatten()(embedding_vector)

        return previous_block_activation, embedding_vector, intermediate_results


class decodingBlock(layers.Layer):
    def __init__(self, filter, norm_type='batch'):
        super(decodingBlock, self).__init__()
        self.filter = filter

        self.activation = layers.Activation("relu")
        self.conv2d_trans1 = layers.Conv2DTranspose(filter, 3, padding="same")
        self.norm1 = NormLayer(norm_type)

        self.conv2d_trans2 = layers.Conv2DTranspose(filter, 3, padding="same")
        self.norm2 = NormLayer(norm_type)

        # Project residual
        self.up_sampling = layers.UpSampling2D(2)
        self.conv2d = layers.Conv2D(filter, 1, padding="same")

    def call(self, inputs, previous_block_activation, down_sample):
        x = self.activation(inputs)
        x = self.conv2d_trans1(x)
        x = self.norm1(x)

        # symmetric skip connection
        assert x.shape == down_sample.shape, f"x:{x.shape}, mlp2_outputs:{down_sample.shape}"
        x = x + down_sample
        up_sample = x

        x = self.activation(x)
        x = self.conv2d_trans2(x)
        x = self.norm2(x)

        # image size * 2
        _, h, w, c = x.shape
        x = tf.image.resize_with_pad(x, target_height=h * 2, target_width=w * 2, method='bilinear', antialias=True)

        residual = self.up_sampling(previous_block_activation)
        residual = self.conv2d(residual)
        x = layers.add([x, residual])  # Add back residual
        return x, up_sample


class Decoder(layers.Layer):
    def __init__(self, filters, name, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.decoding_blocks = self.get_blocks()

        self.conv2d_trans = layers.Conv2DTranspose(8, 4, strides=2, padding="same")
        self.conv2d_last = layers.Conv2D(3, 3, activation='sigmoid', padding="same")

    def get_blocks(self):
        blocks = []
        for filter in self.filters:
            if len(blocks)+1 == len(self.filters):
                blocks.append(decodingBlock(filter, norm_type='layer'))
            else:
                blocks.append(decodingBlock(filter))

        return blocks

    def call(self, inputs, down_samples):
        n = len(self.filters)
        x = inputs
        previous_block_activation = inputs

        # [Second half of the network: upsampling inputs]
        intermediate_results = []
        for i, decoding_block in enumerate(self.decoding_blocks):
            x, up_sample = decoding_block(x, previous_block_activation, down_samples[n-i-1])
            previous_block_activation = x  # Set aside next residual
            intermediate_results.append(up_sample)

        # adjust the number of channel
        x = self.conv2d_trans(x)
        outputs = self.conv2d_last(x)

        return outputs #, intermediate_results


class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches, patch_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_dim = patch_dim

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patches = tf.reshape(patches, [batch_size, self.num_patches, -1])
        return patches


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=embedding_dim),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)

        # Transpose inputs from [num_batches, num_patches, hidden_units]
        # to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)

        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches]
        # to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independently.
        mlp2_outputs = self.mlp2(x_patches)

        # Add skip connection.
        assert x.shape == mlp2_outputs.shape, f"x:{x.shape}, mlp2_outputs:{mlp2_outputs.shape}"
        x = x + mlp2_outputs

        return x


class residualBlocksWithMLP(layers.Layer):
    def __init__(self, name, patch_size=1, num_patches=4, embedding_dim=32, **kwargs):
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embedding_dim = patch_size*patch_size*embedding_dim
        self.make_patches = Patches(patch_size=self.patch_size, num_patches=self.num_patches, patch_dim=self.embedding_dim)
        self.mixer1 = MLPMixerLayer(num_patches=self.num_patches, embedding_dim=self.embedding_dim, dropout_rate=0)
        self.mixer2 = MLPMixerLayer(num_patches=self.num_patches, embedding_dim=self.embedding_dim, dropout_rate=0)
        self.mixer3 = MLPMixerLayer(num_patches=self.num_patches, embedding_dim=self.embedding_dim, dropout_rate=0)

    def call(self, inputs):
        patches = self.make_patches(inputs) # num_patches, embedding
        x = self.mixer1(patches)
        x = self.mixer2(x)
        x = self.mixer3(x)
        outputs = tf.reshape(x, (-1, self.patch_size, self.patch_size*4, self.embedding_dim))
        return outputs


class residualBlocks(layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 1, strides=1, padding='same', activation='sigmoid')
        self.conv2 = layers.Conv2D(32, 1, strides=1, padding='same', activation='sigmoid')
        self.conv3 = layers.Conv2D(32, 1, strides=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        outputs = self.conv3(x)
        return outputs


class Identity(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs


class GAN(keras.Model):
    def __init__(self, img_size, type='mlp', filters=[8, 16, 16, 16, 32, 32, 32], num_decoder=1,  *args, **kwargs):
        '''
        filters=[8, 16, 16, 16, 32, 32, 32]
        '''
        super(GAN, self).__init__(*args, **kwargs)
        self.filters = filters
        self.type = type
        self.num_decoder = num_decoder
        self.down_samples = None
        self.up_samples = None

        self.s = 0.5            # for l1
        self.t = 1              # for perceptual
        self.u = 1              # for image quality
        self.v = 1              # for gan loss

        self.vgg = Vgg16()
        self.vgg2 = Vgg16()

        self.encoder = Encoder(filters=filters, name='encoder')
        if type == 'mlp':
            n = len(filters)
            if n < 4:
                patch_size = 16
                num_patches = 4*tf.pow(4, (4-n))
            else:
                patch_size = 7-n+1
                num_patches = 4

            self.residual_blocks1 = residualBlocksWithMLP(patch_size=patch_size, num_patches=num_patches, embedding_dim=32, name='high2low')
            self.residual_blocks2 = residualBlocksWithMLP(patch_size=patch_size, num_patches=num_patches, embedding_dim=32, name='low2high')
        elif type == 'conv':
            self.residual_blocks1 = residualBlocks()
            self.residual_blocks2 = residualBlocks()
        elif type == 'identity':
            self.residual_blocks1 = Identity()
            self.residual_blocks2 = Identity()
        else:
            raise Exception("Type must be mlp or conv or identity")

        if num_decoder == 1:
            self.decoder = Decoder(filters=filters[::-1], name='decoder')
        elif num_decoder == 2:
            self.decoder2high = Decoder(filters=filters[::-1], name='decoder2high')
            self.decoder2low = Decoder(filters=filters[::-1], name='decoder2low')
        else:
            raise Exception("Too many decoder for GAN")

        self.generator = self.build_generator(img_size)
        self.low_discriminator = get_discriminator(input_shape=img_size, n_blocks=5, n_filters=8, )
        self.high_discriminator = get_discriminator(input_shape=img_size, n_blocks=5, n_filters=8, )

    def build_generator(self, img_size):
        high_expos = layers.Input(shape=img_size, name="high_input")
        low_expos = layers.Input(shape=img_size, name="low_input")

        high_encoded, high_embedding_vector, high_intermediate_results = self.encoder(high_expos)
        high_transformed = self.residual_blocks1(high_encoded)
        if self.num_decoder == 1:
            generated_low = self.decoder(inputs=high_transformed, down_samples=high_intermediate_results)
            high_reconstructed = self.decoder(inputs=high_encoded, down_samples=high_intermediate_results)
        else:
            generated_low = self.decoder2low(inputs=high_transformed, down_samples=high_intermediate_results)
            high_reconstructed = self.decoder2high(inputs=high_encoded, down_samples=high_intermediate_results)

        low_encoded, low_embedding_vector, low_intermediate_results = self.encoder(low_expos)
        low_transformed = self.residual_blocks2(low_encoded)
        if self.num_decoder == 1:
            generated_high = self.decoder(inputs=low_transformed, down_samples=low_intermediate_results)
            low_reconstructed = self.decoder(inputs=low_encoded, down_samples=low_intermediate_results)
        else:
            generated_high = self.decoder2high(inputs=low_transformed, down_samples=low_intermediate_results)
            low_reconstructed = self.decoder2low(inputs=low_encoded, down_samples=low_intermediate_results)

        model = tf.keras.Model(inputs=[high_expos, low_expos],
                               outputs=[generated_low, high_reconstructed, generated_high, low_reconstructed],
                               name="generator_in_gan")

        return model

    def compile(self,
                optimizer,
                reconstruction_loss,
                metrics
                ):
        super(GAN, self).compile()
        self.optimizer = optimizer
        self.l1_loss = reconstruction_loss
        self.l2_loss = metrics

        self.low_discriminator.compile(optimizer=optimizer)
        self.high_discriminator.compile(optimizer=optimizer)

    def call(self, data, training=False):
        high_images = data
        generated_images = self.generate_high2low(high_images)

        return generated_images

    def reconstruct(self, images, type=None):
        encoded, embedding_vector, intermediate_results = self.encoder(images)
        if self.num_decoder == 1:
            decoded = self.decoder(inputs=encoded, down_samples=intermediate_results)
        else:
            if type == 'low':
                decoded = self.decoder2low(inputs=encoded, down_samples=intermediate_results)
            elif type == 'high':
                decoded = self.decoder2high(inputs=encoded, down_samples=intermediate_results)
            else:
                raise Exception("improper type for reconstruct")
        return decoded

    def generate_high2low(self, images):
        encoded, embedding_vector, intermediate_results = self.encoder(images)
        transformed = self.residual_blocks1(encoded)
        if self.num_decoder == 1:
            decoded = self.decoder(inputs=transformed, down_samples=intermediate_results)
        else:
            decoded = self.decoder2low(inputs=transformed, down_samples=intermediate_results)
        return decoded

    def generate_low2high(self, images):
        encoded, embedding_vector, intermediate_results = self.encoder(images)
        transformed = self.residual_blocks2(encoded)
        if self.num_decoder == 1:
            decoded = self.decoder(inputs=transformed, down_samples=intermediate_results)
        else:
            decoded = self.decoder2high(inputs=transformed, down_samples=intermediate_results)
        return decoded


    def train_step(self, data):
        high_expos, low_expos = data
        generated_low, high_reconstructed, generated_high, low_reconstructed = \
            self.generator(inputs=[high_expos, low_expos], training=True)
        with tf.GradientTape() as tape:
            disc_low_real = self.low_discriminator(low_expos, training=True)
            disc_low_gen = self.low_discriminator(generated_low, training=True)
            disc_low_re = self.low_discriminator(low_reconstructed, training=True)

            disc_loss = 0
            disc_loss += discriminator_loss(disc_low_real, disc_low_re)
            disc_loss += discriminator_loss(disc_low_real, disc_low_gen)
            disc_low_loss = disc_loss

            low_disc_grads = tape.gradient(disc_low_loss, self.low_discriminator.trainable_weights)
            self.optimizer.apply_gradients(zip(low_disc_grads, self.low_discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            disc_high_real = self.high_discriminator(high_expos, training=True)
            disc_high_gen = self.high_discriminator(generated_high, training=True)
            disc_high_re = self.high_discriminator(high_reconstructed, training=True)

            disc_loss = 0
            disc_loss += discriminator_loss(disc_high_real, disc_high_re)
            disc_loss += discriminator_loss(disc_high_real, disc_high_gen)
            disc_high_loss = disc_loss

            high_disc_grads = tape.gradient(disc_high_loss, self.high_discriminator.trainable_weights)
            self.optimizer.apply_gradients(zip(high_disc_grads, self.high_discriminator.trainable_weights))

        with tf.GradientTape() as tape:

            high_reconstructed = self.reconstruct(high_expos, type='high')
            low_reconstructed = self.reconstruct(low_expos, type='low')

            reconst_image_quality_loss = 0
            reconst_image_quality_loss += self.image_quality_loss(high_expos, high_reconstructed)
            reconst_image_quality_loss += self.image_quality_loss(low_expos, low_reconstructed)

            disc_high_re = self.high_discriminator(high_reconstructed, training=True)
            disc_low_re = self.low_discriminator(low_reconstructed, training=True)

            gen_low_loss = generator_loss(disc_low_re)
            gen_high_loss = generator_loss(disc_high_re)
            gen_reconst_loss = gen_low_loss + gen_high_loss
            gen_total_loss = reconst_image_quality_loss * self.u + gen_reconst_loss * self.v

            reconst_grads = tape.gradient(gen_total_loss, self.generator.trainable_weights)
            self.optimizer.apply_gradients(zip(reconst_grads, self.generator.trainable_weights))

        with tf.GradientTape() as tape:
            generated_low = self.generate_high2low(high_expos)
            generated_high = self.generate_low2high(low_expos)

            # input sequence of image_quality loss is (input, target, reconst, generated)
            generated_image_quality_loss = 0
            generated_image_quality_loss += self.image_quality_loss(high_expos, generated_high)
            generated_image_quality_loss += self.image_quality_loss(low_expos, generated_low)

            disc_high_gen = self.high_discriminator(generated_high, training=True)
            disc_low_gen = self.low_discriminator(generated_low, training=True)

            gen_low_loss = generator_loss(disc_low_gen)
            gen_high_loss = generator_loss(disc_high_gen)
            gen_gen_loss = gen_low_loss + gen_high_loss
            gen_total_loss = generated_image_quality_loss * self.u + gen_gen_loss * self.v

            gen_grads = tape.gradient(gen_total_loss, self.generator.trainable_weights)
            self.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))

        disc_total_loss = disc_low_loss + disc_low_loss
        image_quality_loss = reconst_image_quality_loss + generated_image_quality_loss

        return {'gen_loss': gen_gen_loss,
                'reconst_loss': gen_reconst_loss,
                'image_quality': image_quality_loss,
                'disc_loss': disc_total_loss, }

    def test_step(self, data):
        high_expos, low_expos = data
        generated_low, high_reconstructed, generated_high, low_reconstructed = \
            self.generator(inputs=[high_expos, low_expos], training=True)

        disc_high_real = self.high_discriminator(high_expos, training=True)
        disc_high_gen = self.high_discriminator(generated_high, training=True)
        disc_high_re = self.high_discriminator(high_reconstructed, training=True)

        disc_loss = 0
        disc_loss += discriminator_loss(disc_high_real, disc_high_re)
        disc_loss += discriminator_loss(disc_high_real, disc_high_gen)
        disc_high_loss = disc_loss

        disc_low_real = self.low_discriminator(low_expos, training=True)
        disc_low_gen = self.low_discriminator(generated_low, training=True)
        disc_low_re = self.low_discriminator(low_reconstructed, training=True)

        disc_loss = 0
        disc_loss += discriminator_loss(disc_low_real, disc_low_re)
        disc_loss += discriminator_loss(disc_low_real, disc_low_gen)
        disc_low_loss = disc_loss

        # input, target, reconst, generated
        low_image_quality_loss, high_image_quality_loss = 0, 0
        low_image_quality_loss += self.image_quality_loss(low_expos, generated_low)
        low_image_quality_loss += self.image_quality_loss(low_expos, low_reconstructed)
        high_image_quality_loss += self.image_quality_loss(high_expos, generated_high)
        high_image_quality_loss += self.image_quality_loss(high_expos, high_reconstructed)
        image_quality_loss = low_image_quality_loss + high_image_quality_loss

        gen_low_loss = generator_loss(disc_low_gen)
        gen_high_loss = generator_loss(disc_high_gen)
        gen_loss = gen_low_loss + gen_high_loss
        gen_total_loss = image_quality_loss * self.u + gen_loss * self.v
        disc_total_loss = disc_high_loss + disc_low_loss

        return {'gen_loss+image_quality': gen_total_loss,
                'gen_loss': gen_loss,
                'image_quality': image_quality_loss,
                'low_image_quality': low_image_quality_loss,
                'high_image_quality': high_image_quality_loss,
                'disc_low+high_loss': disc_total_loss,
                'disc_low': disc_low_loss,
                'disc_high': disc_high_loss, }

    def MS_SSIM(self, img1, img2):
        ms_ssim = tf.image.ssim_multiscale(img1, img2, max_val=255)
        return tf.math.reduce_mean(ms_ssim)

    def perceptual_loss(self, img1, img2, training=False):
        vgg_pool1, vgg_pool2, vgg_pool3 = self.vgg(img1, training=False)
        vgg2_pool1, vgg2_pool2, vgg2_pool3 = self.vgg2(img2, training=False)
        perceptual_loss = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)))
        perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)))
        perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)))

        return perceptual_loss

    def image_quality_loss(self, target_images, input_images):

        l1_loss = self.l1_loss(input_images, target_images)
        perceptual_loss = self.perceptual_loss(input_images, target_images)
        ms_ssim = self.MS_SSIM(input_images, target_images)

        total_loss = l1_loss * self.s + perceptual_loss * self.t + ms_ssim

        return total_loss
