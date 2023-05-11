import sys
import time

import tensorflow as tf


def scheduler(epoch, lr):
    if epoch != 0 and epoch % 5 == 0:
        return lr * tf.math.exp(-0.1)
    else:
        return lr


class Train(object):
    """Train class.
        Args:
            epochs: Number of epochs
            enable_function: If True, wraps the train_step and test_step in @tf.function
            model: Densenet model.
            batch_size: Batch size.
            strategy: Distribution strategy in use.
    """

    def __init__(self, epochs, optimizer, enable_function, model, batch_size, strategy):
        self.epochs = epochs
        self.batch_size = batch_size
        self.enable_function = enable_function

        self.model = model
        self.strategy = strategy
        self.optimizer = optimizer

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        # self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        #     name='train_accuracy')
        # self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        #     name='test_accuracy')
        self.test_loss_metric = tf.keras.metrics.Sum(name='test_loss')

    def compute_loss(self, label, predictions):
        loss = tf.reduce_sum(self.loss_object(label, predictions)) * (
            1. / self.batch_size)
        loss += (sum(self.model.losses) * 1. / self.strategy.num_replicas_in_sync)
        return loss

    def train_step(self, inputs):
        """One train step.
        Args:
          inputs: one batch input.
        Returns:
          loss: Scaled loss.
        """
        train_loss_dict = self.model.train_step(inputs)

        return train_loss_dict

    def test_step(self, inputs):
        """One test step.
        Args:
          inputs: one batch input.
        """
        test_loss_dict = self.model.test_step(inputs)

        # self.test_acc_metric(label, predictions)
        # self.test_loss_metric(unscaled_test_loss)
        return test_loss_dict

    def custom_loop(self, train_dist_dataset, test_dist_dataset, strategy, callbacks):
        """Custom training and testing loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          test_dist_dataset: Testing dataset created using strategy.
          strategy: Distribution strategy.
          callbacks: list of callbacks which is tf.keras.callbacks.CallbackList type
        Returns:
          train_loss, train_accuracy, test_loss, test_accuracy
        """

        def distributed_train_epoch(ds):
            total_gen_loss, total_reconst_loss, total_disc_loss, total_image_quality_loss = 0.0, 0.0, 0.0, 0.0
            num_train_batches = 0.0

            for one_batch in ds:
                if num_train_batches % 100 == 0:
                    tf.print(num_train_batches, 'th step in train batch', output_stream=sys.stderr)

                per_replica_loss = strategy.run(self.train_step, args=(one_batch,))

                """
                return {'gen_loss': gen_gen_loss,
                        'gen_reconst_loss': gen_reconst_loss,
                        'image_quality': image_quality_loss,
                        'disc_loss': disc_total_loss, }
                """
                # per_replica_loss 는 gpu 길이와 같은 길이의 튜플을 리턴
                gen_reconst_loss = per_replica_loss['reconst_loss']
                gen_gen_loss = per_replica_loss['gen_loss']
                disc_loss = per_replica_loss['disc_loss']
                image_quality = per_replica_loss['image_quality']

                total_reconst_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, gen_reconst_loss, axis=None)
                total_gen_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, gen_gen_loss, axis=None)
                total_disc_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
                total_image_quality_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, image_quality, axis=None)
                num_train_batches += 1

            total_train_loss_dict = {'train_disc_loss': total_disc_loss/(num_train_batches*self.batch_size),
                                     'train_generating_loss': total_gen_loss/(num_train_batches*self.batch_size),
                                     'train_reconst_loss': total_reconst_loss / (num_train_batches * self.batch_size),
                                     'train_image_quality': total_image_quality_loss/(num_train_batches*self.batch_size)}
            return total_train_loss_dict, num_train_batches

        def distributed_test_epoch(ds):
            total_gen_loss, total_image_quality_loss, total_disc_high_loss, total_disc_low_loss = 0.0, 0.0, 0.0, 0.0
            num_test_batches = 0.0

            for one_batch in ds:
                if num_test_batches % 100 == 0:
                    tf.print(num_test_batches, 'th step in test batch', output_stream=sys.stderr)

                per_replica_loss = strategy.run(self.test_step, args=(one_batch,))
                """
                return {'gen_loss+image_quality': gen_total_loss,
                        'gen_loss': gen_loss,
                        'image_quality': image_quality_loss,
                        'disc_low+high_loss': disc_total_loss,
                        'disc_low': disc_low_loss,
                        'disc_high': disc_high_loss, }
                """
                gen_loss = per_replica_loss['gen_loss']
                disc_low_loss = per_replica_loss['disc_low']
                disc_high_loss = per_replica_loss['disc_high']
                image_quality = per_replica_loss['image_quality']

                total_gen_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
                total_disc_low_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, disc_low_loss, axis=None)
                total_disc_high_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, disc_high_loss, axis=None)
                total_image_quality_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, image_quality, axis=None)

                num_test_batches += 1

            total_test_loss_dict = {'test_disc_low': total_disc_low_loss/(num_test_batches*self.batch_size),
                                    'test_disc_high': total_disc_high_loss/(num_test_batches*self.batch_size),
                                    'test_gen_loss': total_gen_loss/(num_test_batches*self.batch_size),
                                    'test_image_quality': total_image_quality_loss/(num_test_batches*self.batch_size)}
            return total_test_loss_dict, num_test_batches

        if self.enable_function:
            distributed_train_epoch = tf.function(distributed_train_epoch)
            distributed_test_epoch = tf.function(distributed_test_epoch)

        for epoch in range(self.epochs):
            start = time.time()
            self.optimizer.learning_rate = scheduler(epoch=epoch, lr=self.optimizer.learning_rate)

            train_total_loss_dict, num_train_batches = distributed_train_epoch(train_dist_dataset)
            train_image_quality = train_total_loss_dict['train_image_quality']/num_train_batches
            train_generating_loss = train_total_loss_dict['train_generating_loss']/num_train_batches
            train_reconst_loss = train_total_loss_dict['train_reconst_loss'] / num_train_batches
            train_disc_loss = train_total_loss_dict['train_disc_loss']/num_train_batches

            test_total_loss_dict, num_test_batches = distributed_test_epoch(test_dist_dataset)
            test_image_quality = test_total_loss_dict['test_image_quality']/num_test_batches
            test_gen_loss = test_total_loss_dict['test_gen_loss']/num_test_batches
            test_disc_low_loss = test_total_loss_dict['test_disc_low']/num_test_batches
            test_disc_high_loss = test_total_loss_dict['test_disc_high']/num_test_batches
            end = time.time()

            template = ('{} Epoch\n'
                        '{} train batches, {} test batches takes {} sec, \n'
                        'Train Geration Loss: {}, Train Reconstruction Loss: {}, Train Discriminator Loss: {}, Train Image Quality: {}, \n'
                        'Test Generator Loss: {}, Test Discriminator Low: {}, Test Discriminator High:{}, '
                        'Test Image Quality: {}')
            print(template.format(epoch,
                                  num_train_batches,
                                  num_test_batches,
                                  end-start,
                                  train_reconst_loss,
                                  train_generating_loss,
                                  train_disc_loss,
                                  train_image_quality,
                                  test_gen_loss,
                                  test_disc_low_loss,
                                  test_disc_high_loss,
                                  test_image_quality)
                  )

            logs = {}
            logs.update(train_total_loss_dict)
            logs.update(test_total_loss_dict)
            logs['lr'] = self.optimizer.learning_rate

            callbacks.on_epoch_end(epoch=epoch, logs=logs)

