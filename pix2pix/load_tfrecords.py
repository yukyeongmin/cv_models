import tensorflow as tf
import glob

from utils import tone_mapping

# https://www.tensorflow.org/tutorials/load_data/tfrecord#write_the_tfrecord_file

TF_TRAIN_DIR = "./tf_records/synthetic/train/*.tfrecords"
TF_VAL_DIR = "./tf_records/synthetic/val/*.tfrecords"
TF_TEST_DIR = "./tf_records/synthetic/test/*.tfrecords"

BATCH_SIZE = 128
IMSHAPE = [32, 128, 3]

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'ldr': tf.io.FixedLenFeature([], tf.string),
        'hdr': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    ref_HDR = tf.io.decode_raw(example['hdr'], tf.float32)
    ref_LDR = tf.io.decode_raw(example['ldr'], tf.float32)

    ref_HDR = tf.reshape(ref_HDR, IMSHAPE)
    ref_LDR = tf.reshape(ref_LDR, IMSHAPE)

    ref_HDR = tone_mapping(ref_HDR)
    
    # ref_HDR = ref_HDR / (1e-6 + tf.reduce_mean(ref_HDR)) * 0.5
    # ref_LDR = ref_LDR / 255.0

    # # BGR2RGB
    # ref_HDR = ref_HDR[:,:,::-1]
    # ref_LDR = ref_LDR[:,:,::-1]

    return ref_LDR, ref_HDR

def load_tfrecords(path):
    train_ds = sorted(glob.glob(path))
    dataset = tf.data.TFRecordDataset(filenames=train_ds, num_parallel_reads=tf.data.AUTOTUNE, compression_type="GZIP")

    parsed_image_dataset = dataset.map(_parse_function).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return parsed_image_dataset

    # ldrs=[]
    # hdrs=[]
    # for ldr, hdr in parsed_image_dataset:
    #     if ldr.shape[0]!= BATCH_SIZE[0]:
    #         break
    #     ldrs.append(ldr)
    #     hdrs.append(hdr)

    # ldrs = tf.data.Dataset.from_tensor_slices(ldrs)
    # hdrs = tf.data.Dataset.from_tensor_slices(hdrs)

    #return ldrs, hdrs #(batch,h,w,c)의 list

# load_tfrecords(TF_TRAIN_DIR)
