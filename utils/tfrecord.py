import tensorflow as tf

# 참고 https://www.tensorflow.org/tutorials/load_data/tfrecord#writing_a_tfrecord_file

def _parse_function(proto):
    feature_description = {
        "image_patches":tf.io.FixedLenFeature([],tf.int64, default_value=0),
        "sun_patch":tf.io.FixedLenFeature([],tf.int64, default_value=-1)
    }
    return tf.io.parse_single_example(proto, feature_description)

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_features(image_patches, sun_patch):
    feature = {
        "image_patches": _int64_feature(image_patches),
        "sun_patch": _int64_feature(sun_patch)
    }

    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()