import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.utils import Sequence

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

