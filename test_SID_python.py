import tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Add, Dropout, Input, Activation
from keras.layers import TimeDistributed, Bidirectional, LSTM, LeakyReLU
from keras.models import Sequential
from keras import optimizers, regularizers
from keras.utils import np_utils, to_categorical
from keras.models import Model, load_model, Sequential
from keras.regularizers import l2

import keras

from IPython.display import clear_output
from tensorflow.python.client import device_lib
from tensorflow.python.keras import backend
import tensorflow as tf

from extract_feat import extract_feats_single_wav

def extract_features_for_all_wavs(dest, label):
    result = np.expand_dims(np.zeros((48, 272)), axis=0)

    for wav in os.listdir(dest):
        vec = extract_feats_single_wav(dest + wav)
        if not str(vec.shape) == '(48, 272)':
            continue
        result = np.vstack((result, np.expand_dims(vec, axis=0)))

    result = result[1:]
    labels = np.expand_dims(np.asarray([label] * len(result)), axis=1)
    print(result.shape)
    print(labels.shape)

    return result, labels

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

def mil_squared_error(y_true, y_pred):
    return tf.keras.backend.square(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))

adam = tf.keras.optimizers.Adam(learning_rate=1e-5)

model = tf.keras.models.load_model('models//cnn.hdf5', custom_objects={'mil_squared_error': mil_squared_error, 'adam': adam})

# perform speaker ID on incoming samples
caregiver_samples = '3-Testing//singles//1-caregiver//'
patient_samples = '3-Testing//singles//2-patient//'

X_caregiver, y_caregiver = extract_features_for_all_wavs(caregiver_samples, 0)
X_patient, y_patient = extract_features_for_all_wavs(patient_samples, 0)
