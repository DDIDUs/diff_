import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
import numpy as np

def format_example(image, label):
    #print(image.shape, label.shape)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (32, 32))
    label = tf.one_hot(label, 10)
    return image, label

def tfds_imgen(ds, imgen, num_batches):
	for images, labels in ds.batch(batch_size=32).prefetch(buffer_size=1):
		labels = tf.one_hot(labels, 10)
		flow = imgen.flow(images, labels, 32)
		for _ in range(num_batches):
			yield next(flow)

class Cifar10C:
	def __init__(self):
		self.BATCH_SIZE = 32
		self.IMG_HEIGHT = 32
		self.IMG_WIDTH = 32
		self.CLASS_NUM = 10

		ds_test = tfds.load(name="cifar10_corrupted", split=tfds.Split.TEST, batch_size=-1)
		ds_test = tfds.as_numpy(ds_test)
		self.x_test, self.y_test = ds_test["image"], ds_test["label"]

	def test_data(self):
		self.x_test = tf.cast(self.x_test, tf.float32)
		self.x_test = self.x_test / 255.0
		self.x_test = tf.image.resize(self.x_test, [32,32])	

		return self.x_test, self.y_test
	
def 
	
class STL10:
	def __init__(self):
		self.BATCH_SIZE = 32
		self.IMG_HEIGHT = 32
		self.IMG_WIDTH = 32
		self.CLASS_NUM = 10

		ds_test = tfds.load(name="stl10", split=tfds.Split.TEST, batch_size=-1)
		ds_test = tfds.as_numpy(ds_test)
		self.x_test, self.y_test = ds_test["image"], ds_test["label"]


	def test_data(self):
		self.x_test = tf.cast(self.x_test, tf.float32)
		self.x_test = self.x_test / 255.0
		self.x_test = tf.image.resize(self.x_test, [32,32], method='area') # area
		
		# Because the label is different from cifar10, additional processing is required.
		self.x_test = np.delete(self.x_test, np.where(self.y_test == 7),axis=0)
		self.y_test = np.delete(self.y_test, np.where(self.y_test == 7),axis=0)

		self.y_test = np.where(self.y_test == 6, 7, self.y_test)
		self.y_test = np.where(self.y_test == 1, 6, self.y_test)
		self.y_test = np.where(self.y_test == 2, 1, self.y_test)
		self.y_test = np.where(self.y_test == 6, 2, self.y_test)

		return self.x_test, self.y_test

class KerasCifar10:
	
	def __init__(self):
		self.BATCH_SIZE = 32
		self.IMG_HEIGHT = 32
		self.IMG_WIDTH = 32
		self.CLASS_NUM = 10
		(x_train, y_train), (self.x_test, self.y_test) = cifar10.load_data()
		self.x_train = x_train[:45000]
		self.x_val = x_train[45000:]
		self.y_train = y_train[:45000]
		self.y_val = y_train[45000:]
	
	def train_data_aug(self):

		self.x_train = tf.cast(self.x_train, tf.float32)
		self.x_val = tf.cast(self.x_val, tf.float32)
		self.x_test = tf.cast(self.x_test, tf.float32)

		self.x_train = self.x_train / 255.0
		self.x_val = self.x_val / 255.0
		self.x_test = self.x_test / 255.0

		self.x_train = tf.image.resize(self.x_train, (32, 32))
		self.x_val = tf.image.resize(self.x_val, (32, 32))
		self.x_test = tf.image.resize(self.x_test, (32, 32))
		
		datagen = ImageDataGenerator(
			# set input mean to 0 over the dataset
			featurewise_center=False,
			# set each sample mean to 0
			samplewise_center=False,
			# divide inputs by std of dataset
			featurewise_std_normalization=False,
			# divide each input by its std
			samplewise_std_normalization=False,
			# apply ZCA whitening
			zca_whitening=False,
			# epsilon for ZCA whitening
			zca_epsilon=1e-06,
			# randomly rotate images in the range (deg 0 to 180)
			rotation_range=0,
			# randomly shift images horizontally
			width_shift_range=0.1,
			# randomly shift images vertically
			height_shift_range=0.1,
			# set range for random shear
			shear_range=0.,
			# set range for random zoom
			zoom_range=0.,
			# set range for random channel shifts
			channel_shift_range=0.,
			# set mode for filling points outside the input boundaries
			fill_mode='nearest',
			# value used for fill_mode = "constant"
			cval=0.,
			# randomly flip images
			horizontal_flip=True,
			# randomly flip images
			vertical_flip=False,
			# set rescaling factor (applied before any other transformation)
			rescale=None,
			# set function that will be applied on each input
			preprocessing_function=None,
			# image data format, either "channels_first" or "channels_last"
			data_format=None,
			# fraction of images reserved for validation (strictly between 0 and 1)
			validation_split=0.0)

		# Compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(self.x_train)

		return datagen, self.x_train, to_categorical(self.y_train, self.CLASS_NUM), self.x_val, to_categorical(self.y_val, self.CLASS_NUM), self.x_test, to_categorical(self.y_test, self.CLASS_NUM)

	def train_data(self):
		self.x_train = tf.cast(self.x_train, tf.float32)
		self.x_val = tf.cast(self.x_val, tf.float32)
		self.x_test = tf.cast(self.x_test, tf.float32)

		self.x_train = self.x_train / 255.0
		self.x_val = self.x_val / 255.0
		self.x_test = self.x_test / 255.0

		self.x_train = tf.image.resize(self.x_train, (32, 32))
		self.x_val = tf.image.resize(self.x_val, (32, 32))
		self.x_test = tf.image.resize(self.x_test, (32, 32))

		self.x_train = np.concatenate((self.x_train, self.x_val), axis=0)
		self.y_train = np.concatenate((self.y_train, self.y_val), axis=None)

		return self.x_train, to_categorical(self.y_train, self.CLASS_NUM), self.x_val, to_categorical(self.y_val, self.CLASS_NUM), self.x_test, to_categorical(self.y_test, self.CLASS_NUM)
	
	def test_data(self):
		self.x_test = tf.cast(self.x_test, tf.float32)
		self.x_test = self.x_test / 255.0
		self.x_test = tf.image.resize(self.x_test, (32, 32))

		return self.x_train, self.y_train 


if __name__ == "__main__":

    data = STL10()
    x_test, y_test = data.train_data()