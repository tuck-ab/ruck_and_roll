import os
import cv2
import numpy as np
import random
import pathlib


import tensorflow as tf
import tensorflow.keras.layers

from tensorflow.keras.layers import TimeDistributed

from keras.models import Sequential
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, Dropout

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request

EPOCHS = 30

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result


class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.mp4'))
    classes = [p.parent.name for p in video_paths] 
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames) 
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label


# Test Code
train_data_dir = pathlib.Path('/dcs/large/u1901447/videos/clips/data_35_frames/train')
val_data_dir = pathlib.Path('/dcs/large/u1901447/videos/clips/data_35_frames/val')
fg = FrameGenerator(train_data_dir, 25, training=True)
#print("get",fg.get_files_and_class_names())

print(fg.class_names)
fg.class_ids_for_name

frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")


# (8, 25, 224, 224, 3)
# Create the training set
output_signature = (tf.TensorSpec(shape = (25, 224, 224, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(train_data_dir, 25, training=True),
                                          output_signature = output_signature)

# print(type(train_ds)) #<class 'tensorflow.python.data.ops.dataset_ops.FlatMapDataset'>

# Create the validation set
val_ds = tf.data.Dataset.from_generator(FrameGenerator(val_data_dir, 25),
                                        output_signature = output_signature)

# Test Code Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

train_ds = train_ds.batch(8)
val_ds = val_ds.batch(8)

print(type(train_ds)) #class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')
print(train_labels)

print(type(train_frames)) #<class 'tensorflow.python.framework.ops.EagerTensor'>
print(len(train_frames)) #8

# net = tf.keras.applications.EfficientNetB0(include_top = False)
# net.trainable = False

# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(scale=255),
#     tf.keras.layers.TimeDistributed(net),
#     # tf.keras.layers.Dense(4), #hmm
#     tf.keras.layers.Dense(13),
#     tf.keras.layers.GlobalAveragePooling3D()
# ])

# def model(input_shape):
#     X_input = tf.keras.layers.Input(shape=input_shape)
#     layer1 = tf.keras.layers.Conv3D(
#         filters=3, kernel_size=3, strides=1)(X_input)
#     X_output = tf.keras.layers.Dense(13)(layer1)
#     model = tf.keras.models.Model(inputs=X_input, outputs=X_output)
#     return model

model = Sequential()
model.add(Conv3D(
    16, (3,3,3), activation='relu', input_shape=(25,224,224,3)
))
model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
model.add(Flatten())
# model.add(Dense(128))
# model.add(Dropout(0.5))
model.add(Dense(13, activation='softmax'))

# print(train_ds.__dict__.keys())

# model = model((25, 224, 224, 3))

# model.compile(optimizer = 'adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#               metrics=['accuracy']
# )

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
# model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"])

history = model.fit(train_ds,
          epochs = EPOCHS, #change this later
          validation_data = val_ds,
          callbacks = tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'))


import matplotlib.pyplot as plt
def plot_history(history):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(18.5, 10.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(history.history['loss'] + history.history['val_loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation']) 

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.show()

plot_history(history)


def get_actual_predicted_labels(dataset): 
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels, ds_type):
  cm = tf.math.confusion_matrix(actual, predicted)
  ax = sns.heatmap(cm, annot=True, fmt='g')
  sns.set(rc={'figure.figsize':(12, 12)})
  sns.set(font_scale=1.4)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  ax.set_xlabel('Predicted Action')
  ax.set_ylabel('Actual Action')
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)

fg = FrameGenerator(train_data_dir, 25, training=True)
labels = list(fg.class_ids_for_name.keys())

actual, predicted = get_actual_predicted_labels(train_ds)

import seaborn as sns
plot_confusion_matrix(actual, predicted, labels, 'training')

test_data_dir = pathlib.Path('/dcs/large/u1901447/videos/clips/data_35_frames/test')
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
n_frames = 25
test_ds = tf.data.Dataset.from_generator(FrameGenerator(test_data_dir, n_frames),
                                         output_signature = output_signature)
test_ds = test_ds.batch(8)

actual, predicted = get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, labels, 'test')

model.save("mymodel_batch", save_format='tf')
# tf.saved_model.save(model, "my_saved_model")
# tf.keras.models.save_model(model, "mymodel_keras")
# with open("mymodel.pkl","wb") as f:
#   pickle.dump(model, f)

