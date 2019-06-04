from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# CSV columns in the input file.
with open(train_file_path, 'r') as f:
    names_row = f.readline()


CSV_COLUMNS = names_row.rstrip('\n').split(',')
print(CSV_COLUMNS)

LABELS = [0, 1]
LABEL_COLUMN = 'survived'

FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,)
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

examples, labels = next(iter(raw_train_data)) # Just the first batch.
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

def process_categorical_data(data, categories):
  """Returns a one-hot encoded tensor representing categorical values."""
  
  # Remove leading ' '.
  data = tf.strings.regex_replace(data, '^ ', '')
  # Remove trailing '.'.
  data = tf.strings.regex_replace(data, r'\.$', '')
  
  # ONE HOT ENCODE
  # Reshape data from 1d (a list) to a 2d (a list of one-element lists)
  data = tf.reshape(data, [-1, 1])
  # For each element, create a new list of boolean values the length of categories,
  # where the truth value is element == category label
  data = tf.equal(categories, data)
  # Cast booleans to floats.
  data = tf.cast(data, tf.float32)
  
  # The entire encoding can fit on one line:
  # data = tf.cast(tf.equal(categories, tf.reshape(data, [-1, 1])), tf.float32)
  return data

class_tensor = examples['class']
class_tensor

class_categories = CATEGORIES['class']
class_categories

processed_class = process_categorical_data(class_tensor, class_categories)
processed_class

print("Size of batch: ", len(class_tensor.numpy()))
print("Number of category labels: ", len(class_categories))
print("Shape of one-hot encoded tensor: ", processed_class.shape)

def process_continuous_data(data, mean):
  # Normalize data
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])

MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

age_tensor = examples['age']
age_tensor

process_continuous_data(age_tensor, MEANS['age'])

def preprocess(features, labels):
  
  # Process categorial features.
  for feature in CATEGORIES.keys():
    features[feature] = process_categorical_data(features[feature],
                                                 CATEGORIES[feature])

  # Process continuous features.
  for feature in MEANS.keys():
    features[feature] = process_continuous_data(features[feature],
                                                MEANS[feature])
  
  # Assemble features into a single tensor.
  features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1)
  
  return features, labels

train_data = raw_train_data.map(preprocess).shuffle(500)
test_data = raw_test_data.map(preprocess)

examples, labels = next(iter(train_data))

examples, labels

def get_model(input_dim, hidden_units=[100]):
  """Create a Keras model with layers.

  Args:
    input_dim: (int) The shape of an item in a batch. 
    labels_dim: (int) The shape of a label.
    hidden_units: [int] the layer sizes of the DNN (input layer first)
    learning_rate: (float) the learning rate for the optimizer.

  Returns:
    A Keras model.
  """

  inputs = tf.keras.Input(shape=(input_dim,))
  x = inputs

  for units in hidden_units:
    x = tf.keras.layers.Dense(units, activation='relu')(x)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs, outputs)
 
  return model

input_shape, output_shape = train_data.output_shapes

input_dimension = input_shape.dims[1] # [0] is the batch size

model = get_model(input_dimension)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))

