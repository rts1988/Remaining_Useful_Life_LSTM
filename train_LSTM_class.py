import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.metrics import Accuracy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def train_LSTM_class(X_train,y_train,X_val,y_val,model_filename,early_stopping_param = 'val_loss'):
  """
  trains LSTM model classification, saves model to file

  input:
  X_train, X_val: 3D array (num rows, timesteps, num sensors) for training and validation data
  y_train, y_val: target remaining useful life at end of timestep window for training and validation
  model_filename: name of file to store fitted model.
  early_stopping_param : ('val_loss' or 'loss') based on what to monitor to stop training

  Output:
  model : fitted LSTM model
  """



  # Feed forward LSTM.
  model = Sequential()
  regularizer = keras.regularizers.l2(0.001) # reduce overfitting regularization


  model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
  model.add(LSTM(32))
  model.add(Dense(16, activation='relu',kernel_regularizer=regularizer))
  #model.add(layers.Dropout(0.1))
  model.add(Dense(3,activation = 'softmax'))

  # Early stopping based on arguments
  callbacks = EarlyStopping(monitor=early_stopping_param, patience=2,restore_best_weights=True)

  # Compile the model
  model.compile(
    # Optimizer
    optimizer=keras.optimizers.Adam(),
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # Metric used to evaluate model
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)


  # fit model
  history = model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[callbacks], validation_data = (X_val,y_val))


  import joblib
  det = joblib.dump(model, '/content/gdrive/MyDrive/' + model_filename)

  y_val_pred = np.apply_along_axis(np.argmax,axis=1,arr = model.predict(X_val)) # change 3D output (probabiltiy of each class to 1 D array containing class of max probability)

  y_train_pred = np.apply_along_axis(np.argmax,axis=1,arr = model.predict(X_train))

  # display fitted model performance on validation data
  from sklearn.metrics import classification_report, confusion_matrix
  print('Validation classification report')
  print(classification_report(y_val, y_val_pred))
  print('Train classification report')
  print(classification_report(y_train, y_train_pred))

  #plt.savefig('/content/gdrive/MyDrive/nn_window10.jpg')

  return model
