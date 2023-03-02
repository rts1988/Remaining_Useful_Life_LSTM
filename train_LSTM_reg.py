import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import joblib




def train_LSTM_reg(X_train,y_train,X_val,y_val,model_filename,early_stopping_param = 'val_loss'):
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

  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, LSTM
  from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
  from tensorflow.keras.callbacks import EarlyStopping

  # Feed forward LSTM.
  model = Sequential()
  regularizer = keras.regularizers.l2(0.001) # reduce overfitting regularization


  model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
  model.add(LSTM(32, return_sequences = True))
  model.add(LSTM(16, return_sequences = False))
  model.add(Dense(16, activation='relu',kernel_regularizer=regularizer))
  model.add(Dense(1))

  # Early stopping based on arguments
  callbacks = EarlyStopping(monitor=early_stopping_param, min_delta= 50,patience=2,restore_best_weights=True)

  # Compile the model
  model.compile(
    # Optimizer
    optimizer=keras.optimizers.Adam(),
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(),
    # Metric used to evaluate model
    metrics=[keras.metrics.MeanSquaredError()]
)


  # fit model
  history = model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[callbacks], validation_data = (X_val,y_val))


  import joblib
  det = joblib.dump(model, '/content/gdrive/MyDrive/' + model_filename)

  y_val_pred = model.predict(X_val)
  y_train_pred = model.predict(X_train)


  # display fitted model performance on validation data
  from sklearn.metrics import r2_score, mean_absolute_error
  print('Validation R squared, mean absolute error')
  print(r2_score(y_val, y_val_pred))
  print(mean_absolute_error(y_val,y_val_pred))

  print('Train R squared, mean absolute error')
  print(r2_score(y_train, y_train_pred))
  print(mean_absolute_error(y_train,y_train_pred))

  # dependence of model performance on rul range trained
  # for maxrul in range(20,250,5):
  #   print('Test score R squared value',maxrul, r2_score(y_val[y_val<=maxrul], y_val_pred[y_val<=maxrul]))

  # plot error and predicted values vs true values
  plt.figure()
  plt.plot(y_val,y_val_pred - y_val,'.',alpha=0.3)
  plt.xlabel("Remaining Useful Life")
  plt.ylabel("Error = predicted - true value")
  plt.title("Validation data: Model error dependence on remaining useful life")

  plt.figure()
  plt.plot(y_val,y_val_pred,'.',alpha=0.3)
  plt.xlabel("True value RUL")
  plt.ylabel("Predicted RUL")
  plt.title("Validation data: Predictd vs True Value")
  #plt.savefig('/content/gdrive/MyDrive/' + model_filename.split('.')[0]+ '.jpg')

  return model
