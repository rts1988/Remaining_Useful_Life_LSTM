import joblib
from sklearn.metrics import r2_score, mean_absolute_error

def test_LSTM_reg(X_test,y_test,model_filename = 'nn_window10.pkl'):
  """
  test a fitted LSTM classification model, display confusion matrix and classification report

  Input:
  X_test: test data 3D array (num_samples, window, num_features)
  y_test: 1D array with classification model target
  model_filename: .pkl extension filename with fitted model to evaluate

  Output:
  y_test: true values
  y_testpred: predicted values

  """

  

  model = joblib.load('/content/gdrive/MyDrive/'+model_filename)

  # predict with model to get performance metrics

  y_testpred = model.predict(X_test)

  print(y_testpred.shape, y_test.shape)

  # # report, log.
  from sklearn.metrics import r2_score, mean_absolute_error

  print("R squared on test data")
  print(r2_score(y_test,y_testpred))

  print("Mean absolute error on test data")
  print(mean_absolute_error(y_test,y_testpred))

  plt.figure()
  plt.plot(y_test,y_testpred - y_test,'.',alpha=0.3)
  plt.xlabel("Remaining Useful Life")
  plt.ylabel("Error = predicted - true value")
  plt.title("Test data: Model error dependence on remaining useful life")

  plt.figure()
  plt.plot(y_test,y_testpred,'.',alpha=0.3)
  plt.xlabel("True value RUL")
  plt.ylabel("Predicted RUL")
  plt.title("Test data: Predictd vs True Value")

  return y_test,y_testpred
