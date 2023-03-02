import joblib
from sklearn.metrics import classification_report, confusion_matrix

def test_LSTM_class(X_test,y_test,model_filename = 'nn_window10.pkl'):
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
  # change 3D output (probabilty of each class )to 1 D array containing class of max probability
  y_testpred = np.apply_along_axis(np.argmax,axis=1,arr = model.predict(X_test))

  print(y_testpred.shape, y_test.shape)

  # report metrics

  print(confusion_matrix(y_test,y_testpred))

  print(classification_report(y_test,y_testpred))

  return y_test,y_testpred
