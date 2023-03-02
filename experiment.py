## -- expriment.py --

from preprocess import *
from train_LSTM_class import *
from test_LSTM_class import *
from train_LSTM_reg import *
from train_LSTM_class import *

# modeling parametres:
train_data_file = 'train_FD001.txt'
test_data_file = 'test_FD001.txt'
y_filename = 'RUL_FD001.txt'
classification = False

# preprocessing conditions
window = 20
num_val_engines = 10
scale_window = 10
normalize = True
use_same_scaler = False
min_RUL = 0
max_RUL = 1000

# model arguments
model_filename = 'lstmreg_window10_001.pkl'


# preprocess training data
X_train,y_train,X_val,y_val,df,stscalerdict_train = preprocess_LSTM(filename = train_data_file, window=window,num_val_engines = num_val_engines,scale_window = scale_window, normalize=normalize,stscalerdict = dict(),classification=classification, min_RUL = min_RUL, max_RUL = max_RUL)

# train model
if classification:
  model = train_LSTM_class(X_train,y_train,X_val,y_val,model_filename)
else:
  model = train_LSTM_reg(X_train,y_train,X_val,y_val,model_filename)

# preprocess testing data
if use_same_scaler:
  stscalerdict = stscalerdict_train
else:
  stscalerdict = dict()



X_nil,y_nil,X_test,y_test,df,stscalerdict  = preprocess_LSTM(filename = test_data_file,window=window,scale_window = scale_window,y_filename = y_filename,stscalerdict = stscalerdict, classification = classification, min_RUL = min_RUL, max_RUL = max_RUL)

# test fitted model
if classification:
  y_test,y_testpred = test_LSTM_class(X_test,y_test,model_filename = model_filename)
else:
  y_test,y_testpred = test_LSTM_reg(X_test,y_test,model_filename = model_filename)
