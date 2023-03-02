# Remaining_Useful_Life_LSTM
Remaining Useful Life (RUL) prediction with LSTM and XGBoost (regressio nand classification models)

Data from: https://www.nasa.gov/collection-asset/chetan-kulkarni-and-external-partners-release-new-turbofan-engine-degradation-0


Environment:
pandas==1.3.5
numpy==1.22.4
scikit-learn==1.2.1
tensorflow==2.11.0
joblib==1.2.0


Code is in following files:
1. experiment.py - uses below code files to execute a modeling experiment based on preprocessing and training conditions. 
2. preprocess_LSTM.py - preprocess raw data into 3D X arrays for training, validation and testing , with 1D target variable 
3. train_LSTM_class.py - train model for classification based on specified multi-class condition (imminent failue, fault occured, no fault zones)
4. train_LSTM_reg.py - train model for regression RUL prediction
5. test_LSTM_class.py - model metrics and evaluation on test data for classification
6. test_LSTM_reg.py - model metrics and evaluation on test data for regression



