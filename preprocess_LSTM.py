from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_LSTM(filename = 'train_FD001.txt', window = 10,normalize = True, scale_window = 5,stscalerdict = dict(),num_val_engines = 10, y_filename = '',train_engines = [], min_RUL = 0, max_RUL = 1000, classification = True):
  """
  preprocess train or test data to get 3D array for LSTM input and 1D array for target for either LSTM regression or classification.

  Input:
  filename : string,  including extension of .txt file containing the raw data for train or test.
  window : integer range (min = 3), length of window for time series LSTM input
  normalize: boolean True or False, normalizing the sensor values to account for individual sensor offset and error characteristics (standard scaler used)
  scale_window : integer (min = 5), window of initial values used for normalizing sensor data.
  stscalerdict : optional, dict type, standard scaler option for using same scaler for train and test
  num_val_engines: number of engines separated for validation
  y_filename : optional, string, including extension of .txt file used for test files only.
  train_engines: optional, set which engines are used based on the engine ID (unit_number in raw data)
  min_RUL , max_RUL : optional, integers (0 to 1000) minimum and maximum RUL (remaining useful life) to include for training - used for model evaluation
  classification : boolean, classification or regression modeling (calls function RUL_class if true)

  Output:
  X_train: 3D array in form (num samples, window, num features) ready for training
  y_train: 1 D array with target
  X_val: 3 D array for validation
  y_val: 1 D array for validation target,
  df: 2D data frame with raw data values
  stscalerdict : standard scaler used for training data for the option to retain for test data

  """

  window = int(window)


  assert type(window)==int, 'window needs to be an integer more than 3'
  assert num_val_engines < 100, 'number of validation engines needs to be less than 100'

  # reading in file to pandas dataframe
  df = pd.read_csv('/content/gdrive/MyDrive/'+filename, sep=" ", header=None)
  columns = ['unit_number','cycle_time','setting_1','setting_2','setting_3'] + list('sensor_'+str(i) for i in range(1,df.shape[1]-5+1))
  df.columns = columns
  print(df.shape)

  # dropping blank and constant value columns (found in EDA):
  df.drop(['setting_3','sensor_1','sensor_10','sensor_18','sensor_19','sensor_22','sensor_23'],axis=1,inplace=True)


  # normalizing each sensor value based on its error characteristics:

  if normalize == True:
    sensor_columns = [col for col in df.columns if col.startswith("sensor")]
    dfsensors = pd.DataFrame()


    for engine_num in df['unit_number'].unique():
      engine_sensors = df.loc[df['unit_number']==engine_num,:]
      if engine_num not in stscalerdict: # if no std scaler trained for this unit
        stscalerdict[engine_num] = StandardScaler()
        stscalerdict[engine_num].fit(engine_sensors.loc[engine_sensors['cycle_time']<scale_window,sensor_columns])
      transformed_sensors = stscalerdict[engine_num].transform(engine_sensors.loc[:,sensor_columns])
      dfsensors = pd.concat([dfsensors,pd.DataFrame(transformed_sensors,columns=sensor_columns)],axis=0)

    dfsensors.index = df.index

    df = pd.concat([df.loc[:,[col for col in df.columns if col not in sensor_columns]],dfsensors],axis=1,ignore_index=False)

  # computing target variable Remaining Useful Life

  if 'train' in filename.lower(): # addressing difference in train and test raw files, since test target variable in separate file with different computation
    RUL = [] # empty list to concatenate
    for engine_num in df['unit_number'].unique(): # for each individual engine
      engine_data = df.loc[df['unit_number']==engine_num,:] # subset of data for individual engine
      failed_cycle = engine_data['cycle_time'].max()# find the last cycle at failure
      RUL += (failed_cycle-engine_data['cycle_time']).tolist() # compute remaining useful life
    df['RUL'] = RUL

  if 'test' in filename.lower(): # vector in file contains last cycle real RUL.
    last_RULs = pd.read_csv('/content/gdrive/MyDrive/'+y_filename,sep=", ",header=None)
    RUL = [] # initialize RUL column
    for engine_num in df['unit_number'].unique():
      engine_data = df.loc[df['unit_number']==engine_num,:] # individual engine data
      numcycles_run = engine_data.shape[0] # number of cycles run
      engine_lastRUL = last_RULs.iloc[engine_num-1,0] # last cycle's RUL value (engine_num-1 because index strts at 0, unit number starts at 1)
      RUL += list(range(engine_lastRUL+numcycles_run-1,engine_lastRUL-1,-1)) # computing RUL values for all the cycles.
    df['RUL'] = RUL

  # subsetting range of min to max RUL
  df = df.loc[(df['RUL']<=max_RUL+window+1) & (df['RUL']>=min_RUL),:]

  # adding timestep features for each sensor based on the window.
  sensor_columns = [col for col in df.columns if col.startswith("sensor")]+['cycle_time'] # including cycle time
  print(len(sensor_columns))
  print(type(window))

  count = 0 # intializing counter for displaying progress

  # split data into training and validation based on engine number.
  if 'train' in filename:
    if train_engines == []:
      val_engines = np.random.choice(a = df['unit_number'].unique(),size = num_val_engines,replace = False)
      train_engines = set(df['unit_number'].unique()).difference(set(val_engines))
      print(train_engines)
    else:
      val_engines = set(df['unit_number'].unique()).difference(set(train_engines))
  else:
    val_engines = df['unit_number'].unique()
    train_engines = []
    X_train = np.array([])
    y_train = np.array([])


  for engine_num in train_engines: # for each engine
    engine_data = df.loc[df['unit_number']==engine_num,sensor_columns+['RUL']] # individual engine dataframe
    engine_wins = np.zeros((engine_data.shape[0]-window+1,window,len(sensor_columns))) # initializing 3d array for individual engine
    target = np.zeros((engine_data.shape[0]-window+1,1)) # initializing array for target for individual engine
    for i,ind in enumerate(engine_data.index[:-window+1]):
      engine_wins[i,:,:] = engine_data.loc[ind:ind+window-1,sensor_columns] # n timesteps of values for each sensor
      target[i] = engine_data.loc[ind+window-1,:]['RUL'] # target at end of window.


    if 'X_train' in locals(): # if X,y already exist, concatenate in new rows, otherwise intialize with first engine data
      X_train = np.concatenate((X_train,engine_wins),axis=0)
      y_train = np.vstack((y_train,target))
    else:
      print('creating')
      X_train = engine_wins
      y_train = target

    count += 1
    if count%10==0:
      print(count,' engines done')

  # making 3d numpy array for LSTM input
  for engine_num in val_engines: # for each engine
      engine_data = df.loc[df['unit_number']==engine_num,sensor_columns+['RUL']] # individual engine dataframe
      engine_wins = np.zeros((engine_data.shape[0]-window+1,window,len(sensor_columns))) # initializing 3d array for individual engine
      target = np.zeros((engine_data.shape[0]-window+1,1)) # initializing array for target for individual engine
      for i,ind in enumerate(engine_data.index[:-window+1]):
        engine_wins[i,:,:] = engine_data.loc[ind:ind+window-1,sensor_columns] # n timesteps of values for each sensor
        target[i] = engine_data.loc[ind+window-1,:]['RUL'] # target at end of window.


      if 'X_val' in locals(): # if X,y already exist, concatenate in new rows, otherwise intialize with first engine data
        X_val = np.concatenate((X_val,engine_wins),axis=0)
        y_val = np.vstack((y_val,target))
      else:
        print('creating')
        X_val = engine_wins
        y_val = target


  # changing y_val and y_train to three classes:
  if classification:
    y_val = RUL_class(y_val)
    y_train = RUL_class(y_train)

  return X_train,y_train,X_val,y_val,df,stscalerdict
