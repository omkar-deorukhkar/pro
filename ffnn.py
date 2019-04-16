import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import r2_score

def ffnn(epoch_no,pred,nif,orig,corr,main_op,main_nif,i_score):
  nif = nif.tolist()
  pred = pred.ravel()
  pred = pred.tolist()
  corr = corr.tolist()
  
  
  pred.append(main_op)
  nif.append(main_nif)
  corr.append(corr[len(corr)-1])
  
  pred = np.asarray(pred)
  nif = np.asarray(nif)
  corr = np.asarray(corr)
 
 
  pred = np.reshape(pred,(len(pred),1))
  nif = np.reshape(nif,(len(nif),1))
  corr = np.reshape(corr,(len(corr),1))
  
  print(pred.shape,nif.shape,corr.shape)
  fmatrix = np.concatenate([pred,nif,corr],axis=1)
  print(fmatrix.shape)
  ftrainer = fmatrix[:len(fmatrix)-1]
  print(ftrainer.shape)
  
  

  fscaler = MinMaxScaler()
  
  matrix_scaled = fscaler.fit_transform(ftrainer)
  matrix_scaledt = fscaler.fit_transform(fmatrix)
  print(matrix_scaled.shape)
  prices = fscaler.fit_transform(orig)

  model = Sequential()

  model.add(Dense(4, activation='linear'))
  #model.add(Dropout(0.2))
  model.add(Dense(6, activation='linear'))
 # model.add(Dropout(0.2))
  model.add(Dense(8, activation='linear'))
 # model.add(Dropout(0.2))
  model.add(Dense(1, activation='linear'))
  model.compile(loss='mse',optimizer='adam', batch_size=32, metrics = ['mse'])
  model.fit(matrix_scaled,prices,epochs=epoch_no, shuffle=True)
  
  #model.fit(matrix_scaled[:100],prices[:100],epochs=1,shuffle=True)

  final_pred = model.predict(matrix_scaledt)
  final_pred_a = fscaler.inverse_transform(final_pred)
  prices_a = fscaler.inverse_transform(prices)
  print(final_pred_a[len(final_pred_a)-2:])

  #plt.plot(final_pred_a, color='green')
  #plt.plot(prices_a, color='black')
  #plt.show()
  

  rsq = r2_score(final_pred_a[1:],prices_a)
  print('R-squared coefficient :',rsq)
  
  return final_pred_a, prices_a
  

#final_pred, prices_a = ffnn(pred,nif,orig,corr,main_op,forcst,i_score)
