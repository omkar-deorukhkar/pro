from nsepy import get_history
from datetime import date
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.core import Dense, Activation, Dropout

def lstmnn(tick,epoch_no):
    data = get_history(symbol=tick,start=date(2017,2,2), end = date.today())
    indx = get_history(symbol='NIFTY 100', start = date(2017,2,2), end = date.today(), index = True)
    
    data = data
    indx = indx
 
    data.reset_index(drop=True, inplace=True)
    indx.reset_index(drop=True, inplace=True)
    global nifty
    nifty = indx['Close']

    rolmean = data['Close'].rolling(5).mean()
    global corr
    corr = nifty.rolling(15).corr(data['Close'])

    data = data[20:]
    nifty = nifty[20:]
    rolmean = rolmean[20:]
    corr = corr[20:]

    cp = data['Close'].values
    op = data['Open'].values
    high = data['High'].values
    low = data['Low'].values
    rolmean = rolmean.values
    nifty = nifty.values


    def catcher(df1, df2):
      l = []
      for i in range(0,len(df1)):
        if np.absolute(df1[i]-df2[i]) < 0.0001:
          l.append(1)
        else:
          l.append(np.absolute(df1[i]-df2[i]))
      l = np.asarray(l)
      return l
    
    
    rng = catcher(high,low)
    cp_u = catcher(cp,rolmean)
    



    
    f1 = cp
    f2 = cp/op
    f3 = cp/cp_u
    f4 = cp/rng
    
  

    f1 = np.reshape(f1,(len(f1),1))
    f2 = np.reshape(f2,(len(f2),1))
    f3 = np.reshape(f3,(len(f3),1))
    f4 = np.reshape(f4,(len(f4),1))
    
    orig = f1


    features = np.concatenate([f1,f2,f3,f4], axis=1)

    
    
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    yscale = scaler.fit_transform(f1)



    future = 15

    X = []
    Y = []
  
    for i in range(future,len(features)):
      X.append(features[i-future:i])
      Y.append(yscale[i])
    nifty = nifty[15:]
    corr = corr[15:]
    orig = orig[15:]
    
    main_ip = []
    main_ip.append(features[len(features)-future:len(features)])
    main_ip = np.asarray(main_ip)
   
    
    
     

    X = np.asarray(X)
    Y = np.asarray(Y)
    
    Y = np.reshape(Y,(len(Y),1))

    X_train = X
    Y_train = Y
   
    
    X_test = X[len(X)-100:]
    Y_test = Y[len(X)-100:]
    



    nodes=200
    regressor = Sequential()
    regressor.add(LSTM(units = nodes,activation='linear',return_sequences = False, stateful=False,input_shape = (X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(4,activation='linear'))
    regressor.add(Dense(5,activation='linear'))
    regressor.add(Dense(6,activation='linear'))
    regressor.add(Dense(1,activation='linear'))

    regressor.compile(optimizer = 'adam', loss = 'mse', metrics=['mse'])
    history = regressor.fit(X_train, Y_train, epochs = epoch_no,shuffle=False  )
    

    yp_test = regressor.predict(X_test)
    yp_train = regressor.predict(X_train)
    main_op = regressor.predict(main_ip)
    ytr = Y_train
    yts = Y_test
 
    yp_testa = scaler.inverse_transform(yp_test)
    y_tsta = scaler.inverse_transform(yts)
    
    yp_traina = scaler.inverse_transform(yp_train)
    y_traina = scaler.inverse_transform(ytr)
    
    main_op_a = scaler.inverse_transform(main_op)

    #plt.plot(yp_traina, color='green', label='Predictions')
    #plt.plot(y_traina, label='Original Values')
    #plt.legend()
    #plt.show()
    main_corr = corr[len(corr)-1]
    return yp_traina,nifty,orig,corr, main_op_a, main_corr,history.history['loss'], y_traina 
#pred,nif,orig,corr, main_op, main_corr, hist_plot,y_traina = lstmnn(input(),10)