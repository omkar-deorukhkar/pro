import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
import pandas as pd
import numpy as np
from nsepy import get_history
from sklearn.metrics import r2_score
from datetime import date

def arima_tsm():
  prices = get_history(symbol='NIFTY 100', start = date(2009,2,1), end = date.today(), index = True)
  prices = prices[['Close']]
  tenyears = prices

  prices = prices[len(prices)-500:len(prices)]

  #plt.plot(tenyears, label='NIFTY 100')
  #plt.title('NIFTY 100 10 years shows upward trend and seasonality')
  #plt.ylabel('NIFTY 100')
  #plt.xlabel('Year 2009-2019')
  #plt.legend()
  #plt.show()

  
  lenprice = len(prices)
  testval = prices.values


  lnprices = np.log(prices)


  acf_1 = acf(lnprices)
  ind = []
  for i in range(0,len(acf_1)):
    ind.append(i)
  
  #plt.bar(ind,acf_1, label='We can observe steady decrease in AC')
  #plt.title('ACF of Undifferenced Series')
  #plt.xlabel('Observation point')
  #plt.ylabel('Auto Covariance')
  #plt.legend()
  #plt.show()


  pacf_1 = pacf(lnprices)
  jnd =[]
  for j in range(0,len(pacf_1)):
    jnd.append(j)

  #plt.bar(jnd,pacf_1, label='We can Observe Sudden drop in PAC')
  #plt.title('PACF of Undifferenced Series')
  #plt.xlabel('Observation point')
  #plt.ylabel('Partial Auto Covariance')
  #plt.legend()
  #plt.show()

 
  ln_diff = lnprices - lnprices.shift()
  diff = ln_diff.dropna()
 
  
  #plot_acf(diff, lags=20)
  acf_2 = acf(diff)
  ind2 = []
  for i2 in range(0,len(acf_2)):
    ind2.append(i2)
  
  #plt.bar(ind2,acf_2,label = 'AC values for d=1, p=2')
  #plt.title('ACF plot for First Differencing')
  #plt.xlabel('Observation Points')
  #plt.ylabel('Auto Covariance Values')
  #plt.legend()
  #plt.show()

  pacf_2 = pacf(diff)
  jnd2 = []
  for j2 in range(0,len(pacf_2)):
    jnd2.append(j2)
  #plot_pacf(diff, lags=20)
  #plt.bar(jnd2,pacf_2,label = 'PAC values for d=1, q=2')
  #plt.title('PACF plot for First Differencing')
  #plt.xlabel('Observation Points')
  #plt.ylabel('Partial Auto Covariance Values')
  #plt.legend()
  #plt.show()

  price_matrix = lnprices.as_matrix()
  print(len(price_matrix))
  model = ARIMA(price_matrix, order = (2,1,2))
  model_fit = model.fit(disp=0)
  pred = model_fit.predict(lenprice-100,lenprice, typ='levels')
  pred_adj = np.exp(pred)
  
  #plt.plot(pred_adj, color='green', label='ARIMA(2,1,2) Forecast')
  #plt.plot(testval[lenprice-100:],color='grey', label='NIFTY-100')
  #plt.title('ARIMA(2,1,2) Forecast')
  #plt.xlabel('Observation points(Days)')
  #plt.ylabel('NIFTY-100')
  #plt.legend()
  #plt.show()
  

  rsq = r2_score(pred_adj,testval[lenprice-101:])
  print(rsq)
  err=[]
  for x1,x2 in zip(testval[lenprice-2:],pred_adj[len(pred_adj)-3:len(pred_adj)-1]):
    err.append(x1-x2)
  print(err)
  return pred_adj[len(pred_adj)-1], tenyears,diff, ind,acf_1,jnd,pacf_1, ind2,acf_2,jnd2,pacf_2,pred_adj,testval[lenprice-100:] #12
  
