import senti
import lstm_nn
import arima_tsm
import ffnn
import matplotlib.pyplot as plt
from math import floor,ceil
from matplotlib.pyplot import figure
from matplotlib.font_manager import FontProperties
plt.style.use('ggplot')


def master_process(tick):
	pred,nif,orig,corr, main_op, main_corr, hist_plot,y_traina = lstm_nn.lstmnn(tick,5)
	forcst, tenyears,diff, ind,acf_1,jnd,pacf_1, ind2,acf_2,jnd2,pacf_2,pred_adj,testval = arima_tsm.arima_tsm()
	score_senti = senti.sentiment_analysis(tick)

	final_pred, prices_a = ffnn.ffnn(40,pred,nif,orig,corr,main_op,forcst,score_senti)

	figure(figsize=(40,15))

	plt.subplot(2,4,1)
	plt.title('LSTM Loss MSE')
	plt.plot(hist_plot,color='red',label='loss MSE')
	plt.xlabel('Epoch Number')
	plt.ylabel('Loss')
	plt.legend()

	plt.subplot(2,4,2)
	plt.title('NIFTY 100 10 years')
	ty = tenyears.values
	plt.plot(ty,color='blue',label='NIFTY-100 shows upward trend and seasonality')
	plt.xlabel('Observation Points (DAYS)')
	plt.ylabel('NIFTY-100')
	plt.legend()


	plt.subplot(2,4,3)

	plt.bar(ind,acf_1,color='blue', label='We can observe steady decrease in AC')
	plt.title('ACF of Undifferenced Series')
	plt.xlabel('Observation point')
	plt.ylabel('Auto Covariance')
	plt.legend()


	plt.subplot(2,4,4)
	plt.bar(jnd,pacf_1,color='blue', label='We can Observe Sudden drop in PAC')
	plt.title('PACF of Undifferenced Series')
	plt.xlabel('Observation point')
	plt.ylabel('Partial Auto Covariance')
	plt.legend()

	plt.subplot(2,4,5)

	plt.bar(ind2,acf_2,color='green',label = 'AC values for d=1, p=2')
	plt.title('ACF plot for First Differencing')
	plt.xlabel('Observation Points')
	plt.ylabel('Auto Covariance Values')
	plt.legend()



	plt.subplot(2,4,6)


	plt.bar(jnd2,pacf_2,color='green',label = 'PAC values for d=1, q=2')
	plt.title('PACF plot for First Differencing')
	plt.xlabel('Observation Points')
	plt.ylabel('Partial Auto Covariance Values')
	plt.legend()

	plt.subplot(2,4,7)

	plt.plot(pred_adj, color='green', label='ARIMA(2,1,2) Forecast')
	plt.plot(testval,color='grey', label='NIFTY-100')
	plt.title('ARIMA(2,1,2) Forecast')
	plt.xlabel('Observation points(Days)')
	plt.ylabel('NIFTY-100')
	plt.legend()

	plt.subplot(2,4,8)
	plt.plot(final_pred, color='green',label='predicted')
	plt.plot(prices_a, color='black',label='actual')
	plt.xlabel('Observation Points (DAYS)')
	plt.ylabel('Value')
	plt.legend()

	plt.suptitle('Graphs', fontsize=20)

	plt.savefig('Output.png')
	plt.close()
