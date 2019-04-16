from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

def sentiment_analysis(kw):
  kw = kw + ' share'
  req1 = 'https://www.google.com/search?q={0}&source=lnms&tbm=nws'.format(kw)
  print(kw)
  req2 = 'https://www.google.com/search?q={0}&source=lnms&tbm=nws'.format('NSE')
  req3 = 'https://www.google.com/search?q={0}&source=lnms&tbm=nws'.format('India')
  
  page1 = requests.get(req1)
  page2 = requests.get(req2)
  page3 = requests.get(req3)

  soup1 = BeautifulSoup(page1.text, 'html.parser')
  soup2 = BeautifulSoup(page2.text, 'html.parser')
  soup3 = BeautifulSoup(page3.text, 'html.parser')
  
  para1 = soup1.find_all('a')
  para2 = soup2.find_all('a')
  para3 = soup3.find_all('a')
  
  words1 = []
  words2 = []
  words3 = []
  
  for p in para1[27:50]:
    if len(str(p.text)) > 1:
      words1.append(str((p.text)))
      
  for p in para2[27:50]:
    if len(str(p.text)) > 1:
      words2.append(str((p.text)))
      
  for p in para3[27:50]:
    if len(str(p.text)) > 1:
      words3.append(str((p.text)))

  analyser = SentimentIntensityAnalyzer()
  

  score1=0
  score2=0
  score3=0
  
  for w in words1:
    score1 += (analyser.polarity_scores(w)['compound'])
  for w in words2:
    score2 += (analyser.polarity_scores(w)['compound'])
  for w in words3:
    score3 += (analyser.polarity_scores(w)['compound'])
  print(len(words1),len(words2),len(words3))
  
  scaled_score1, scaled_score2, scaled_score3 = score1/len(words1), score2/len(words2), score3/len(words3)
  final_score = (0.1*scaled_score1 + 0.2*scaled_score2 + 0.7*scaled_score3)/3
  
  return final_score 

print(sentiment_analysis('mrf'))