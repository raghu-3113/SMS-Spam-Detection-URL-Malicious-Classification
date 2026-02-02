import pandas as pd
import numpy as np
import urllib
from urllib.parse import urlparse
from sklearn.preprocessing import normalize
import pickle
from xgboost import XGBClassifier

#function to convert URL into features like number of slash occurence, dot and other characters
def get_features(df):
    needed_cols = ['url', 'domain', 'path', 'query', 'fragment']
    for col in needed_cols:
        df[f'{col}_length']=df[col].str.len()
        df[f'qty_dot_{col}'] = df[[col]].applymap(lambda x: str.count(x, '.'))
        df[f'qty_hyphen_{col}'] = df[[col]].applymap(lambda x: str.count(x, '-'))
        df[f'qty_slash_{col}'] = df[[col]].applymap(lambda x: str.count(x, '/'))
        df[f'qty_questionmark_{col}'] = df[[col]].applymap(lambda x: str.count(x, '?'))
        df[f'qty_equal_{col}'] = df[[col]].applymap(lambda x: str.count(x, '='))
        df[f'qty_at_{col}'] = df[[col]].applymap(lambda x: str.count(x, '@'))
        df[f'qty_and_{col}'] = df[[col]].applymap(lambda x: str.count(x, '&'))
        df[f'qty_exclamation_{col}'] = df[[col]].applymap(lambda x: str.count(x, '!'))
        df[f'qty_space_{col}'] = df[[col]].applymap(lambda x: str.count(x, ' '))
        df[f'qty_tilde_{col}'] = df[[col]].applymap(lambda x: str.count(x, '~'))
        df[f'qty_comma_{col}'] = df[[col]].applymap(lambda x: str.count(x, ','))
        df[f'qty_plus_{col}'] = df[[col]].applymap(lambda x: str.count(x, '+'))
        df[f'qty_asterisk_{col}'] = df[[col]].applymap(lambda x: str.count(x, '*'))
        df[f'qty_hashtag_{col}'] = df[[col]].applymap(lambda x: str.count(x, '#'))
        df[f'qty_dollar_{col}'] = df[[col]].applymap(lambda x: str.count(x, '$'))
        df[f'qty_percent_{col}'] = df[[col]].applymap(lambda x: str.count(x, '%'))

with open('model/xgb.txt', 'rb') as file:
    xgb_cls = pickle.load(file)
file.close()

while True:
    #exexute this block to enter test URL and then extension XGBOOST will predict weather URL is leitimate or Phishing
    testURL = input("Enter URL : ")
    print(testURL)
    test = []
    test.append([testURL])
    data = pd.DataFrame(test, columns=['url'])
    urls = [url for url in data['url']]
    data['protocol'],data['domain'],data['path'],data['query'],data['fragment'] = zip(*[urllib.parse.urlsplit(x) for x in urls])
    get_features(data)
    data = data.drop(columns=['url', 'protocol', 'domain', 'path', 'query', 'fragment'])
    data = data.values
    data = normalize(data)
    temp = []
    temp.append(data)
    predict = xgb_cls.predict(data)[0]
    if predict == 0:
        print(testURL+" ====> Predicted AS Normal Link")
    else:
        print(testURL+" ====> Predicted AS Malicious Link")
