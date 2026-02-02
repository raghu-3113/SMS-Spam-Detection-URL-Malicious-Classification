from sentence_transformers import SentenceTransformer #loading bert sentence model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

#loading & initializing BERT model for news embedding
bert = SentenceTransformer('bert-base-multilingual-cased')
print("BERT initialization completed")

dataset = pd.read_csv('Dataset/data-en-hi-de-fr.csv')
dataset = dataset.values

if os.path.exists("model/en_X.npy"):
    print("loading")
    en_X = np.load("model/en_X.npy")
    hi_X = np.load("model/hi_X.npy")
    Y = np.load("model/Y.npy")
else:
    en_X = []
    Y = []
    hi_X = []
    for i in range(len(dataset)): 
        label = str(dataset[i,0])
        english = str(dataset[i,1])
        hindi = str(dataset[i,2])
        label = label.strip().lower()
        english = english.strip().lower()
        hindi = hindi.strip()
        if len(label) > 0 and len(english) > 0:
            english = re.sub('[^a-z]+', ' ', english)#clean news data
            en_X.append(english)
            hi_X.append(hindi)
            if label == "ham":
                Y.append(0)
            else:
                Y.append(1)
            print(str(i)+" "+label+" "+english+" "+hindi)    
    en_X = np.asarray(en_X)
    hi_X = np.asarray(hi_X)                        
    Y = np.asarray(Y)
    np.save("model/en_X", en_X)
    np.save("model/hi_X", hi_X)
    np.save("model/Y", Y)
    embeddings = bert.encode(en_X, convert_to_tensor=True)#apply bert on news data to start embedding
    en_X = embeddings.numpy()
    np.save("model/en_X", en_X)
    np.save("model/hi_X", hi_X)
       
print("BERT vector = "+str(en_X))

if os.path.exists("model/models.pckl"):
    f = open('model/models.pckl', 'rb')
    models = pickle.load(f)
    f.close()
    en_rf, hi_rf = models
else:
    en_rf = RandomForestClassifier()
    en_rf.fit(X_train, y_train)
    hi_rf = RandomForestClassifier()
    hi_rf.fit(X_train, y_train)
    models = [en_rf, hi_rf]
    f = open('model/models.pckl', 'wb')
    pickle.dump(models, f)
    f.close()
X_train, X_test, y_train, y_test = train_test_split(en_X, Y, test_size=0.2)
predict = en_rf.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)
X_train, X_test, y_train, y_test = train_test_split(hi_X, Y, test_size=0.2)
predict = hi_rf.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)


def prediction(msg, model):
    embeddings = bert.encode([msg], convert_to_tensor=True)#apply bert on news data to start embedding
    msg = embeddings.numpy()
    predict = model.predict(msg)
    print(predict)

test_data = pd.read_csv("Dataset/test_sms.csv")
test_data = test_data.values

for i in range(len(test_data)):
    if i == 0:
        msg = test_data[i, 0]
        msg = msg.strip().lower()
        msg = re.sub('[^a-z]+', ' ', msg)#clean news data
        print("eng "+msg)
        prediction(msg, en_rf)
    elif i == 1:
        msg = test_data[i,0]
        print("hi "+msg)
        prediction(msg, hi_rf)
    elif i > 1 and (i % 2) == 0:
        msg = test_data[i, 0]
        msg = msg.strip().lower()
        msg = re.sub('[^a-z]+', ' ', msg)#clean news data
        print("eng "+msg)
        prediction(msg, en_rf)
    elif i > 1 and (i % 2) == 1:
        msg = test_data[i,0]
        print("hi "+msg)
        prediction(msg, hi_rf)
        










