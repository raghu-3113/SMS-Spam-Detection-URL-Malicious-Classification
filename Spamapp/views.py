from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from sentence_transformers import SentenceTransformer #loading bert sentence model
import os
import pickle
import os
from django.core.files.storage import FileSystemStorage
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import os
import re
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import urllib
from urllib.parse import urlparse
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.metrics import confusion_matrix

global uname, graph, en_rf, hi_rf
#loading & initializing BERT model for news embedding
bert = SentenceTransformer('bert-base-multilingual-cased')
print("BERT initialization completed")

accuracy = []
precision = []
recall = [] 
fscore = []

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    global graph
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

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
conf_matrix = confusion_matrix(y_test, predict)
calculateMetrics("MBERT Spam Detection", y_test, predict)
X_train, X_test, y_train, y_test = train_test_split(hi_X, Y, test_size=0.2)
predict = hi_rf.predict(X_test)
calculateMetrics("Random Forest URL Classification", y_test, predict)   
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


def TrainModels(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, conf_matrix
        labels = ['Ham', 'Spam']
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
        output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
        output+='</tr>'
        algorithms = ['MBERT Spam Detection', 'Random Forest URL Classification']
        for i in range(len(algorithms)):
            output += '<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([['MBERT Spam Detection','Accuracy',accuracy[0]],['MBERT Spam Detection','Precision',precision[0]],['MBERT Spam Detection','Recall',recall[0]],['MBERT Spam Detection','FSCORE',fscore[0]],
                           ['Random Forest URL Classification','Accuracy',accuracy[1]],['Random Forest URL Classification','Precision',precision[1]],['Random Forest URL Classification','Recall',recall[1]],['Random Forest URL Classification','FSCORE',fscore[1]],
                          ],columns=['Parameters','Algorithms','Value'])

        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Performance Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(labels)])    
        df.pivot("Parameters", "Algorithms", "Value").plot(ax=axis[1], kind='bar')
        plt.title("All Algorithms Performance Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def prediction(msg, model):
    embeddings = bert.encode([msg], convert_to_tensor=True)#apply bert on news data to start embedding
    msg = embeddings.numpy()
    predict = model.predict(msg)
    output = "<font size=4 color=green>HAM</option>"
    if predict[0] == 1:
        output = "<font size=4 color=red>SPAM</option>"
    return output    

def SMSPredictAction(request):
    if request.method == 'POST':
        global bert, en_rf, hi_rf        
        msg = request.POST.get('t1', False)
        lang = request.POST.get('t2', False)
        data = msg
        output = None
        if lang == "English":
            msg = msg.strip().lower()
            msg = re.sub('[^a-z]+', ' ', msg)#clean news data
            output = prediction(msg, en_rf)
        else:
            output = prediction(msg, hi_rf)
        context= {'data':'SMS Predicted As : '+output}
        return render(request, 'SMSPredict.html', context)

def URLPredictAction(request):
    if request.method == 'POST':
        global xgb_cls
        testURL = request.POST.get('t1', False)
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
        output = ""
        if predict == 0:
            output += testURL+" <font size=3 color=green>====> Predicted AS Normal Link</font>"
        else:
            output += testURL+" <font size=3 color=red>====> Predicted AS Malicious Link</font>"
        context= {'data':output}
        return render(request, 'URLPredict.html', context)    

def URLPredict(request):
    if request.method == 'GET':
        return render(request, 'URLPredict.html', {}) 

def SMSPredict(request):
    if request.method == 'GET':
        return render(request, 'SMSPredict.html', {})    

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})    

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})   

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == "admin" and password == "admin":
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)        
    

def LoadDataset(request):
    if request.method == 'GET':
        global X, Y, X1, Y1
        output = "Total Records found in Dataset = "+str(en_X.shape[0])+"<br/>"
        output += "<br/>Labels found in Dataset = Ham & Spam<br/>"
        output += "<br/>Dataset Train & Test Split Details<br/>"
        output += "80% records using to train Algorithms : "+str(X_train.shape[0])+"<br/>"
        output += "20% records using to test Algorithms : "+str(X_test.shape[0])+"<br/><br/>"
        dataset = pd.read_csv("Dataset/data-en-hi-de-fr.csv", usecols=['labels','text','text_hi'])
        columns = dataset.columns
        dataset = dataset.values
        output+='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(dataset)):
            output += '<tr>'
            for j in range(len(dataset[i])):
                output += '<td><font size="3" color="black">'+str(dataset[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'UserScreen.html', context)      

