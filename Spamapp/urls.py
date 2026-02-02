from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
               path("LoadDataset", views.LoadDataset, name="LoadDataset"),
	       path("TrainModels", views.TrainModels, name="TrainModels"),
	       path("SMSPredict.html", views.SMSPredict, name="SMSPredict"),
               path("SMSPredictAction", views.SMSPredictAction, name="SMSPredictAction"),	
	       path("URLPredict.html", views.URLPredict, name="URLPredict"),
               path("URLPredictAction", views.URLPredictAction, name="URLPredictAction"),
]
