import pandas as pd 
import numpy as np

data=pd.read_csv('auto-mpg.csv') 

#%% Veri Setimize Bakalım

#%% Info
info=data.info()

print(info)

#%% Head
head=data.head()

print(head)

#%% Missing Values

mis_val=data.isnull()

#%% Sadece sütunlar için bakıyoruz

mis_val1=data.isnull().any()

#%% Horsepower sütununda ki eksik verileri görüyoruz

print(data.horsepower.unique())

#%% Missing value ? >> NaN yapılması

data=data.replace('?',np.nan)

#%% Horsepower sütununu alıyoruz

horsepower=data.iloc[:,3:4].values

data=data.drop(['horsepower'],axis=1)

#%% Eksik verileri ortalama ile dolduruyruz

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

horsepower=imputer.fit_transform(horsepower)

#%% origine göre Usa,Europe ve Asia ekledik ve bunu one hot encoder yaptık

origin=data.pop('origin')

data['Usa'] = (origin == 1)*1.0
data['Europe'] = (origin == 2)*1.0
data['Asia'] = (origin == 3)*1.0

#%% Car Name sütununun ve origin sütununun silinmesi  

data=data.drop(['car name'],axis=1)

#%% Şimdi de horsepower ekledik

horsepower=pd.DataFrame(data=horsepower,index=range(398),columns=['horsepower'])

data=pd.concat([data,horsepower],axis=1)

#%% cylinders sütununun alınması

cylinders=data.iloc[:,1:2].values

#%% Train-Test olarak ayrılması

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,cylinders,test_size=0.33,random_state=0)

#%% Model Fit

from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)

logr.fit(x_train,y_train)

# Tahmin
pred=logr.predict(x_test)

#%% Logistic Regression Confusion Matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,pred)

#%% Logistic Regression R2 score

from sklearn.metrics import r2_score

score=r2_score(y_test,logr.predict(x_test))

print("Logistic Regression score:",score)

#%% KNN kullanıyoruz

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski')

knn.fit(x_train,y_train)

# KNN Tahmin
knn_pred=knn.predict(x_test)

#%% KNN Confusion Matrix

cm_knn=confusion_matrix(y_test,knn_pred)

#%% KNN R2 Score

knn_score=r2_score(y_test,knn_pred)

print("KNN r2 score:",knn_score)

#%% Support Vector Machine

from sklearn.svm import SVC

svc=SVC(kernel='linear')

svc.fit(x_train,y_train)

#%% SVC Tahmin

svc_pred=svc.predict(x_test)

#%% SVC Confusion Matrix

cm_svc=confusion_matrix(y_test,svc_pred)

#%% SVC R2 Score

svc_score=r2_score(y_test,svc_pred)

print("SVC Score:",svc_score)

#%% Decision Tree 

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion='entropy')

dtc.fit(x_train,y_train)

#%% Decision Tree Tahmin

dtc_pred=dtc.predict(x_test)

#%% Decision Tree confusion matrix

cm_dtc=confusion_matrix(y_test,dtc_pred)

#%% Decision Tree R2 Score

dtc_score=r2_score(y_test,dtc_pred)

print("DTC Score:",dtc_score)

#%% Random Forest

from sklearn.ensemble import RandomForestClassifier 

rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')

rfc.fit(x_train,y_train)

#%% Random Forest Tahmin

rfc_pred=rfc.predict(x_test)

#%% Random Forest Confusion Matrix

cm_rfc=confusion_matrix(y_test,rfc_pred)

#%% Random Forest R2 Score

rfc_score=r2_score(y_test,rfc_pred)

print("RFC Score:",rfc_score)













































