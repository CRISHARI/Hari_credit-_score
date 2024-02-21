import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pickle
import gzip

data= pd.read_csv("credit.csv")
data= data.loc[data['Age']>17]#removing customers under Age 18
data= data.loc[data['Annual_Income']>10000]

data["Age"] = data["Age"].astype(int)
data["Num_of_Loan"] = data["Num_of_Loan"].astype(int)
data["Num_Bank_Accounts"] =data["Num_Bank_Accounts"].astype(int)
data["Credit_History_Age"] = data["Credit_History_Age"].astype(int)
data["Num_Credit_Inquiries"] = data["Num_Credit_Inquiries"].astype(int)
data["Num_Credit_Card"] = data["Num_Credit_Card"].astype(int)
data["Delay_from_due_date"] = data["Delay_from_due_date"].astype(int)
data["Num_of_Delayed_Payment"] = data["Num_of_Delayed_Payment"].astype(int)
data["Changed_Credit_Limit"] = data["Changed_Credit_Limit"].astype(int)

data1=data.drop(["ID","Customer_ID","SSN","Month","Name","Type_of_Loan","Occupation",
                 "Amount_invested_monthly","Credit_Utilization_Ratio","Total_EMI_per_month","Monthly_Inhand_Salary"],axis=1)


#label encoding
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data1['Credit_Score'] = le.fit_transform(data1['Credit_Score'])
#data1['Occupation'] = le.fit_transform(data1['Occupation'])
data1['Payment_of_Min_Amount'] = le.fit_transform(data1['Payment_of_Min_Amount'])
data1['Credit_Mix'] = le.fit_transform(data1['Credit_Mix'])
data1['Payment_Behaviour'] = le.fit_transform(data1['Payment_Behaviour'])


X = data1.drop(["Credit_Score"],axis=1)
Y = pd.DataFrame(data1["Credit_Score"])

smote = SMOTE()
X,Y = smote.fit_resample(X,Y)



scaler_x = MinMaxScaler()
X_scale= scaler_x.fit_transform(X)

Y= np.squeeze(Y)

x_train,x_test,y_train,y_test = train_test_split(X_scale,Y, test_size = 0.3, random_state=42)

rf_cls =RandomForestClassifier()

model_rf = rf_cls.fit(x_train,y_train)

y_pred_rf = model_rf.predict(x_test)

pipeline = make_pipeline(MinMaxScaler(),RandomForestClassifier())

model = pipeline.fit(x_train, y_train)
y_pred = model.predict(x_test)

filename="model.pickle"
with gzip.open(filename,"wb") as file:
   pickle.dump(model,file)