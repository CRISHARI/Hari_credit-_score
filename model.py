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

data["Age"] = data["Age"].astype(int)
data["Num_of_Loan"] = data["Num_of_Loan"].astype(int)
data["Num_Bank_Accounts"] =data["Num_Bank_Accounts"].astype(int)
data["Credit_History_Age"] = data["Credit_History_Age"].astype(int)
data["Num_Credit_Inquiries"] = data["Num_Credit_Inquiries"].astype(int)
data["Num_Credit_Card"] = data["Num_Credit_Card"].astype(int)
data["Delay_from_due_date"] = data["Delay_from_due_date"].astype(int)
data["Num_of_Delayed_Payment"] = data["Num_of_Delayed_Payment"].astype(int)
data["Changed_Credit_Limit"] = data["Changed_Credit_Limit"].astype(int)

train=data.drop(["ID","Customer_ID","SSN","Month","Name","Type_of_Loan","Occupation",
                 "Amount_invested_monthly","Credit_Utilization_Ratio","Total_EMI_per_month","Monthly_Inhand_Salary"],axis=1)


CreditScore ={"Good" :0,"Poor":1,"Standard":2}
train["Credit_Score"] = train["Credit_Score"].map(CreditScore)
PaymentBehaviour={"Low_spent_Small_value_payments":5,"High_spent_Medium_value_payments":1,
                   "High_spent_Large_value_payments":0,"Low_spent_Medium_value_payments":4,
                   "High_spent_Small_value_payments":2,"Low_spent_Large_value_payments":3}
train["Payment_Behaviour"] =train["Payment_Behaviour"].map(PaymentBehaviour)
PaymentofMinAmount ={"Yes":2,"No":1,"NM":0}
train["Payment_of_Min_Amount"]=train["Payment_of_Min_Amount"].map(PaymentofMinAmount)
CreditMix = {"Standard":2,"Good":1,"Bad":0}
train["Credit_Mix"]=train["Credit_Mix"].map(CreditMix)


X = train.drop(["Credit_Score"],axis=1)
Y = pd.DataFrame(train["Credit_Score"])

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