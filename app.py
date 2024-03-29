import numpy as np
from flask import Flask, request, render_template
import gzip
import pickle
app = Flask(__name__)
with gzip.open("model.pickle","rb") as file:
    load_model = pickle.load(file)
@app.route('/')
def index():

    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
 
        if request.method == 'POST':
            
           
           Age = float(request.form["Age"])
           Annual_Income =float(request.form["Annual_Income"])
           Num_Bank_Accounts =float(request.form["Num_Bank_Accounts"])
           Num_Credit_Card = float(request.form["Num_Credit_Card"])
           Num_of_Loan = float(request.form["Num_of_Loan"])
           Interest_Rate = float(request.form["Interest_Rate"])
           Monthly_Balance = float(request.form["Monthly_Balance"])
           Outstanding_Debt =float(request.form["Outstanding_Debt"])
           Delay_from_due_date = float(request.form["Delay_from_due_date"])
           Num_of_Delayed_Payment = float(request.form["Num_of_Delayed_Payment"])
           Changed_Credit_Limit =float(request.form["Changed_Credit_Limit"])
           Num_Credit_Inquiries = float(request.form["Num_Credit_Inquiries"])
           Credit_History_Age = float(request.form["Credit_History_Age"])
           Payment_of_Min_Amount = request.form["Payment_of_Min_Amount"]
           Payment_Behaviour =request.form["Payment_Behaviour"]
           Credit_Mix = request.form["Credit_Mix"]
           
           
           prediction = load_model.predict(np.array([[Annual_Income,Age,Num_Bank_Accounts,Num_Credit_Card,
                                              Num_of_Loan,Interest_Rate,Monthly_Balance,Outstanding_Debt,
                                              Delay_from_due_date,Num_of_Delayed_Payment,Credit_Mix,
                                              Changed_Credit_Limit,Num_Credit_Inquiries,Credit_History_Age,
                                              Payment_Behaviour,Payment_of_Min_Amount]]))[0]
                                             
           output = int(prediction)
           if (output == 0):
         
               return render_template("result.html",prediction_text=" predicted Credit Score is   Good")
           
           elif (output==1):
                return render_template("result.html",prediction_text="predicted Credit Score is  Poor")
        
           else :
                return render_template("result.html",prediction_text= " predicted Credit Score is  Standard")
                                        
                                 
          

           
        # Handle the case where the input is not a valid float
        return "Invalid input. Please enter a valid value in the  fields."

if __name__ == '__main__':
    app.run(debug=True)