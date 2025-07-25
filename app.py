import streamlit as st
import tensorflow
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

model=load_model("model.h5")
with open("LabelEncoder_Gender.pkl","rb") as f1:
    Label_Encoder=pickle.load(f1)
with open("OneHotEncoder_Geography.pkl","rb") as f2:
    One_Hot_Encoder=pickle.load(f2)
with open("Scaler.pkl","rb") as f3:
    Scaler=pickle.load(f3)

st.title("Churn Prediction Model")
geography=st.selectbox("Select Region",One_Hot_Encoder.categories_[0])
credit_score=st.slider("Enter CreditScore",1,800)
gender=st.selectbox("Enter gender",Label_Encoder.classes_[0])
age=st.slider("Enter Age",1,60)
tenure=st.slider("Enter Tenure",1,20)
balance=st.number_input("Enter Balance")
no_of_products=st.selectbox("Enter No of Products",[1,2,3,4])
is_activeMember=st.selectbox("IsActiveMember",[0,1])
salary=st.number_input("Enter Salary")
has_card=st.selectbox("Has Card?",[0,1])




input_data = {
    'CreditScore':[credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [no_of_products],
    'HasCrCard': [has_card],
    'IsActiveMember': [is_activeMember],
    'EstimatedSalary': [salary]
}



ohe_encoded=One_Hot_Encoder.transform([[geography]])
encode_df=pd.DataFrame(ohe_encoded.toarray(),columns=One_Hot_Encoder.get_feature_names_out())

input_dataframe=pd.DataFrame(input_data)


input_dataframe['Gender']=Label_Encoder.transform(input_dataframe['Gender'])

final_df=pd.concat([input_dataframe.drop('Geography',axis=1),encode_df],axis=1)

scaled_input=Scaler.transform(final_df)

prediction=model.predict(scaled_input)

prediction_proba=prediction[0][0]

st.write("Churn Probablity",prediction_proba)

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')


