import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import pickle

#!pip install plotly

st.title('Heart Disease or Attack Checkup')
st.sidebar.header('Patient Data')

data_cols = ['Diabetes', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker','Stroke', 'PhysActivity', 'Fruits','Veggies', 'Alcohol_Consumption', 'Healthcare_Coverage', 'Consulted_Doc','GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age','Education', 'Income']
len(data_cols)
st.write("""
# Heart Heart Disease or Attack Prediction App

This app predicts whether the patient is suffering from heart related disease or heart attack using diabetes data.
""")

uploaded_file = st.sidebar.file_uploader(
                                          "Upload your input CSV file",
                                           type=["csv"]
                                         )
st.subheader('Training Data Stats')


def user_input_features():
    Diabetes = st.sidebar.radio('Diabetes: 0:Non diabetic, 1:Prediabetic, 2:Diabetic',(0,1,2))
    HighBP = st.sidebar.radio('High Blood Pressure: 1:Yes, 0:No',(0,1))
    HighChol = st.sidebar.radio('High Cholestrol: 1:Yes, 0:No',(0,1))
    CholCheck = st.sidebar.radio('Cholestrol check in 5 years: 1:Yes, 0:No',(0,1))
    BMI = st.sidebar.slider('BMI', 0,100, 20 )
    Smoker = st.sidebar.radio('Smoker: 1:Yes, 0:No',(0,1))
    Stroke = st.sidebar.radio('Did you ever had a stroke: 1:Yes, 0:No',(0,1))
    PhysActivity = st.sidebar.radio('Physical Activity in past 30 days: 1:Yes, 0:No',(0,1))
    Fruits = st.sidebar.radio('Consume fruits atleast once a day: 1:Yes, 0:No',(0,1))
    Veggies = st.sidebar.radio('Consume vegetables atleast once a day: 1:Yes, 0:No',(0,1))
    Alcohol_Consumption = st.sidebar.radio('Consume alcohol atleast once per week: 1:Yes, 0:No',(0,1))
    Healthcare_Coverage = st.sidebar.radio('Do you have a healthcare coverage plan: 1:Yes, 0:No',(0,1))
    Consulted_Doc = st.sidebar.radio('Did you face difficulty in consulting the doctor: 1:Yes, 0:No',(0,1))
    GenHlth =  st.sidebar.slider('Your general health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor',1,5,3)
    MentHlth = st.sidebar.slider('for how many days during the past 30 days was your mental health not good? scale 1-30 days', 0,30,5)
    PhysHlth = st.sidebar.slider('for how many days during the past 30 days was your physical health not good? scale 1-30 days', 0,30,5)
    DiffWalk = st.sidebar.radio('Do you have serious difficulty walking or climbing stairs? 1:Yes, 0:No', (0,1))
    Sex =  st.sidebar.radio('Gender: Male:1, Female:0',(0,1))
    Age = st.sidebar.slider('Age: [1:18 to 24],[2:25 to 31],[3:32 to 38],[4:39 to 45],[5:46 to 52],[6:53 to 59],[7:60 to 66],[8:67 to 73],[9:74 to 80],[10:81 to 87],[11:88 to 90],[12:91 to 92],[13:92 or above]', 1,13, 4 )
    Education = st.sidebar.slider('Education level: scale 1-6 1 = Never attended school 2 = Grades 1-8 3 = Grades 9-11 4 = Grade 12 5 = College 1-3 years 6 = College 4+',1,6,3)
    Income = st.sidebar.slider('Income Level: scale 1-8 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more ',1,8,3)

    user_report_data = {
                        'Diabetes':Diabetes,
                        'HighBP':HighBP,
                        'HighChol':HighChol,
                        'CholCheck':CholCheck,
                        'BMI':BMI,
                        'Smoker':Smoker,
                        'Stroke':Stroke,
                        'PhysActivity':PhysActivity,
                        'Fruits':Fruits,
                        'Veggies':Veggies,
                        'Alcohol_Consumption':Alcohol_Consumption,
                        'Healthcare_Coverage':Healthcare_Coverage,
                        'Consulted_Doc':Consulted_Doc,
                        'GenHlth':GenHlth,
                        'MentHlth':MentHlth,
                        'PhysHlth':PhysHlth,
                        'DiffWalk':DiffWalk,
                        'Sex':Sex,
                        'Age':Age,
                        'Education':Education,
                        'Income':Income
                        }

    return user_report_data

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    input_df = pd.DataFrame( [tuple([np.nan] * len(data_cols))] , columns = data_cols )
    user_report_data = user_input_features()
    input_df['Diabetes'] =user_report_data['Diabetes']
    input_df['HighBP'] =user_report_data['HighBP']
    input_df['HighChol'] =user_report_data['HighChol']
    input_df['CholCheck'] =user_report_data['CholCheck']
    input_df['BMI'] =user_report_data['BMI']
    input_df['Smoker'] =user_report_data['Smoker']
    input_df['Stroke'] =user_report_data['Stroke']
    input_df['PhysActivity'] =user_report_data['PhysActivity']
    input_df['Fruits'] =user_report_data['Fruits']
    input_df['Veggies'] =user_report_data['Veggies']
    input_df['Alcohol_Consumption'] =user_report_data['Alcohol_Consumption']
    input_df['Healthcare_Coverage'] =user_report_data['Healthcare_Coverage']
    input_df['Consulted_Doc'] =user_report_data['Consulted_Doc']
    input_df['GenHlth'] =user_report_data['GenHlth']
    input_df['MentHlth'] =user_report_data['MentHlth']
    input_df['PhysHlth'] =user_report_data['PhysHlth']
    input_df['DiffWalk'] =user_report_data['DiffWalk']
    input_df['Sex'] =user_report_data['Sex']
    input_df['Age'] =user_report_data['Age']
    input_df['Education'] =user_report_data['Education']
    input_df['Income'] =user_report_data['Income']

st.write(input_df.describe())

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

load_clf = pickle.load(open('/home/ashok/Desktop/FA_Project/health_indicators_model.pkl', 'rb'))

prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('Your Report: ')
output=''
if prediction[0]==0:
  output = 'You are Healthy'
else:
  output = 'You may be having a heart disease or attack'

st.title(output)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.title('Visualised Patient Report')
if prediction[0]==0:
  color = 'blue'
else:
  color = 'red'

st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = input_df, palette='rainbow')
ax12 = sns.scatterplot(x = input_df['Age'], y = input_df['BMI'], s = 150, color = color)
plt.xticks(np.arange(0,13,1))
plt.yticks(np.arange(0,100,5))
st.pyplot(fig_bmi)

st.header('BMI Value Graph (Others vs Yours)')
fig_edu = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'Education', data = input_df, palette='rainbow')
ax12 = sns.scatterplot(x = input_df['Age'], y = input_df['Education'], s = 150, color = color)
plt.xticks(np.arange(0,13,1))
plt.yticks(np.arange(0,6,1))
st.pyplot(fig_edu)

st.header('BMI Value Graph (Others vs Yours)')
fig_inc = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'Income', data = input_df, palette='rainbow')
ax12 = sns.scatterplot(x = input_df['Age'], y = input_df['Income'], s = 150, color = color)
plt.xticks(np.arange(0,13,1))
plt.yticks(np.arange(0,8,1))
st.pyplot(fig_inc)
