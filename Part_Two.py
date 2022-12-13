import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# Objective: Develop an effective marketing proposal that can predict social media usage based on insights derived from a classification model focused on consumer demographics and platform attributes.

# In[ ]:

# Q1) Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[54]:

s = pd.read_excel('./social_media_usage.xlsx')


# Q2) Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected.

# In[42]:

def clean_sm(df,col):
    data = np.where(df[col] == 1, 1, 0)
    return(data)


# In[46]:

data = {'educ2': [1, 2, 3], 'level': [0, 1, 2]}  
toydf = pd.DataFrame(data)  
print(toydf)    


# In[47]:

clean_sm(toydf,'educ2')

# Q3) Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable (that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[67]:

ss = pd.DataFrame({
    "sm_li":clean_sm(s,"web1h"),
    "Income":np.where(s["income"] <= 9,s["income"],np.nan),
    "Education":np.where(s["educ2"] <= 8,s["educ2"],np.nan),
    "Parent":np.where(s["par"] >= 1, 1, 0),
    "Married":np.where(s["marital"] == 1,1,0),
    "Female":np.where(s["gender"] == 2, 1, 0),
    "Age":np.where(s["age"] <= 98,s["age"],np.nan)}).dropna()    

# Q4) Create a target vector (y) and feature set (X)

# In[104]:

y = ss["sm_li"]
X= ss[["Income", "Education", "Parent", "Married", "Female", "Age"]]


# Q5)Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning.
# 
# Train (x,y) contains eighty percent of the database (including all features) used to refine / scale the predict model.
# Test (x,y) contains twenty percent of the database (including all features) used to evaluate the predictive model performance with new information.

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=987)

# Q6) Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# Initialize algorithm 
lr = LogisticRegression(class_weight = 'balanced')

# Fit algorithm to training data
lr.fit(X_train, y_train)

# Q7) Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

# Compare those predictions to the actual test data using a confusion matrix (positive class=1)
confusion_matrix(y_test, y_pred)


# Q8) Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

pd.DataFrame(confusion_matrix(y_test, y_pred),
             columns=["Predicted negative", "Predicted positive"],
             index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")


# Q9) Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# Accuracy: The number of times the model will correctly predict an outcome (either true positive or true negative) out of the total number of times it made predictions. This metric is useful when all categories are of equal importance but can be influenced (with bias) if one group has a disproportionate number of observations.

# Recall: Measures the proportion of correctly identified outputs with the objective to minimize the chance of missing positive cases. The metric emphasizes on how the positive observations are classified and is independent of how the negative observations are classified. This is important when testing for fraud, etc. Calculated as (true positive/(true positive + false negative).

# In[138]:

recall = ((62)/(62+22))

# Precision: Goal to minimize incorrectly predicting positive. This is helpful when you start with a business strategy (marketing / advertising project, political campaign, etc.). Calculated as (true positive/(true positive + false positive).

precision = ((62)/(62+58))

# F1 Score: The weighted average of recall and precision calculated (2* precision * recall) / (precision + recall). The metric is preferred over accuracy when the classification groups are unbalanced. The F1 score takes into consideration the data distribution but can harder to explain / interpret (vs. accuracy). This metric is useful when testing for health reasons (pregnancy, cancer, etc.).

F1 = (2 * precision * recall) / (precision + recall)

print(classification_report(y_test, y_pred))

# Q10) Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# Based on the established model, age is a key influential factor in determining whether an individual is a LinkedIn user (first participant has a 72% probability and the second individual has a 45% probability).

#new_data = pd.DataFrame({
    #"Income":[8,8],
    #"Education":[7,7],
    #"Parent":[0,0],
    #"Married":[1,1],
    #"Female":[2,2],
    #"Age":[42,82]
#})   

#new_data

#new_data["prediction_LinkendIn_User"] = lr.predict(new_data)

#person1 = [8, 7, 0, 1, 2, 42]
#person2 = [8, 7, 0, 1, 2, 82]

#predicted_class1 = lr.predict([person1])
#predicted_class2 = lr.predict([person2])

#probs1 = lr.predict_proba([person1])
#probs2 = lr.predict_proba([person2])

#print(f"Probability that this person is LinkedIn User: {probs1[0][1]}")
#print(f"Probability that this person is LinkedIn User: {probs2[0][1]}")

######### Part Two - Predictive Model Survey

st.write("LinkedIn User Prediction Model")

st.write("Thank you for participating in this brief survey.")

st.write("Created by Matthew Booth")
        
st.write("Disclaimer: All information will remain private and will not be stored / tracked.")

if __name__ == '__main__':
    Age = st.slider("Please Enter Your Age", min_value=1, max_value=98, value=18, step=1)

    Gender = st.radio("Please Select Your Gender Identity", ['Male', 'Female', 'Non-Binary / Non-Comforming', 'Do Not Know', 'Refused'], horizontal=True)
    Gender_dict = {'Male':1, 'Female':2, 'Non-Binary / Non-Comforming':3, 'Do Not Know':98, 'Refused':99}    
    Gender_category = Gender_dict.get(Gender)

    Parental_Status = st.radio("Please Select Your Current Parental Status", ['Yes', 'No', 'Do Not Know', 'Refused'], horizontal=True)
    Parental_dict = {'Yes':1, 'No':2, 'Do Not Know':8, 'Refused':9}    
    Parental_category = Parental_dict.get(Parental_Status)

    Marital = st.selectbox("Please Select Your Current Martial Status",['Married', 'Living with a Partner', 'Divorced',
                                                               'Separated', 'Widowed', 'Never Been Married',
                                                               'Do Not Know', 'Refused'])
    Marital_dict = {'Married':1, 'Living with a Partner':2, 'Divorced':3,'Separated':4, 'Widowed':5, 'Never Been Married':6,
                                                               'Do Not Know':8, 'Refused':9}
    Marital_category = Marital_dict.get(Marital)

    Income = st.selectbox("Please Select Your Annual Household Income",['Less than $10,000', '10 to under $20,000', '20 to under $30,000',
                                                               '30 to under $40,000', '40 to under $50,000', '50 to under $75,000', '75 to under $100,000','100 to under $150,000, OR',
                                                               '$150,000 or more?',
                                                               'Do Not Know', 'Refused'])
    Income_dict = {'Less than $10,000':1, '10 to under $20,000':2, '20 to under $30,000':3,
                                                               '30 to under $40,000':4, '40 to under $50,000':5, '50 to under $75,000':6, '75 to under $100,000':7,'100 to under $150,000, OR':8,
                                                               '$150,000 or more?':9,
                                                               'Don Not Know':98, 'Refused':99}
    Income_category = Income_dict.get(Income)

    Education = st.selectbox("Please Select Highest Level of Education",['Less than High School', 'High School Incomplete', 'High School Graduate',
                                                               'Some College, No Degree', 'Two-year Associate Degree', 'Bachelor’s Degree', 'Some Postgraduate or Professional Schooling','Postgraduate or Professional Degree',
                                                               'Do Not Know', 'Refused'])
    Education_dict = {'Less than High School':1, 'High School Incomplete':2, 'High School Graduate':3,
                                                               'Some College, No Degree':4, 'Two-year Associate Degree':5, 'Bachelor’s Degree':6, 'Some Postgraduate or Professional Schooling':7,'Postgraduate or Professional Degree':8,
                                                               'Do Not Know':98, 'Refused':99}
    Education_category = Education_dict.get(Education)

    user_data = pd.DataFrame({
    "Income":[Income_category],
    "Education":[Education_category],
    "Parent":[Parental_category],
    "Married":[Marital_category],
    "Female":[Gender_category],
    "Age":[Age]
})  
    
#user_data["prediction_LinkendIn_User"] = lr.predict(user_data)
#results = print(f"Probability that You Are a LinkedIn User: {probsTBD[0][1]}")
#st.dataframe(results)

user1 = [Age, Income_category, Education_category, Parental_category, Marital_category, Gender_category]
predicted_classTBD = lr.predict([user1])
probsTBD = lr.predict_proba([user1])
st.write(f"Classifcation that you are a LinkedIn user: {predicted_classTBD}")
st.write(f"Probability that you are a LinkedIn user: {probsTBD[0][1]}")

st.write("Thank you for participating in this survey.")