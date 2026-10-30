import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Food Delivery Time Prediction")

order_time = pd.read_csv(r"C:\Users\sriva\Desktop\ml_models\food_delivery_time_prediction\FOOD_DELIVERY_REPORT")

X = order_time[['City','Distance_km','Traffic_Level','Weather_Conditions','Time_of_Day','Restaurant_Preparation_Time_min','Number_of_Items','Restaurant_Active_Orders','Order_Value_INR']].copy()

X['City'] = X['City'].map({'Bengaluru':0,'Kolkata':1,'Hyderabad':2,'Pune':3,'Mumbai':4,'Delhi':5,'Chennai':6})

X['Traffic_Level'] = X['Traffic_Level'].map({'Low':0,'Medium':1,'High':2,'Gridlock':3})

X['Weather_Conditions'] = X['Weather_Conditions'].map({'Sunny':0,'Cloudy':1,'Rainy':2})

X['Time_of_Day'] = X['Time_of_Day'].map({'Morning':0,'Afternoon':1,'Night':2})

X = X.fillna(0)

y = order_time['Delivery_Time_Minutes']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

model = RandomForestRegressor(n_estimators=100,max_depth=5,random_state=42)

model.fit(x_train,y_train)

CT = st.number_input("Enter City 0.Bengaluru 1.Kolkata 2.Hyderabad 3.Pune 4.Mumbai 5.Delhi 6.Chennai",step=1)

D = st.number_input("Enter Distance in KM",step=1)

TL = st.number_input("Enter Traffic Level 0.Low,1.Medium,2.High,3.Gridlock")

WC = st.number_input("Enter Weather 0.Sunny,1.Cloudy,2.Rainy")

TOD = st.number_input("Enter Time of Day 0.Morning,1.Afternoon,2.Night")

RPT = st.number_input("Enter Restaurant Preparation Time")

NI = st.number_input("Enter Number of Items")

RAO = st.number_input("Enter Restaurant Active Orders")

OV = st.number_input("Enter Order Value")

bt = st.button("Submit")

if bt:

    output = model.predict([[CT,D,TL,WC,TOD,RPT,NI,RAO,OV]])

    st.write("Estimated Delivery Time:",output[0],"minutes")

train_output = model.predict(x_train)

test_output = model.predict(x_test)

st.write("Train R2:",r2_score(y_train,train_output))

st.write("Test R2:",r2_score(y_test,test_output))

st.write("Test MAE:",mean_absolute_error(y_test,test_output))

st.write("Test RMSE:",mean_squared_error(y_test,test_output)**0.5)