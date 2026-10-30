
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.markdown("""
<style>

.stApp {
    background-color: #FFF4E6;
}
.block-container {
    background-color: rgba(255, 255, 255, 0.75);
    padding: 40px;
    border-radius: 15px;
}
h1 {
    color: #704214;
    text-align: center;
}
h1, h2, h3 {
    color: #000000 !important;
    font-weight: bold;
    padding:20
}
label {
    color: #704214 !important;
    font-weight: bold;
}
.stButton > button {
    background-color: #B87945;
    color: white;
    border-radius: 8px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)
# order_time = pd.read_csv(r"C:\Users\sriva\Desktop\ml_models\food_delivery_time_prediction\FOOD_DELIVERY_REPORT")
order_time = pd.read_csv(r"FOOD_DELIVERY_REPORT.csv")
# order_time = pd.read_csv(
#     r"C:\Users\sriva\Desktop\ml_models\food_delivery_time_prediction\FOOD_DELIVERY_REPORT.csv"
# )

x = order_time[['City','Distance_km','Traffic_Level','Weather_Conditions','Time_of_Day','Restaurant_Preparation_Time_min','Number_of_Items','Restaurant_Active_Orders','Order_Value_INR']].copy()

y = order_time['Delivery_Time_Minutes']

city_mapping = {'Bengaluru': 0,'Kolkata': 1,'Hyderabad': 2,'Pune': 3,'Mumbai': 4,'Delhi': 5,'Chennai': 6}

traffic_mapping = {'Low': 0,'Medium': 1,'High': 2,'Gridlock': 3}

# weather_mapping = {'Sunny': 0,'Cloudy': 1,'Rainy': 2,'Foggy': 3,'Overcast': 4,'Stormy': 5,'Clear': 6,'Thunderstorm': 7}
weather_mapping={'Sunny': 0,'Cloudy': 1,'Rainy': 2}

# time_mapping = {'Breakfast': 0,'Morning': 1,'Afternoon': 2,'Lunch': 3,'Evening': 4,'Dinner': 5,'Night': 6,'Latenight': 7,'Snacks': 8,'Regular': 9}
time_mapping={'Morning': 0,'Afternoon': 1,'Night': 2}

x['City'] = x['City'].map(city_mapping)
x['Traffic_Level'] = x['Traffic_Level'].map(traffic_mapping)
x['Weather_Conditions'] = x['Weather_Conditions'].map(weather_mapping)
x['Time_of_Day'] = x['Time_of_Day'].map(time_mapping)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)

model = RandomForestRegressor(n_estimators=100,max_depth=5,random_state=42)

model.fit(x_train, y_train)

train_output = model.predict(x_train)
test_output = model.predict(x_test)

st.title("🚴‍♂️ Food Delivery Time Prediction")

st.subheader("Enter Order Details 📦")
city = st.selectbox("Select City",list(city_mapping.keys()))

traffic = st.selectbox("Select Traffic Level",list(traffic_mapping.keys()))

weather = st.selectbox("Select Weather Conditions",list(weather_mapping.keys()))

time_of_day = st.selectbox("Select Time of Day",list(time_mapping.keys()))

distance = st.number_input("Distance (km)",min_value=0.0,value=5.0)

preparation_time = st.number_input("Restaurant Preparation Time (min)", min_value=0, value=20)

items = st.number_input("Number of Items",min_value=1,value=3)

active_orders = st.number_input("Restaurant Active Orders",min_value=0,value=2)

order_value = st.number_input("Order Value (INR)",min_value=0.0,value=500.0)


if st.button("Predict Delivery Time"):
    input_data = pd.DataFrame([[city_mapping[city],distance,traffic_mapping[traffic],weather_mapping[weather],time_mapping[time_of_day],preparation_time,items,active_orders,order_value]], columns=x.columns)

    prediction = model.predict(input_data)

    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")

# Model evaluation
# st.subheader("Model Evaluation")

# st.write("Train R²:", r2_score(y_train, train_output))
# st.write("Test R²:", r2_score(y_test, test_output))
# st.write("Test MAE:", mean_absolute_error(y_test, test_output))
# st.write("Test RMSE:", mean_squared_error(y_test, test_output) ** 0.5)
