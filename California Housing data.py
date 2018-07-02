from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#read the data
cali_data = pd.read_csv('C://Users//Madhura Snehal//Desktop//Madhura//MyProjects//Tensorflow Course//FULL-TENSORFLOW-NOTES-AND-DATA//Tensorflow-Bootcamp-master//02-TensorFlow-Basics//cal_housing_clean.csv')
cali_data.head()

#separating the value to be predicted
y_label = cali_data['medianHouseValue']
split_data = cali_data.drop('medianHouseValue', axis = 1)


#splitting the data into training(70%) and test(30%)
x_train, x_test, y_train, y_test = train_test_split(split_data,y_label, test_size=0.3, random_state=101)
scaled_data = MinMaxScaler()#transforms features by scaling each feature to a specific/given range

#fit the data
scaled_data.fit(x_train)

#set the data into a dataframe instead if a numpy array and transform
x_scaled_df_data = pd.DataFrame(data = scaled_data.transform(x_train), 
                                columns = x_train.columns, 
                                index = x_train.index)
x_scaled_test_df_data = pd.DataFrame(data = scaled_data.transform(x_test),
                                    columns = x_test.columns,
                                    index = x_test.index)

#creating feature columns
cali_data.columns
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
household = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

#aggregating the columns
feature_col = [age, rooms, bedrooms, population, household, income]

#Creating the input function for the estimator object
input_func = tf.estimator.inputs.pandas_input_fn(x = x_train, y= y_train, batch_size=14, num_epochs=1000, shuffle=True)

#create the estimator model. using DNN regressor for 3 layers with 8 neurons
model = tf.estimator.DNNRegressor(hidden_units=[15,15,15], feature_columns=feature_col)

#train the model for more than 1000 steps
model.train(input_fn=input_func, steps=35000)

#create a prediction function

prediction_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1, shuffle=False)
prediction = model.predict(prediction_input_func)
#creating a list of predictions
list_preds = list(prediction)

final_preds = []

for i in list_preds:
    final_preds.append(i['predictions'])
    
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,final_preds)**0.5
