from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#read the data
cali_data = pd.read_csv('census_data.csv')
cali_data.head(3)
#cali_data.describe().transpose()

#replacing prediction values with 0 and 1
def  replace_stringsreplace (x):
    if x ==' <=50K':
        return 0
    else:
        return 1
cali_data['income_bracket'] = cali_data['income_bracket'].apply(replace_strings)
x_data = cali_data.drop('income_bracket', axis=1)
y_label = cali_data['income_bracket']
x_train,x_test,y_train,y_test = train_test_split(x_data, y_label, test_size=0.3, random_state=101)
cali_data.columns

#creating feature columns for categorical values using vocabulary lists or just hash buckets. 
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",["Female","Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status",hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country",hash_bucket_size=1000)

##creating feature columns for numerical values
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

#Putting all these variables into a single list with the variable name feature_cols
feature_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]
#feature_cols = [gender,occupation,workclass,education,marital_status,relationship,native_country,age,education_num,capital_gain,capital_loss,hours_per_week]

#Creating the input function
input_func = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size=100,num_epochs=None, shuffle=True)

#creating data model using LinearClassifier
model = tf.estimator.LinearClassifier(feature_columns=feature_cols)

#train your model data
model.train(input_fn=input_func, steps=500)

#create a prediction function
prediction_input = tf.estimator.inputs.pandas_input_fn(x = x_test, batch_size=len(x_test),shuffle=False)

#using predict function and generating number of prediction converting into a list#using  
generated_prediction = model.predict(input_fn=input_func)
list_preds = list(generated_prediction)
final_preds = []
for pred in list_preds:
    final_preds.append(pred['class_ids'][0])

from sklearn.metrics import classification_report
print(classification_report(y_test, final_preds))
