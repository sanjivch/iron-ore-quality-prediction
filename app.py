import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import random
import sklearn



st.set_page_config(layout='wide')


@st.cache
def load_data():
    df = pd.read_csv('data/Test.csv')
    return df
st.title('Iron Ore - Quality Prediction')

st.subheader('Inputs')
X_test = load_data()
rand_num = random.randint(0, len(X_test))

X_input = X_test.values[rand_num]
y_actual = pd.read_csv('data/targets.csv')
y_actual = y_actual.values[rand_num]
feature_list = ['% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow',
       'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
       'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow',
       'Flotation Column 03 Air Flow', 'Flotation Column 04 Air Flow',
       'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow',
       'Flotation Column 07 Air Flow', 'Flotation Column 01 Level',
       'Flotation Column 02 Level', 'Flotation Column 03 Level',
       'Flotation Column 04 Level', 'Flotation Column 05 Level',
       'Flotation Column 06 Level', 'Flotation Column 07 Level',
       '% Iron Concentrate']
with st.expander('Inputs'):
    X = []
    col1, col2, col3  = st.columns(3)
    for index,i in enumerate(feature_list[:7]):
        X.append(col1.number_input(str(i), value=X_input[index]))
    for index, i in enumerate(feature_list[7:14]):
        X.append(col2.number_input(str(i), value=X_input[7+index]))
    for index, i in enumerate(feature_list[14:]):
        X.append(col3.number_input(str(i), value=X_input[14+index]))
st.subheader('% Silica at the end of the flotation process ')
#st.write(X)
# load the model from disk
loaded_model = pickle.load(open('notebooks/model_lr.pkl', 'rb'))
x_scaler = pickle.load(open('notebooks/scaler_lr.pkl', 'rb'))
X = np.array(X)
X_input_sc = x_scaler.transform(X.reshape(1, -1))
print(X_input_sc)
result = loaded_model.predict(X_input_sc)
result = round(result[0], 2)
st.write(f'Predicted: {result}%')
st.write(f'Actual: {y_actual[0]}%')


