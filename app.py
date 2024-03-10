import streamlit as st
import pickle
import numpy as np

# 加载模型
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 创建Streamlit应用
st.title('Iris Species Prediction')

# 用户输入特征值
sepal_length = st.number_input('Sepal Length', value=5.0)
sepal_width = st.number_input('Sepal Width', value=3.0)
petal_length = st.number_input('Petal Length', value=4.0)
petal_width = st.number_input('Petal Width', value=1.0)

iris_species = ['Setosa', 'Versicolor', 'Virginica']

# 预测按钮
if st.button('Prediction'):
    test_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(test_array)
    st.write(f'Species is：{iris_species[prediction[0]]}')