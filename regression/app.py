import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained regression model
model = tf.keras.models.load_model('salary_model.h5')

# Load encoders
with open('label_encoder_gender.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App UI
st.title('Salary Prediction')

# Input form
with st.form("salary_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        geography = st.selectbox('Geography', geo_encoder.categories_[0])
        gender = st.selectbox('Gender', gender_encoder.classes_)
        age = st.slider('Age', 18, 100, 30)
        credit_score = st.slider('Credit Score', 300, 850, 650)
        Exited = st.selectbox('Has Exited?',[True,False],index=0)
        
    with col2:
        balance = st.number_input('Balance', value=0.0)
        tenure = st.slider('Tenure (years)', 0, 10, 5)
        num_of_products  = st.slider('Products', 1, 4, 1)
        is_active_member = st.selectbox('Is Active Member',[True,False],index=0)
        has_cr_card = st.selectbox('Has Credit Card',[True,False],index=0)
        
    
    submitted = st.form_submit_button('Predict Salary')

# Prediction logic
if submitted:
    # Encode inputs
    geo_encoded = geo_encoder.transform([[geography]]).toarray()
    gender_encoded = gender_encoder.transform([gender])[0]
    
    # Create feature DataFrame
    features = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'IsActiveMember': [1 if is_active_member else 0],
        'HasCrCard': [1 if has_cr_card else 0],
        'Exited' : [1 if Exited else 0]
    })
    
    # Add geography features
    geo_df = pd.DataFrame(geo_encoded, 
                         columns=geo_encoder.get_feature_names_out(['Geography']))
    features = pd.concat([features.reset_index(drop=True), geo_df], axis=1)
    
    
    # Scale and predict
    expected_columns = scaler.feature_names_in_  # Get expected feature names
    features = features[expected_columns]  # Reorder features
   

    scaled_features = scaler.transform(features)
    salary = model.predict(scaled_features)[0][0]
    
    # Display only the salary
    st.subheader(f"Predicted Salary: ${salary:,.2f}")