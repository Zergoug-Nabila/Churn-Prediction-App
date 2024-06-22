import streamlit as st
import pandas as pd
import pickle


st.write("""
# Churn Prediction App

Dataset description: 
         Expresso is an African telecommunications services company that provides telecommunication 
    services in two African markets: Mauritania and Senegal. The data describes 2.5 million Expresso clients with more 
    than 15 behaviour variables in order to predict the clients' churn probability.
""")



# Sidebar for user input features
st.sidebar.header('User Input Features')

# Define the user input function
def user_input_features():
    REGION = st.sidebar.selectbox('REGION', ('DAKAR', 'LOUGA', 'THIES', 'KOLDA', 'KAOLACK', 'KAFFRINE',
                                            'TAMBACOUNDA', 'MATAM', 'SAINT-LOUIS', 'SEDHIOU', 'DIOURBEL',
                                            'ZIGUINCHOR', 'FATICK', 'KEDOUGOU')) 
    MONTANT = st.sidebar.slider('MONTANT', 0.0, 470000.0, 10.0)
    FREQUENCE_RECH = st.sidebar.slider('FREQUENCE_RECH', 0, 133, 1)
    REVENUE = st.sidebar.slider('REVENUE', 0.0, 532177.0, 1.0)
    FREQUENCE = st.sidebar.slider('FREQUENCE', 0, 91, 1)
    DATA_VOLUME = st.sidebar.slider('DATA_VOLUME', 0.0, 1823866.0, 0.0)
    REGULARITY = st.sidebar.slider('REGULARITY', 0, 62, 1)
    FREQ_TOP_PACK = st.sidebar.slider('FREQ_TOP_PACK', 0, 713, 1)

    data = {
        'REGION': REGION,
        'MONTANT': MONTANT,
        'FREQUENCE_RECH': FREQUENCE_RECH,
        'REVENUE': REVENUE,
        'FREQUENCE': FREQUENCE,
        'DATA_VOLUME': DATA_VOLUME,
        'REGULARITY': REGULARITY,
        'FREQ_TOP_PACK': FREQ_TOP_PACK
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Load the trained model
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocess the input data to match the training data
# Here we need to encode categorical variables
# Assume 'REGION' and 'TOP_PACK' are categorical

# Encoding function (manual mapping based on dataset)
def encode_features(df):
    # Example encoding - replace with actual values and mappings
    region_mapping = {'DAKAR':0, 'LOUGA':1, 'THIES':2, 'KOLDA':3, 'KAOLACK':4, 'KAFFRINE':5,
                    'TAMBACOUNDA':6, 'MATAM':7, 'SAINT-LOUIS':8, 'SEDHIOU':9, 'DIOURBEL':10,
                    'ZIGUINCHOR':11, 'FATICK':12, 'KEDOUGOU':13}
    
    df['REGION'] = df['REGION'].map(region_mapping)
    
    return df

# Encode the input features
input_df = encode_features(input_df)

# Display the user input features
st.subheader('User Input features')
st.write(input_df)

# Predict and display the result
if st.button('Predict'):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")
