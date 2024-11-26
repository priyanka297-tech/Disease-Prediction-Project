import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Disease Prediction",
    page_icon="💊",
    layout="wide",
)

# Load the model with st.cache_resource
@st.cache_resource
def load_model():
    try:
        with open('Disease model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error("Failed to load the model. Ensure the file exists and is compatible.")
        raise e

# Load the model
try:
    model = load_model()
except Exception as e:
    st.stop()  # Stop execution if the model cannot be loaded

disease_descriptions = {
     "Itching is a sensation that prompts the desire to scratch. It can be caused by various conditions, including dry skin, allergies, insect bites, or underlying diseases such as eczema or psoriasis. Chronic itching may indicate more serious conditions, such as liver or kidney issues. Treatment focuses on addressing the underlying cause and relieving the itch with moisturizers, antihistamines, or topical creams."
    # Add all diseases with their respective descriptions
}
# Define the prediction function
def predictDisease(symptoms):
    symptoms_list = [
        "Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing",
        "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity",
        "Ulcers on Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition",
        "Fatigue", "Weight Gain", "Anxiety", "Cold Hands and Feet", "Mood Swings",
        "Weight Loss", "Restlessness", "Lethargy", "Patches in Throat",
        "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness"
    ]
    
    symptom_index_map = {symptom: index for index, symptom in enumerate(symptoms_list)}
    
    # Create input data as a list of zeros
    input_data = [0] * len(symptoms_list)
    
    # Set the corresponding indices to 1 for the selected symptoms
    for symptom in symptoms:
        if symptom in symptom_index_map:
            input_data[symptom_index_map[symptom]] = 1
    
    # Reshape the input data into a 2D array as expected by the model
    input_data = np.array(input_data).reshape(1, -1)
    
    # Make the prediction
    try:
        prediction = model.predict(input_data)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return disease_descriptions

    
    # symptom_vector = [1 if symptom in symptoms else 0 for symptom in symptoms_list]
    # try:
    #     prediction = model.predict([symptom_vector])[0]
    #     return prediction
    # except Exception as e:
    #     st.error(f"Prediction failed: {e}")
    #     return None

# Streamlit UI
st.title("🩺 Disease Prediction Based on Symptoms")

st.write("""
Welcome to the Disease Prediction App!  
Please select your symptoms from the options below, and the app will predict possible diseases.
""")

# Add a visual header image
st.image("image.jpg", width= 200)

symptoms_list = [
    "Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing",
    "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity",
    "Ulcers on Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition",
    "Fatigue", "Weight Gain", "Anxiety", "Cold Hands and Feet", "Mood Swings",
    "Weight Loss", "Restlessness", "Lethargy", "Patches in Throat",
    "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness"
]

# Dropdown for symptom selection
selected_symptoms = st.multiselect(
    "Select symptoms from the dropdown:", 
    options=symptoms_list,
    help="Choose symptoms based on what you're experiencing.")
# Prediction button
if st.button("Predict Disease"):
    if selected_symptoms:
        # Call the prediction function
        prediction = predictDisease(selected_symptoms)
        st.success(prediction)
    else:
        st.error("Please select at least one symptom to get a prediction.")