import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model (ensure the path is correct)
try:
    model = joblib.load("Disease_Model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'Disease_Model.pkl' is in the correct path.")
    st.stop()

# Predefined symptoms list
symptoms_list = [
    "Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", 
    "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity", 
    "Ulcers on Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition",
    "Fatigue", "Weight Gain", "Anxiety", "Cold Hands and Feet", "Mood Swings", 
    "Weight Loss", "Restlessness", "Lethargy", "Patches in Throat", 
    "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness",
    # Add more symptoms here...
]

# Define the prediction function using the trained model
def predictDisease(symptoms):
    # Create a binary vector for the symptoms
    symptom_vector = [0] * len(symptoms_list)
    for symptom in symptoms:
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            symptom_vector[index] = 1  # Mark the symptom as present

    # Convert the vector to a NumPy array and reshape it
    symptom_vector = np.array(symptom_vector).reshape(1, -1)
    symptom_vector=[[1, 1, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 
12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 
12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 
12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 
12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 
12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]]
    symptom_vector = np.array(symptom_vector).reshape(1, -1)

    try:
        # Make prediction using the trained model
        prediction = model.predict(symptom_vector[0])
        return f"Predicted disease: {prediction[0]}"
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# Streamlit UI
st.title("Disease Prediction Based on Symptoms")

st.write("""
Select your symptoms from the dropdown.  
The app will predict the possible diseases based on the selected symptoms and provide additional information.
""")

# Dropdown for symptom selection
selected_symptoms = st.multiselect(
    "Select symptoms from the dropdown:", 
    options=symptoms_list
)

# Prediction button
if st.button("Predict Disease"):
    if selected_symptoms:
        # Call the prediction function
        prediction = predictDisease(selected_symptoms)
        st.success(prediction)
    else:
        st.error("Please select at least one symptom to get a prediction.")
