import streamlit as st
import numpy as np
import pickle

# Load the model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    return crop_dict.get(prediction[0], "Sorry, we could not determine the best crop to be cultivated with the provided data.")

# Streamlit app
def main():
    st.title("Crop Recommendation System 🌱")

    st.write("### Enter the following details to get crop recommendations")

    # Input fields
    N = st.number_input("Nitrogen (kg/ha)", min_value=0.0, step=0.1)
    P = st.number_input("Phosphorus (kg/ha)", min_value=0.0, step=0.1)
    K = st.number_input("Potassium (kg/ha)", min_value=0.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=-50.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, step=0.1)
    ph = st.number_input("pH value", min_value=0.0, step=0.01)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

    if st.button("Get Recommendation"):
        result = predict_crop(N, P, K, temp, humidity, ph, rainfall)
        st.write(f"**Recommended Crop for cultivation is:** {result}")

    # Add additional content or styling as needed
    st.write("---")
    st.write("For more information or assistance, please contact us.")
