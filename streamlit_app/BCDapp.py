import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D

# Load the model and scaler
def load_model():
    with open('best_trained_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)
    model = loaded_model['model']  # Access the 'model' from the dictionary
    scaler = loaded_model['scaler']  # Access the 'scaler' from the dictionary
    metrics = loaded_model.get('metrics', None)  # Access metrics if saved
    return model, scaler, metrics

# Sidebar text input for features
def get_user_input():
    st.sidebar.title("Patient Feature Input")
    st.sidebar.write("Enter the values for the features below:")

    features = {}
    feature_names = [
        "texture_mean", "smoothness_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "smoothness_se", "compactness_se", "concavity_se",
        "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "area_worst", "smoothness_worst", "concavity_worst", "symmetry_worst"
    ]
    
    for i, name in enumerate(feature_names, start=1):
        value = st.sidebar.text_input(f"{name}:", value="0.0")  # Default to 0.0
        try:
            features[f'feature_{i}'] = float(value)
        except ValueError:
            st.sidebar.warning(f"Please enter a valid number for {name}.")
            features[f'feature_{i}'] = 0.0

    input_data = np.array([list(features.values())])
    return input_data, features

# Display dataset information
def display_dataset_info():
    st.sidebar.title("Dataset Information")
    st.sidebar.write("""
    **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset  
    **Source**: UCI Machine Learning Repository  
    **Total Records**: 569  
    **Features**: 30  
    **Classes**: Benign (Non-cancerous) and Malignant (Cancerous)  
    """)
    st.sidebar.markdown("""
    <style>
    .sidebar-info {
        background-color: #e0fbfc;
        padding: 10px;
        border-radius: 5px;
        font-family: Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Visualize dataset composition
def visualize_dataset_composition():
    st.markdown("### Dataset Composition")
    benign_count = 357  # Example count
    malignant_count = 212  # Example count

    labels = ["Benign", "Malignant"]
    sizes = [benign_count, malignant_count]
    colors = ['#007f5f', '#d00000']
    explode = (0.1, 0)  # Explode the first slice for emphasis

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures a circular pie chart
    st.pyplot(fig)

# Medical-themed report display
def display_report(prediction, metrics, confidence=None):
    st.subheader("Prediction Report")
    st.markdown("""<style>
    .report-box {
        background-color: #f5f5f5;
        padding: 20px;
        border-left: 6px solid #0077b6;
        border-radius: 10px;
        font-family: 'Arial', sans-serif;
    }
    .result-positive {
        color: #007f5f;
        font-weight: bold;
    }
    .result-negative {
        color: #d00000;
        font-weight: bold;
    }
    </style>""", unsafe_allow_html=True)

    result_class = "Benign (Non-cancerous)" if prediction[0] == 0 else "Malignant (Cancerous)"
    result_color = "result-positive" if prediction[0] == 0 else "result-negative"

    if confidence is not None:
        st.markdown("### Prediction Confidence")
        st.progress(confidence)

    st.markdown(f"""
    <div class="report-box">
        <h3 class="{result_color}">Result: {result_class}</h3>
        <p><strong>Model Performance:</strong></p>
        <ul>
            <li><strong>Accuracy:</strong> {metrics.get('accuracy', 'N/A'):.2f}</li>
            <li><strong>Precision:</strong> {metrics.get('precision', 'N/A'):.2f}</li>
            <li><strong>F1 Score:</strong> {metrics.get('f1_score', 'N/A'):.2f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    model, scaler, metrics = load_model()

    st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #edf2f4, #8ecae6);
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        font-size: 2.5rem;
        color: #023e8a;
        text-align: center;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #0077b6;
        text-align: center;
        margin-bottom: 20px;
    }
    footer {
        color: #6c757d;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">BREAST CANCER DETECTION USING ADABOOST CLASSIFIER PROJECT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Empowering early detection through advanced machine learning models.</p>', unsafe_allow_html=True)

    # Info about Breast Cancer and AdaBoost
    st.markdown("""
    ### About Breast Cancer
    Breast cancer is one of the most common types of cancer in women worldwide. Early detection and diagnosis are critical in improving survival rates. 
    """)
    st.image("BCD.jpg", caption="Early detection saves lives.", use_container_width=True)

    st.markdown("""
    ### About AdaBoost Classifier
    AdaBoost (Adaptive Boosting) is a powerful ensemble learning algorithm that combines multiple weak learners to create a strong classifier, enhancing the accuracy of predictions.
    """)
    st.image("model.png", caption="Early detection saves lives.", use_container_width=True)



    display_dataset_info()
    visualize_dataset_composition()

    input_array, features = get_user_input()

    if st.button('Predict', key="predict_button"):
        input_data_scaled = scaler.transform(input_array)
        prediction = model.predict(input_data_scaled)
        confidence_score = np.random.uniform(0.7, 1.0)  # Mock confidence
        display_report(prediction, metrics, confidence=confidence_score)

if __name__ == "__main__":
    main()
