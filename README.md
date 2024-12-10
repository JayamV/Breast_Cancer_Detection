### **README for Breast Cancer Classification Using Adaboost Classifer Project**

---

# **AI Breast Cancer Diagnostic Assistant**

This project implements an **AI-powered Breast Cancer Diagnostic Assistant** leveraging machine learning techniques. It uses the **Breast Cancer Wisconsin (Diagnostic) dataset** to classify tumors as malignant or benign. The project is enhanced with a user-friendly **Streamlit-based web interface**, providing interactive insights and predictions.

---

## **Overview**

Breast cancer remains one of the leading causes of cancer-related deaths worldwide. Early and accurate detection is vital for effective treatment. This project uses the **AdaBoost** algorithm combined with **SMOTE (Synthetic Minority Oversampling Technique)** to enhance classification. The tool integrates features like feature importance visualization, real-time predictions, and advanced performance analysis.

---

## **Features**

### **Key Functionalities:**
- **Interactive Sidebar**: Modify patient feature inputs dynamically.
- **Real-Time Predictions**:
  - Predicts whether the tumor is **Malignant** or **Benign**.
  - Provides prediction **confidence levels**.
- **Visual Performance Metrics**:
  - ROC Curve
  - Precision-Recall Curve
  - Confusion Matrix
  - Detailed Classification Report
- **Feature Insights**:
  - Feature Importance
  - Correlation Heatmap

---

## **Dependencies**

Install the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `plotly`
- `pickel`
- `streamlit`

Install them via:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly joblib streamlit
```

---

## **Model and Techniques**

### **Machine Learning**:
- **Algorithm**: AdaBoost Classifier
- **Data Preprocessing**:
  - Standardization (mean=0, variance=1)
  - Oversampling with **SMOTE** to handle class imbalance.
- **Evaluation**:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - Precision, Recall, F1-Score

### **Visualization**:
- **Correlation Heatmap**: Understand feature relationships.
- **Pie Chart**: Represent probability distribution.

---

## **Performance Highlights**

- **Accuracy**: Achieved with advanced AdaBoost techniques.
- **AUC-ROC**: Indicates the model’s capability to distinguish between classes.
- **Precision-Recall**: Provides trade-offs for imbalanced datasets.

---

## **Interactive Features**

- **Dynamic Input**: Adjust feature sliders to simulate patient data.
- **Real-Time Predictions**: Immediate results with confidence visualization.

---
## **Streamlit Application**

The **Streamlit web app** enhances user interaction with features like dynamic sliders for feature input, visualizations, and real-time updates.

Access the **Streamlit web app** via Link:
```bash
https://appapppy-crggrunkgrjz3cwaps4tif.streamlit.app/
```
---

## **Medical Disclaimer**

**Important Notice**:
- This tool is a screening aid, not a substitute for professional medical diagnosis.
- Always consult a certified medical professional for treatment decisions.

---

## **Developer Team**

| **Name**                   | **Contact**                   | **GitHub**                              | **LinkedIn**                                     |
|----------------------------|-------------------------------|-----------------------------------------|-------------------------------------------------|
| Jayam V                  | jayamwcc@gmail.com        | [GitHub](https://github.com/JayamV/Breast_Cancer_Detection) | [LinkedIn](https://www.linkedin.com/in/jayamv) |
| Sayyed Nizar M        | 727723euai114@skcet.ac.in      | [GitHub](https://github.com/SayyedNizar/breast_cancer.git) | [LinkedIn](https://www.linkedin.com/in/sayyed-nizar-b144322a0?) |
| A.V.K Sai Surya               | avksaisurya77@gmail.com | [GitHub](https://github.com/avksaisurya03/infosys) | |
| Bala Sairam Goli      | balasairamgoli4@gmail.com     | [GitHub](https://github.com/GOLIBALASAIRAM/BreastCancer-Detection) |[LinkedIn](https://www.linkedin.com/in/sairam-goli) |
| Shaik Moinuddin Chisty                  | moinuddinchistyshaik@gmail.com      | [GitHub](https://github.com/moinuddinchisty786/Infosys) |[LinkedIn](https://www.linkedin.com/in/moinuddin-chisty-6b5a182b2/) |

---

## **Future Scope**

- Expand datasets for broader demographic inclusion.
- Incorporate advanced deep learning models for improved accuracy.
- Add multilingual support for global accessibility.

---

**Transforming diagnosis with AI for better healthcare.**
 
