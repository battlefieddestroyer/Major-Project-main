import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
import base64
from fpdf import FPDF
import time

# Set Streamlit Page Configuration
st.set_page_config(page_title="🌱 Plant Disease Detection", layout="wide")

# Sidebar Navigation
st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "About", "Disease Recognition", "Live Camera Capture"])

# **Load Trained Model**
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('vgg161model.keras')
    model.make_predict_function()  # Optimize inference speed
    return model

model = load_model()


# **Disease Solutions Dictionary**
disease_solutions =   {'Apple___Apple_scab': ("A fungal disease causing dark, scaly lesions on leaves and fruit.", 
                            "Use fungicides like captan, and plant scab-resistant apple varieties."),
    'Apple___Black_rot': ("A fungal infection that causes fruit rot and leaf lesions.", 
                          "Prune infected branches, use copper-based sprays."),
    'Apple___Cedar_apple_rust': ("A fungal disease leading to orange spots on leaves.", 
                                 "Apply fungicides and remove nearby juniper trees."),
    'Apple___healthy': ("No disease detected, plant is healthy.", 
                        "Maintain good watering and soil conditions."),
    'Blueberry___healthy': ("No disease detected, plant is healthy.", 
                            "Ensure proper pruning and soil pH balance."),
    'Cherry_(including_sour)___Powdery_mildew': ("A fungal disease causing white powdery spots.", 
                                                 "Use sulfur-based fungicides and remove infected leaves."),
    'Cherry_(including_sour)___healthy': ("No disease detected, plant is healthy.", 
                                          "Ensure good air circulation and proper watering."),
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ("A fungal disease causing gray lesions on leaves.", 
                                                           "Use resistant varieties and apply fungicides."),
    'Corn_(maize)___Common_rust_': ("Rust disease leading to reddish-brown pustules on leaves.", 
                                    "Apply fungicides and rotate crops."),
    'Corn_(maize)___Northern_Leaf_Blight': ("Dark lesions on leaves caused by a fungal pathogen.", 
                                            "Use disease-resistant hybrids and fungicides."),
    'Corn_(maize)___healthy': ("No disease detected, plant is healthy.", 
                               "Ensure proper irrigation and nutrient management."),
    'Grape___Black_rot': ("A fungal disease causing black spots on leaves and fruit.", 
                          "Use fungicides and remove infected plant material."),
    'Grape___Esca_(Black_Measles)': ("A fungal disease causing yellow streaks and fruit rot.", 
                                      "Prune infected vines and apply fungicides."),
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ("A fungal infection causing leaf necrosis.", 
                                                   "Apply copper-based fungicides."),
    'Grape___healthy': ("No disease detected, plant is healthy.", 
                        "Maintain good vineyard hygiene and soil balance."),
    'Orange___Haunglongbing_(Citrus_greening)': ("A bacterial disease spread by psyllids, causing leaf yellowing and fruit deformities.", 
                                                 "Remove infected trees and control psyllid population."),
    'Peach___Bacterial_spot': ("Dark, sunken spots on leaves and fruit caused by bacteria.", 
                               "Use copper-based sprays and disease-free seedlings."),
    'Peach___healthy': ("No disease detected, plant is healthy.", 
                        "Ensure good pruning and avoid overwatering."),
    'Pepper,_bell___Bacterial_spot': ("A bacterial disease causing dark, water-soaked spots.", 
                                      "Use resistant varieties and avoid overhead watering."),
    'Pepper,_bell___healthy': ("No disease detected, plant is healthy.", 
                               "Maintain proper fertilization and pest control."),
    'Potato___Early_blight': ("Dark, concentric rings on leaves caused by a fungal pathogen.", 
                              "Apply fungicides and remove infected leaves."),
    'Potato___Late_blight': ("Water-soaked lesions and rapid decay due to a fungal pathogen.", 
                             "Use resistant varieties and copper fungicides."),
    'Potato___healthy': ("No disease detected, plant is healthy.", 
                         "Ensure proper crop rotation and soil nutrition."),
    'Raspberry___healthy': ("No disease detected, plant is healthy.", 
                            "Maintain well-drained soil and proper pruning."),
    'Soybean___healthy': ("No disease detected, plant is healthy.", 
                          "Use balanced fertilizers and pest control methods."),
    'Squash___Powdery_mildew': ("White powdery growth on leaves caused by fungi.", 
                                "Use sulfur-based sprays and improve air circulation."),
    'Strawberry___Leaf_scorch': ("Brown, dried leaf edges due to fungal infection.", 
                                 "Remove infected leaves and apply fungicides."),
    'Strawberry___healthy': ("No disease detected, plant is healthy.", 
                             "Keep soil well-drained and control pests."),
    'Tomato___Bacterial_spot': ("Dark spots on leaves and fruit due to bacterial infection.", 
                                "Use disease-free seeds and apply copper sprays."),
    'Tomato___Early_blight': ("Brown, concentric rings on leaves due to fungus.", 
                              "Apply fungicides and rotate crops."),
    'Tomato___Late_blight': ("Gray lesions on leaves causing rapid decay.", 
                             "Use resistant varieties and fungicides."),
    'Tomato___Leaf_Mold': ("Yellowing and moldy patches on leaves.", 
                           "Improve ventilation and apply fungicides."),
    'Tomato___Septoria_leaf_spot': ("Small, dark leaf spots caused by fungi.", 
                                    "Use fungicides and remove affected leaves."),
    'Tomato___Spider_mites Two-spotted_spider_mite': ("Tiny mites causing leaf stippling and yellowing.", 
                                                      "Use neem oil and increase humidity."),
    'Tomato___Target_Spot': ("Dark spots with yellow halos on leaves.", 
                             "Apply copper-based fungicides."),
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ("Virus causing curled and yellow leaves.", 
                                               "Use insect-resistant nets and remove infected plants."),
    'Tomato___Tomato_mosaic_virus': ("Mosaic-like leaf discoloration due to viral infection.", 
                                     "Use virus-free seeds and control aphids."),
    'Tomato___healthy': ("No disease detected, plant is healthy.", 
                         "Ensure good soil nutrition and pest control.")
}



# **Function for Model Prediction**
def model_prediction(image):
    try:
        image = cv2.resize(image, (224, 224))  # Resize for VGG16 model
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)  # VGG16 preprocessing
        prediction = model.predict(image)
        confidence = np.max(prediction) * 100  # Confidence Score
        result_index = np.argmax(prediction)
        return result_index, confidence
    except Exception:
        return None, None  # Handle invalid images gracefully

# **Function to Generate PDF Report**
def generate_pdf_report(predicted_disease, confidence, solution):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Plant Disease Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {predicted_disease}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Disease Info: {solution[0]}", ln=True)
    pdf.cell(200, 10, txt=f"Solution: {solution[1]}", ln=True)
    pdf_output = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_output).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="plant_disease_report.pdf">📥 Download PDF Report</a>'

# **Function to Check if Image is a Plant Leaf**
def is_plant_leaf(image):
    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_image, (36, 25, 25), (86, 255, 255))
        green_ratio = cv2.countNonZero(green_mask) / (image.size / 3)
        return green_ratio > 0.2  # Adjust threshold as needed
    except Exception as e:
        print(f"Error in is_plant_leaf: {e}")
        return False

# **Home Page**
if page == "Home":
    st.title("🌿 PLANT DISEASE DETECTION SYSTEM")
    st.image("homepage.webp", use_column_width=True)
    st.markdown("""
    ### Welcome to the Plant Disease Recognition System! 🌱📸
    - 📷 Upload a plant image to detect diseases instantly.
    - 🛡️ Get disease solutions & prevention tips.
    - 🚀 Powered by AI & Deep Learning (VGG16 Model).
    """)

# **About Page**
elif page == "About":
    st.title("📖 About the Project")
    st.markdown("""
    - This system uses **Deep Learning (VGG16)** to detect plant diseases.
    - Dataset consists of **87,000+ images** from 38 plant classes.
    - Trained with **TensorFlow & Keras** for high accuracy.
    """)

# **Disease Recognition Page**
elif page == "Disease Recognition":
    st.title("🔍 Disease Recognition")
    st.markdown("**Upload a plant leaf image to detect diseases.**")

    test_image = st.file_uploader("📸 Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        img_bytes = test_image.read()
        
        if len(img_bytes) == 0:
            st.error("⚠️ Uploaded image is empty. Please upload a valid image.")
        else:
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            st.image(image_cv, channels="BGR", width=300)

            if st.button("🚀 Predict Disease"):
                with st.spinner("Analyzing Image... Please Wait ⏳"):
                    time.sleep(2)
                    result_index, confidence = model_prediction(image_cv)

                    if result_index is None:
                        st.error("⚠️ Error processing image. Please upload a clear image.")
                    else:
                        class_names = list(disease_solutions.keys())
                        predicted_disease = class_names[result_index] if result_index < len(class_names) else "Unknown Disease"
                        solution = disease_solutions.get(predicted_disease, ("No specific solution available.", ""))

                        st.metric(label="🧠 Confidence Score", value=f"{confidence:.2f}%")
                        st.success(f"✅ **Prediction:** {predicted_disease}")
                        st.info(f"📌 **Disease Info:** {solution[0]}")
                        st.warning(f"💡 **Solution:** {solution[1]}")
                        st.markdown(generate_pdf_report(predicted_disease, confidence, solution), unsafe_allow_html=True)

# **Live Camera Capture Page**
elif page == "Live Camera Capture":
    st.title("📷 Live Camera Capture")
    st.markdown("Take a photo using your webcam and analyze the plant disease.")

    captured_image = st.camera_input("📸 Capture an Image")

    if captured_image is not None:
        img_bytes = captured_image.read()
        
        if len(img_bytes) == 0:
            st.error("⚠️ Captured image is empty. Please try again.")
        else:
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            st.image(image_cv, channels="BGR", width=300)

            if not is_plant_leaf(image_cv):
                st.error("⚠️ No plant leaf detected. Please capture an image of a plant leaf.")
            else:
                if st.button("🚀 Predict Disease from Live Image"):
                    with st.spinner("Processing Image... ⏳"):
                        time.sleep(2)
                        result_index, confidence = model_prediction(image_cv)

                        if result_index is None:
                            st.error("⚠️ Error processing image. Please try again.")
                        else:
                            class_names = list(disease_solutions.keys())
                            predicted_disease = class_names[result_index] if result_index < len(class_names) else "Unknown Disease"
                            solution = disease_solutions.get(predicted_disease, ("No specific solution available.", ""))

                            st.metric(label="🧠 Confidence Score", value=f"{confidence:.2f}%")
                            st.success(f"✅ **Prediction:** {predicted_disease}")
                            st.info(f"📌 **Disease Info:** {solution[0]}")
                            st.warning(f"💡 **Solution:** {solution[1]}")
                            st.markdown(generate_pdf_report(predicted_disease, confidence, solution), unsafe_allow_html=True)
