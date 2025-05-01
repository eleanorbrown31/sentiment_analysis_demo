import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model.sentiment_model import SentimentModel
from components.training import show_training_section
from components.prediction import show_prediction_section
from components.results import show_results_section

st.set_page_config(
    page_title="ML Sentiment Analysis Demo",
    page_icon="🔍",
    layout="wide",
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = SentimentModel()
    
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
    
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
    
if 'added_to_training' not in st.session_state:
    st.session_state.added_to_training = False

# App title
st.title("Machine Learning Sentiment Analysis")

# Create three main sections
tab1, tab2, tab3 = st.tabs(["1. Training the Model", "2. Using the Model", "3. Analysis Results"])

# Training section
with tab1:
    show_training_section()
    
# Prediction section
with tab2:
    show_prediction_section()
    
# Results section
with tab3:
    show_results_section()

# Footer with explanation
st.markdown("---")
with st.expander("How This Shows Machine Learning", expanded=False):
    st.markdown("""
    - We start with labelled training data (examples of positive and negative text)
    - The model learns patterns from this data during training
    - With each epoch, the model's accuracy typically improves
    - The trained model can then analyse new text it hasn't seen before
    - The model gives both a prediction and confidence score
    - **Feedback loop:** When predictions are wrong, you can correct them
    - **Model improvement:** Corrections are added to training data to improve future predictions
    - **Continuous learning:** Retrain the model with the expanded dataset to improve accuracy
    """)
    
    st.subheader("Interactive Learning Cycle")
    st.markdown("""
    1. Make predictions on new text
    2. Identify incorrect predictions
    3. Provide feedback by marking correct sentiment
    4. Add corrections to training data
    5. Retrain model with enhanced dataset
    6. Observe improved accuracy on similar future inputs
    """)
    
    if st.button("Retrain With Feedback", disabled=st.session_state.model.is_training):
        with st.spinner(f"Training model (this might take a moment)..."):
            st.session_state.model.train()
        st.success("Model retrained with feedback!")