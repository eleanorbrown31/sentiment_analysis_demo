import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from components.training import show_training_section
from components.prediction import show_prediction_section
from components.results import show_results_section
from model.sentiment_model import SentimentModel

# Set page title and configuration
st.set_page_config(page_title="Machine Learning Sentiment Analysis Demo", layout="wide")
st.title("Machine Learning Sentiment Analysis Demo")

# Initialize session state for persistence across reruns
if 'model' not in st.session_state:
    st.session_state.model = SentimentModel()
if 'submissions' not in st.session_state:
    st.session_state.submissions = []
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

# Display explanatory information
st.markdown("""
This application demonstrates how machine learning works with a simple sentiment analysis model:
1. **Training**: Start with labelled examples of positive and negative text
2. **Testing**: See how the model performs on new text
3. **Feedback**: Provide corrections to help the model improve
""")

# Main app layout with three columns
col1, col2 = st.columns([1, 1])

with col1:
    # Training section 
    show_training_section(st.session_state.model, st.session_state.training_history)

with col2:
    # Prediction section
    show_prediction_section(st.session_state.model, st.session_state.submissions)

# Results section (spans full width)
show_results_section(st.session_state.model, st.session_state.submissions, st.session_state.training_history)

# App explanation
st.markdown("""
### How This Shows Machine Learning

* We start with labelled training data (examples of positive and negative text)
* The model learns patterns from this data during training
* With each epoch, the model's accuracy typically improves
* The trained model can then analyse new text it hasn't seen before
* The model gives both a prediction and confidence score
* **Feedback loop:** When predictions are wrong, you can correct them
* **Model improvement:** Corrections are added to training data to improve future predictions
* **Continuous learning:** Retrain the model with the expanded dataset to improve accuracy

This demonstrates the core concepts of supervised machine learning in an interactive way.
""")