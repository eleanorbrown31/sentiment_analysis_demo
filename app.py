import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import time
import random
from collections import Counter

# Set page config
st.set_page_config(page_title="ML Sentiment Analysis Demo", layout="wide")

# Model class
class SentimentModel:
    def __init__(self):
        self.training_data = self.load_initial_data()
        self.vocabulary = {}
        self.word_weights = {'__bias__': 0}
        self.trained = False
        self.create_vocabulary()
        
    def load_initial_data(self):
        # Initial training data
        data = [
            {"text": "I love this presentation", "sentiment": 1},
            {"text": "This is really helpful", "sentiment": 1},
            {"text": "I don't understand this", "sentiment": 0},
            {"text": "This is boring", "sentiment": 0},
            {"text": "Great explanation of machine learning", "sentiment": 1},
            {"text": "Too complicated for beginners", "sentiment": 0},
            {"text": "I'm learning so much", "sentiment": 1},
            {"text": "Unclear examples", "sentiment": 0},
            {"text": "The visuals help a lot", "sentiment": 1},
            {"text": "Moving too quickly", "sentiment": 0},
            {"text": "This demonstration is amazing", "sentiment": 1},
            {"text": "I'm confused by these concepts", "sentiment": 0},
            {"text": "Excellent presentation style", "sentiment": 1},
            {"text": "The examples don't make sense", "sentiment": 0},
            {"text": "Very clear explanation", "sentiment": 1},
            {"text": "I'm lost and can't follow along", "sentiment": 0},
            {"text": "Best ML demo I've seen", "sentiment": 1},
            {"text":