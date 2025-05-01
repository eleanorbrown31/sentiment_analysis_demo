import numpy as np
import pandas as pd
import streamlit as st
import json
import re
from pathlib import Path
import time
import matplotlib.pyplot as plt
from model.data_utils import load_initial_training_data, extract_features, create_vocabulary

class SentimentModel:
    def __init__(self):
        # Load initial training data
        self.training_data = load_initial_training_data()
        self.vocabulary = {}
        self.word_weights = {}
        self.is_training = False
        self.accuracy = 0
        self.training_history = []
        self.epochs = 30  # Default number of epochs
        self.create_vocabulary()
        
    def create_vocabulary(self):
        """Create vocabulary from training data"""
        self.vocabulary = create_vocabulary(self.training_data)
        
    def predict_sentiment_score(self, features, weights=None):
        """Make a prediction using logistic regression"""
        if weights is None:
            weights = self.word_weights
            
        if not weights:
            return 0.5  # Default if no model is trained
            
        score = 0
        
        # Add bias term
        score += weights.get('__bias__', 0)
        
        # Add weighted sum of features
        for word, count in features.items():
            if word in weights:
                score += count * weights[word]
                
        # Apply sigmoid function to get probability
        return 1 / (1 + np.exp(-score))
        
    def train(self, progress_callback=None):
        """Train model using logistic regression with mini-batch gradient descent"""
        self.is_training = True
        current_epoch = 0
        self.training_history = []
        
        # Initialize weights with small random values
        weights = {'__bias__': 0}
        for word in self.vocabulary:
            weights[word] = np.random.uniform(-0.05, 0.05)
            
        # Parameters
        base_learning_rate = 0.1
        history = []
        
        # Split data for training/validation
        np.random.shuffle(self.training_data)
        split_idx = int(len(self.training_data) * 0.8)
        training_set = self.training_data[:split_idx]
        validation_set = self.training_data[split_idx:]
        
        # Train for specified epochs
        for epoch in range(self.epochs):
            current_epoch = epoch + 1
            if progress_callback:
                progress_callback(current_epoch, self.epochs)
                
            # Adaptive learning rate
            learning_rate = base_learning_rate / (1 + epoch * 0.1)
            
            # Shuffle training data
            np.random.shuffle(training_set)
            
            batch_size = 5
            total_loss = 0
            
            # Process in mini-batches
            for i in range(0, len(training_set), batch_size):
                batch = training_set[i:i+batch_size]
                batch_gradients = {'__bias__': 0}
                
                # Compute gradients for batch
                for example in batch:
                    features = extract_features(example['text'], self.vocabulary)
                    prediction = self.predict_sentiment_score(features, weights)
                    target = example['sentiment']
                    error = target - prediction
                    
                    # Update bias gradient
                    batch_gradients['__bias__'] += error
                    
                    # Update feature gradients
                    for word, count in features.items():
                        if word not in batch_gradients:
                            batch_gradients[word] = 0
                        batch_gradients[word] += error * count
                        
                    # Track loss
                    total_loss += error ** 2
                    
                # Apply batch gradients
                weights['__bias__'] += learning_rate * batch_gradients['__bias__'] / len(batch)
                
                for word, gradient in batch_gradients.items():
                    if word != '__bias__':
                        if word in weights:
                            weights[word] += learning_rate * gradient / len(batch)
                
            # Add regularization to prevent overfitting
            regularization_rate = 0.01
            for word in weights:
                if word != '__bias__':
                    weights[word] *= (1 - regularization_rate * learning_rate)
            
            # Evaluate on validation set
            correct = 0
            for example in validation_set:
                features = extract_features(example['text'], self.vocabulary)
                prediction = self.predict_sentiment_score(features, weights)
                predicted_class = 1 if prediction > 0.5 else 0
                if predicted_class == example['sentiment']:
                    correct += 1
                    
            epoch_accuracy = correct / len(validation_set) if validation_set else 0
            self.accuracy = epoch_accuracy
            
            # Store metrics
            history.append({
                'epoch': current_epoch,
                'accuracy': epoch_accuracy,
                'loss': total_loss / len(training_set) if training_set else 0
            })
            
            self.training_history = history.copy()
            
            # Small delay to show progress
            time.sleep(0.1)
            
        # Save final weights
        self.word_weights = weights
        self.is_training = False
        
        # Return feature importance
        sorted_words = sorted(
            [(word, abs(weights[word])) 
             for word in weights if word != '__bias__'],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_words[:15]
        
    def predict(self, text):
        """Make a prediction for new text"""
        if not self.word_weights:
            return None
            
        features = extract_features(text, self.vocabulary)
        score = self.predict_sentiment_score(features)
        
        sentiment = 'positive' if score > 0.5 else 'negative'
        confidence = score * 100 if score > 0.5 else (1 - score) * 100
        
        # Get important words for explanation
        important_words = self.get_important_words(text)
        
        return {
            'id': int(time.time() * 1000),
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'score': score,
            'important_words': important_words,
            'added_to_training': False
        }
        
    def get_important_words(self, text):
        """Get words that were most important for this prediction"""
        if not self.word_weights:
            return []
            
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        word_scores = []
        
        for word in words:
            if word in self.vocabulary and word in self.word_weights:
                word_scores.append({
                    'word': word,
                    'weight': self.word_weights[word],
                    'impact': 'positive' if self.word_weights[word] > 0 else 'negative'
                })
                
        # Sort by absolute weight value and take top 5
        word_scores.sort(key=lambda x: abs(x['weight']), reverse=True)
        return word_scores[:5]
        
    def add_to_training(self, text, sentiment):
        """Add example to training data"""
        sentiment_value = 1 if sentiment == 'positive' else 0
        
        self.training_data.append({
            'text': text,
            'sentiment': sentiment_value
        })
        
        # Update vocabulary with new words
        self.create_vocabulary()
        
        return True