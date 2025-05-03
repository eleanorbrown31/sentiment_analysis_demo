import re
import numpy as np
import pandas as pd
import json
import os
from collections import Counter
import time

class SentimentModel:
    def __init__(self):
        self.training_data = self.load_initial_data()
        self.vocabulary = {}
        self.word_weights = {'__bias__': 0}
        self.trained = False
        self.create_vocabulary()
        
    def load_initial_data(self):
        # Load initial training data - similar to the React version
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
            # Original additional training examples
            {"text": "This demonstration is amazing", "sentiment": 1},
            {"text": "I'm confused by these concepts", "sentiment": 0},
            {"text": "Excellent presentation style", "sentiment": 1},
            {"text": "The examples don't make sense", "sentiment": 0},
            {"text": "Very clear explanation", "sentiment": 1},
            {"text": "I'm lost and can't follow along", "sentiment": 0},
            {"text": "Best ML demo I've seen", "sentiment": 1},
            {"text": "This is a waste of time", "sentiment": 0},
            {"text": "The interactive elements are engaging", "sentiment": 1},
            {"text": "Too much information too quickly", "sentiment": 0},
            # Adding more examples
            {"text": "I appreciate the step-by-step approach", "sentiment": 1},
            {"text": "The content is poorly organized", "sentiment": 0},
            {"text": "Very insightful and educational", "sentiment": 1},
            {"text": "The presenter seems unprepared", "sentiment": 0},
            {"text": "This makes machine learning accessible", "sentiment": 1}
        ]
        return data
        
    def create_vocabulary(self):
        # Track word frequencies to filter out rare words
        word_freq = {}
        bigram_freq = {}
        
        # Collect frequencies
        for item in self.training_data:
            words = re.sub(r'[^\w\s]', '', item["text"].lower()).split()
            
            # Count individual words
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Count bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]}_{words[i+1]}"
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
        
        # Filter out uncommon words and bigrams
        min_frequency = 2
        self.vocabulary = {}
        
        # Add words that appear at least min_frequency times
        for word, freq in word_freq.items():
            if freq >= min_frequency:
                self.vocabulary[word] = True
        
        # Add bigrams that appear at least min_frequency times
        for bigram, freq in bigram_freq.items():
            if freq >= min_frequency:
                self.vocabulary[bigram] = True
        
        # Add special features
        self.vocabulary['__has_negation__'] = True
        self.vocabulary['__exclamation__'] = True
        
    def extract_features(self, text):
        features = {}
        
        # Normalize text and split into words
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        
        # Track unigrams (single words)
        for word in words:
            if word in self.vocabulary:
                features[word] = features.get(word, 0) + 1
        
        # Add bigrams (pairs of adjacent words)
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            if bigram in self.vocabulary:
                features[bigram] = features.get(bigram, 0) + 1
        
        # Add some basic sentiment indicators
        negation_words = ['not', 'no', "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't", "weren't"]
        has_negation = any(word in negation_words for word in words)
        if has_negation:
            features['__has_negation__'] = 1
        
        # Add exclamation mark feature
        if '!' in text:
            features['__exclamation__'] = 1
        
        return features
    
    def sigmoid(self, x):
        # Apply sigmoid function to get probability
        return 1 / (1 + np.exp(-x))
    
    def predict_sentiment_score(self, features):
        if not self.word_weights:
            return 0.5  # Default if not trained
        
        score = 0
        
        # Add bias term
        score += self.word_weights.get('__bias__', 0)
        
        # Add weighted sum of features
        for word, count in features.items():
            if word in self.word_weights:
                score += count * self.word_weights[word]
        
        # Apply sigmoid function
        return self.sigmoid(score)
    
    def train(self, epochs=30, callback=None):
        self.trained = False
        
        # Reset history
        history = []
        
        # Initialize weights with small random values
        self.word_weights = {'__bias__': 0}
        for word in self.vocabulary:
            self.word_weights[word] = np.random.uniform(-0.05, 0.05)
        
        # Parameters
        base_learning_rate = 0.1
        
        # Split data into training (80%) and validation (20%) sets
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(self.training_data))
        split_idx = int(len(self.training_data) * 0.8)
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]
        
        training_set = [self.training_data[i] for i in train_indices]
        validation_set = [self.training_data[i] for i in valid_indices]
        
        # Train for specified number of epochs
        for epoch in range(epochs):
            # Adaptive learning rate that decreases with epochs
            learning_rate = base_learning_rate / (1 + epoch * 0.1)
            
            # Shuffle training set for each epoch
            np.random.shuffle(train_indices)
            shuffled_training = [self.training_data[i] for i in train_indices]
            
            batch_size = 5
            total_loss = 0
            
            # Process in mini-batches
            for i in range(0, len(shuffled_training), batch_size):
                batch = shuffled_training[i:i + batch_size]
                batch_gradients = {'__bias__': 0}
                
                # Compute gradients for batch
                for example in batch:
                    features = self.extract_features(example["text"])
                    prediction = self.predict_sentiment_score(features)
                    target = example["sentiment"]
                    error = target - prediction
                    
                    # Accumulate gradient for bias
                    batch_gradients['__bias__'] = batch_gradients.get('__bias__', 0) + error
                    
                    # Accumulate gradients for features
                    for word, count in features.items():
                        batch_gradients[word] = batch_gradients.get(word, 0) + error * count
                    
                    # Compute loss for monitoring (squared error)
                    total_loss += error * error
            
            # Apply batch gradients
            self.word_weights['__bias__'] += learning_rate * batch_gradients['__bias__'] / len(batch)
            
            for word, gradient in batch_gradients.items():
                if word != '__bias__':
                    self.word_weights[word] = self.word_weights.get(word, 0) + learning_rate * gradient / len(batch)
            
            # Add L2 regularization to prevent overfitting
            regularization_rate = 0.01
            for word in self.word_weights:
                if word != '__bias__':
                    self.word_weights[word] *= (1 - regularization_rate * learning_rate)
            
            # Evaluate accuracy on validation set
            correct = 0
            for example in validation_set:
                features = self.extract_features(example["text"])
                prediction = self.predict_sentiment_score(features)
                predicted_class = 1 if prediction > 0.5 else 0
                if predicted_class == example["sentiment"]:
                    correct += 1
            
            epoch_accuracy = correct / len(validation_set)
            
            # Record history
            history.append({
                'epoch': epoch + 1,
                'accuracy': epoch_accuracy,
                'loss': total_loss / len(training_set)
            })
            
            # Call the callback function with current progress
            if callback:
                callback(epoch + 1, epochs, epoch_accuracy, history)
            
            # Pause to show progress visually
            time.sleep(0.1)
        
        self.trained = True
        return history
    
    def predict(self, text):
        if not self.trained and not self.word_weights:
            return None
        
        features = self.extract_features(text)
        score = self.predict_sentiment_score(features)
        
        predicted_sentiment = 'positive' if score > 0.5 else 'negative'
        confidence = score * 100 if score > 0.5 else (1 - score) * 100
        
        prediction = {
            'id': int(time.time() * 1000),
            'text': text,
            'sentiment': predicted_sentiment,
            'confidence': confidence,
            'score': score,
            'added_to_training': False
        }
        
        return prediction
    
    def add_to_training(self, text, sentiment_label):
        sentiment_value = 1 if sentiment_label == 'positive' else 0
        
        new_example = {
            "text": text,
            "sentiment": sentiment_value
        }
        
        self.training_data.append(new_example)
        self.create_vocabulary()  # Update vocabulary with new words
        
        return len(self.training_data)
    
    def get_important_words(self, text, top_n=5):
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
        
        # Sort by absolute weight and take top N
        word_scores.sort(key=lambda x: abs(x['weight']), reverse=True)
        return word_scores[:top_n]
    
    def get_training_data_sample(self, n=6):
        """Return a small sample of training data for display"""
        if len(self.training_data) <= n:
            return self.training_data
        return self.training_data[:n]