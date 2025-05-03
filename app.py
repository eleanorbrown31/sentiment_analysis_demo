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
            {"text": "This is a waste of time", "sentiment": 0},
            {"text": "The interactive elements are engaging", "sentiment": 1},
            {"text": "Too much information too quickly", "sentiment": 0},
            {"text": "I appreciate the step-by-step approach", "sentiment": 1},
            {"text": "The content is poorly organised", "sentiment": 0},
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
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))  # Clip to avoid overflow
    
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
                if len(batch) > 0:  # Ensure we don't divide by zero
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
            
            epoch_accuracy = correct / len(validation_set) if len(validation_set) > 0 else 0
            
            # Record history
            history.append({
                'epoch': epoch + 1,
                'accuracy': epoch_accuracy,
                'loss': total_loss / len(training_set) if len(training_set) > 0 else 0
            })
            
            # Call the callback function with current progress
            if callback:
                callback(epoch + 1, epochs, epoch_accuracy, history)
            
            # Small delay to show progress visually
            time.sleep(0.1)
        
        self.trained = True
        return history
    
    def predict(self, text):
        features = self.extract_features(text)
        score = self.predict_sentiment_score(features)
        
        predicted_sentiment = 'positive' if score > 0.5 else 'negative'
        confidence = score * 100 if score > 0.5 else (1 - score) * 100
        
        prediction = {
            'id': int(time.time() * 1000) + random.randint(0, 999),  # Add randomness to avoid duplicate IDs
            'text': text,
            'sentiment': predicted_sentiment,
            'confidence': confidence,
            'score': score,
            'added_to_training': False,
            'corrected': False
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
    
    def evaluate_model_performance(self, submissions):
        """Calculate model performance metrics based on user feedback"""
        if not submissions:
            return {
                "accuracy": 0,
                "positive_accuracy": 0,
                "negative_accuracy": 0,
                "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "stats": {"total": 0, "corrected": 0, "correct": 0}
            }
        
        corrected_submissions = [s for s in submissions if s.get('corrected', False)]
        
        # Count total, corrected, and correct examples
        total = len(submissions)
        corrected = len(corrected_submissions)
        correct = total - corrected
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Initialize confusion matrix
        tp, tn, fp, fn = 0, 0, 0, 0
        
        # Count positive and negative examples
        positive_total = sum(1 for s in submissions if s['sentiment'] == 'positive')
        negative_total = sum(1 for s in submissions if s['sentiment'] == 'negative')
        
        # For corrected submissions, count confusion matrix
        for s in corrected_submissions:
            orig = s.get('originalSentiment', '')
            curr = s.get('sentiment', '')
            
            if orig == 'positive' and curr == 'negative':
                # Was positive, should be negative (false positive)
                fp += 1
            elif orig == 'negative' and curr == 'positive':
                # Was negative, should be positive (false negative)
                fn += 1
        
        # Calculate true positives and true negatives
        tp = positive_total - fn
        tn = negative_total - fp
        
        # Calculate positive and negative accuracy
        positive_accuracy = tp / (tp + fp) if (tp + fp) > 0 else 0
        negative_accuracy = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "positive_accuracy": positive_accuracy,
            "negative_accuracy": negative_accuracy,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "stats": {"total": total, "corrected": corrected, "correct": correct}
        }
    
    def get_top_feature_importance(self, n=10):
        """Get the top n most important features (words) for classification"""
        if not self.word_weights:
            return []
        
        # Get all words except bias term
        word_weights = [(word, weight) for word, weight in self.word_weights.items() 
                        if word != '__bias__']
        
        # Sort by absolute weight
        word_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top n
        return word_weights[:n]
        
    def get_confusion_terms(self, submissions):
        """Identify terms that commonly lead to misclassification"""
        if not submissions:
            return {"false_positive": [], "false_negative": []}
        
        # Get corrected submissions
        corrected = [s for s in submissions if s.get('corrected', False)]
        
        false_positives = []
        false_negatives = []
        
        # Collect words from misclassified examples
        for s in corrected:
            words = re.sub(r'[^\w\s]', '', s['text'].lower()).split()
            
            if s.get('originalSentiment') == 'positive' and s['sentiment'] == 'negative':
                false_positives.extend(words)
            elif s.get('originalSentiment') == 'negative' and s['sentiment'] == 'positive':
                false_negatives.extend(words)
        
        # Count frequencies
        fp_counter = Counter(false_positives)
        fn_counter = Counter(false_negatives)
        
        # Return top 5 most common terms for each category
        return {
            "false_positive": fp_counter.most_common(5),
            "false_negative": fn_counter.most_common(5)
        }

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = SentimentModel()
if 'submissions' not in st.session_state:
    st.session_state.submissions = []
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'sample_text' not in st.session_state:
    st.session_state.sample_text = ""
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'batch_index' not in st.session_state:
    st.session_state.batch_index = 0

# App title and description
st.title("Machine Learning Sentiment Analysis Demo")

st.markdown("""
This application demonstrates how machine learning works with a simple sentiment analysis model:
1. **Training**: Start with labelled examples of positive and negative text
2. **Testing**: See how the model performs on new text
3. **Feedback**: Provide corrections to help the model improve
""")

# Main layout with tabs
tab1, tab2, tab3 = st.tabs(["Training", "Prediction", "Results & Analysis"])

# Training tab content
with tab1:
    st.header("Training the Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Training examples:** {len(st.session_state.model.training_data)}")
        st.markdown(f"**Vocabulary size:** {len(st.session_state.model.vocabulary)} words")
        
        # Training parameters
        epochs = st.slider("Training epochs:", min_value=1, max_value=50, value=10, step=1)
        
        # Training button
        if st.button("Train Model"):
            # Progress tracking elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define callback for training progress
            def update_progress(epoch, total_epochs, accuracy, history):
                progress = epoch / total_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch}/{total_epochs} - Accuracy: {accuracy*100:.1f}%")
                st.session_state.training_history = history
            
            # Train the model
            st.session_state.model.train(epochs=epochs, callback=update_progress)
            st.success("Training completed!")
    
    with col2:
        # Show training data examples
        st.markdown("### Sample Training Data:")
        
        sample_data = st.session_state.model.get_training_data_sample(6)
        
        for item in sample_data:
            sentiment = "Positive" if item["sentiment"] == 1 else "Negative"
            color = "green" if item["sentiment"] == 1 else "red"
            
            st.markdown(
                f"""
                <div style="border: 1px solid {color}; border-radius: 5px; padding: 10px; margin: 5px 0; background-color: rgba({0 if color=='green' else 255}, {255 if color=='green' else 0}, 0, 0.1)">
                    <p>{item['text']}</p>
                    <p style="font-size: smaller">Label: <b>{sentiment}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Training history chart
    if st.session_state.training_history:
        st.markdown("### Training Progress")
        
        # Convert history to DataFrame for plotting
        history_df = pd.DataFrame(st.session_state.training_history)
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 4))
        
        # Plot accuracy on primary y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color='tab:blue')
        ax1.plot(history_df['epoch'], history_df['accuracy'], color='tab:blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create second y-axis for loss
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='tab:orange')
        ax2.plot(history_df['epoch'], history_df['loss'], color='tab:orange', marker='s')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        fig.tight_layout()
        st.pyplot(fig)

# Prediction tab content
with tab2:
    st.header("Using the Model for Prediction")
    
    # Check if model is trained
    if not st.session_state.model.trained and not st.session_state.training_history:
        st.warning("Please train the model first before making predictions.")
    else:
        # Create tabs for single vs batch prediction
        pred_tab1, pred_tab2 = st.tabs(["Single Prediction", "Batch Testing"])
        
        # Single prediction tab
        with pred_tab1:
            # Sample text examples
            sample_texts = [
                "I really enjoyed this presentation!",
                "This is confusing and hard to follow",
                "The examples are very clear and helpful",
                "I'm not learning anything new",
                "This demonstration is fascinating"
            ]
            
            # Handle the "Try a Sample" button
            if st.button("Try a Sample", key="single_sample_btn"):
                st.session_state.sample_text = random.choice(sample_texts)
            
            # Text input for prediction
            user_input = st.text_input("Enter text to analyse:", value=st.session_state.sample_text)
            
            # Make prediction
            if st.button("Analyse", key="single_analyse_btn") and user_input:
                prediction = st.session_state.model.predict(user_input)
                
                if prediction:
                    # Add to submissions list
                    st.session_state.submissions.append(prediction)
                    
                    # Display prediction result
                    st.markdown("### Prediction Result:")
                    
                    sentiment_color = "green" if prediction['sentiment'] == 'positive' else "red"
                    
                    st.markdown(
                        f"""
                        <div style="border: 1px solid {sentiment_color}; border-radius: 5px; padding: 15px; 
                             background-color: rgba({0 if sentiment_color=='green' else 255}, {255 if sentiment_color=='green' else 0}, 0, 0.1)">
                            <p style="font-size: 16px">"{prediction['text']}"</p>
                            <p>Sentiment: <b>{prediction['sentiment']}</b></p>
                            <p>Confidence: <b>{prediction['confidence']:.1f}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Show important words
                    important_words = st.session_state.model.get_important_words(user_input)
                    
                    if important_words:
                        st.markdown("#### Important words that influenced the prediction:")
                        
                        word_html = ""
                        for word_info in important_words:
                            word = word_info['word']
                            weight = word_info['weight']
                            impact = word_info['impact']
                            
                            word_color = "green" if impact == 'positive' else "red"
                            weight_sign = '+' if weight > 0 else ''
                            
                            word_html += f"""
                            <span style="background-color: rgba({0 if word_color=='green' else 255}, {255 if word_color=='green' else 0}, 0, 0.2); 
                                   padding: 3px 8px; margin: 2px; border-radius: 12px; display: inline-block;">
                                {word} ({weight_sign}{weight:.2f})
                            </span>
                            """
                        
                        st.markdown(f"<div>{word_html}</div>", unsafe_allow_html=True)
                    
                    # Feedback buttons
                    st.markdown("#### Provide feedback to improve the model:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✓ This is correct", key="correct_button_single"):
                            # Add to training data with current label
                            st.session_state.model.add_to_training(prediction['text'], prediction['sentiment'])
                            
                            # Update the submission's status
                            for i, submission in enumerate(st.session_state.submissions):
                                if submission['id'] == prediction['id']:
                                    st.session_state.submissions[i]['added_to_training'] = True
                            
                            st.success("Added to training data with current label!")
                    
                    with col2:
                        if prediction['sentiment'] == 'positive':
                            if st.button("✗ This is actually negative", key="wrong_button_single"):
                                # Add to training data with corrected label
                                st.session_state.model.add_to_training(prediction['text'], 'negative')
                                
                                # Update the submission's status
                                for i, submission in enumerate(st.session_state.submissions):
                                    if submission['id'] == prediction['id']:
                                        st.session_state.submissions[i]['added_to_training'] = True
                                        st.session_state.submissions[i]['corrected'] = True
                                        st.session_state.submissions[i]['originalSentiment'] = submission['sentiment']
                                        st.session_state.submissions[i]['sentiment'] = 'negative'
                                
                                st.success("Added to training data with corrected label!")
                        else:
                            if st.button("✗ This is actually positive", key="wrong_button_single"):
                                # Add to training data with corrected label
                                st.session_state.model.add_to_training(prediction['text'], 'positive')
                                
                                # Update the submission's status
                                for i, submission in enumerate(st.session_state.submissions):
                                    if submission['id'] == prediction['id']:
                                        st.session_state.submissions[i]['added_to_training'] = True
                                        st.session_state.submissions[i]['corrected'] = True
                                        st.session_state.submissions[i]['originalSentiment'] = submission['sentiment']
                                        st.session_state.submissions[i]['sentiment'] = 'positive'
                                
                                st.success("Added to training data with corrected label!")
        
        # Batch prediction tab
        with pred_tab2:
            st.markdown("### Batch Testing")
            st.markdown("""
            Test multiple examples at once to see how well your model performs. 
            Enter up to 10 examples (one per line) or use random samples.
            """)
            
            # Batch text input
            batch_text = st.text_area(
                "Enter up to 10 examples (one per line):", 
                height=150,
                placeholder="Enter one example per line or click 'Generate 10 Samples'"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Generate batch samples
                if st.button("Generate 10 Samples"):
                    batch_samples = [
                        "This was very informative and helpful",
                        "I didn't understand most of the examples",
                        "The presentation was engaging and clear",
                        "Too much information in too little time",
                        "I learned a lot from this workshop",
                        "The speaker seemed unprepared",
                        "Great examples that made sense",
                        "The content was too basic for me",
                        "I'm excited to try these techniques",
                        "The slides were confusing and cluttered"
                    ]
                    batch_text = "\n".join(batch_samples)
                    # Force a rerun to update the text area
                    st.experimental_rerun()
            
            with col2:
                # Analyse batch
                if st.button("Analyse Batch") and batch_text:
                    # Split text into lines and filter out empty lines
                    lines = [line.strip() for line in batch_text.split("\n") if line.strip()]
                    
                    # Limit to 10 examples
                    lines = lines[:10]
                    
                    # Process each example
                    batch_results = []
                    for line in lines:
                        prediction = st.session_state.model.predict(line)
                        batch_results.append(prediction)
                    
                    # Save results to session state
                    st.session_state.batch_results = batch_results
                    st.session_state.batch_index = 0
                    
                    # Force a rerun to show the results
                    st.experimental_rerun()
            
            # Display batch results
            if st.session_state.batch_results:
                st.markdown("### Batch Results")
                
                # Create a progress bar to show which example we're on
                total_examples = len(st.session_state.batch_results)
                current_index = st.session_state.batch_index
                
                st.progress((current_index + 1) / total_examples)
                st.markdown(f"**Example {current_index + 1} of {total_examples}**")
                
                # Get current example
                current_example = st.session_state.batch_results[current_index]
                
                # Display current example
                sentiment_color = "green" if current_example['sentiment'] == 'positive' else "red"
                
                st.markdown(
                    f"""
                    <div style="border: 1px solid {sentiment_color}; border-radius: 5px; padding: 15px; 
                         background-color: rgba({0 if sentiment_color=='green' else 255}, {255 if sentiment_color=='green' else 0}, 0, 0.1)">
                        <p style="font-size: 16px">"{current_example['text']}"</p>
                        <p>Sentiment: <b>{current_example['sentiment']}</b></p>
                        <p>Confidence: <b>{current_example['confidence']:.1f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Feedback buttons for batch
                st.markdown("#### Is this prediction correct?")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("✓ Correct", key="correct_batch"):
                        # Add example to submissions with current label
                        st.session_state.submissions.append(current_example)
                        
                        # Add to training data
                        st.session_state.model.add_to_training(
                            current_example['text'], 
                            current_example['sentiment']
                        )
                        
                        # Mark as added to training
                        st.session_state.batch_results[current_index]['added_to_training'] = True
                        
                        # Move to next example
                        if current_index < total_examples - 1:
                            st.session_state.batch_index += 1
                            st.experimental_rerun()
                        else:
                            st.success("You've reviewed all examples!")
                
                with col2:
                    # Add negative feedback
                    if current_example['sentiment'] == 'positive':
                        if st.button("✗ Actually Negative", key="negative_batch"):
                            # Create corrected version
                            corrected_example = dict(current_example)
                            corrected_example['corrected'] = True
                            corrected_example['originalSentiment'] = corrected_example['sentiment']
                            corrected_example['sentiment'] = 'negative'
                            
                            # Add to submissions and training data
                            st.session_state.submissions.append(corrected_example)
                            st.session_state.model.add_to_training(corrected_example['text'], 'negative')
                            
                            # Mark as added
                            st.session_state.batch_results[current_index]['added_to_training'] = True
                            st.session_state.batch_results[current_index]['corrected'] = True
                            
                            # Move to next example
                            if current_index < total_examples - 1:
                                st.session_state.batch_index += 1
                                st.experimental_rerun()
                            else:
                                st.success("You've reviewed all examples!")
                    else:
                        if st.button("✗ Actually Positive", key="positive_batch"):
                            # Create corrected version
                            corrected_example = dict(current_example)
                            corrected_example['corrected'] = True
                            corrected_example['originalSentiment'] = corrected_example['sentiment']
                            corrected_example['sentiment'] = 'positive'
                            
                            # Add to submissions and training data
                            st.session_state.submissions.append(corrected_example)
                            st.session_state.model.add_to_training(corrected_example['text'], 'positive')
                            
                            # Mark as added
                            st.session_state.batch_results[current_index]['added_to_training'] = True
                            st.session_state.batch_results[current_index]['corrected'] = True
                            
                            # Move to next example
                            if current_index < total_examples - 1:
                                st.session_state.batch_index += 1
                                st.experimental_rerun()
                            else:
                                st.success("You've reviewed all examples!")
                
                with col3:
                    # Skip button
                    if st.button("Skip", key="skip_batch"):
                        if current_index < total_examples - 1:
                            st.session_state.batch_index += 1
                            st.experimental_rerun()
                        else:
                            st.info("You've reached the last example!")
                
                # Navigation buttons
                nav_col1, nav_col2 = st.columns([1, 1])
                with nav_col1:
                    if current_index > 0:
                        if st.button("Previous", key="prev_batch"):
                            st.session_state.batch_index -= 1
                            st.experimental_rerun()
                with nav_col2:
                    if current_index < total_examples - 1:
                        if st.button("Next", key="next_batch"):
                            st.session_state.batch_index += 1
                            st.experimental_rerun()
                
                # Show batch summary
                st.markdown("### Batch Summary")
                
                # Count completed examples
                completed = sum(1 for ex in st.session_state.batch_results if ex.get('added_to_training', False))
                
                st.markdown(f"**Reviewed: {completed}/{total_examples}**")
                
                # Add to training button
                if completed > 0 and completed < total_examples:
                    if st.button("Add All Reviewed Examples to Training Data"):
                        # Add all completed examples to submissions if not already there
                        for ex in st.session_state.batch_results:
                            if ex.get('added_to_training', False) and not any(s['id'] == ex['id'] for s in st.session_state.submissions):
                                st.session_state.submissions.append(ex)
                        
                        st.success(f"Added {completed} examples to training data!")

# Results tab content
with tab3:
    st.header("Results & Analysis")
    
    if not st.session_state.submissions:
        st.info("Make some predictions first to see analysis here.")
    else:
        # Calculate model performance metrics
        metrics = st.session_state.model.evaluate_model_performance(st.session_state.submissions)
        
        # Create columns for metrics
        st.markdown("### Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Accuracy", f"{metrics['accuracy']*100:.1f}%")
        
        with col2:
            st.metric("Positive Accuracy", f"{metrics['positive_accuracy']*100:.1f}%")
        
        with col3:
            st.metric("Negative Accuracy", f"{metrics['negative_accuracy']*100:.1f}%")
        
        # Add a more detailed metrics view
        with st.expander("Detailed Performance Metrics"):
            # Show confusion matrix
            cm = metrics['confusion_matrix']
            
            st.markdown("#### Confusion Matrix")
            
            # Create a dataframe for the confusion matrix
            cm_df = pd.DataFrame([
                [cm['tp'], cm['fp']],
                [cm['fn'], cm['tn']]
            ], index=['Predicted Positive', 'Predicted Negative'], 
               columns=['Actually Positive', 'Actually Negative'])
            
            st.dataframe(cm_df)
            
            # Add prediction statistics
            st.markdown("#### Prediction Statistics")
            
            stats = metrics['stats']
            st.markdown(f"""
            - **Total predictions:** {stats['total']}
            - **Correct predictions:** {stats['correct']} ({(stats['correct']/stats['total']*100):.1f}% if total > 0 else 0}%)
            - **Corrected predictions:** {stats['corrected']} ({(stats['corrected']/stats['total']*100):.1f}% if total > 0 else 0}%)
            """)
        
        # Corrected examples analysis
        st.markdown("### Corrected Examples Analysis")
        
        # Find examples that were corrected
        corrected_examples = [s for s in st.session_state.submissions if s.get('corrected', False)]
        
        if corrected_examples:
            st.markdown(f"Found **{len(corrected_examples)}** examples where the model prediction was corrected:")
            
            for example in corrected_examples:
                orig_sentiment = example.get('originalSentiment', 'unknown')
                current_sentiment = example.get('sentiment', 'unknown')
                orig_color = "green" if orig_sentiment == 'positive' else "red"
                current_color = "green" if current_sentiment == 'positive' else "red"
                
                st.markdown(
                    f"""
                    <div style="border: 1px solid {current_color}; border-radius: 5px; padding: 10px; margin: 10px 0; 
                         background-color: rgba({0 if current_color=='green' else 255}, {255 if current_color=='green' else 0}, 0, 0.1)">
                        <p>"{example['text']}"</p>
                        <p>
                            <span style="color: {orig_color}">
                                Initially predicted: <b>{orig_sentiment}</b> ({example['confidence']:.1f}% confidence)
                            </span>
                            <br>
                            <span style="color: {current_color}">
                                Corrected to: <b>{current_sentiment}</b>
                            </span>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Get features that commonly lead to confusion
            confusion_terms = st.session_state.model.get_confusion_terms(st.session_state.submissions)
            
            st.markdown("#### Common Confusion Terms")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**False Positives (predicted positive but actually negative):**")
                if confusion_terms['false_positive']:
                    for term, count in confusion_terms['false_positive']:
                        st.markdown(f"- {term} ({count} occurrences)")
                else:
                    st.markdown("*No data yet*")
            
            with col2:
                st.markdown("**False Negatives (predicted negative but actually positive):**")
                if confusion_terms['false_negative']:
                    for term, count in confusion_terms['false_negative']:
                        st.markdown(f"- {term} ({count} occurrences)")
                else:
                    st.markdown("*No data yet*")
        else:
            st.info("No corrected predictions yet. Provide feedback on predictions to see analysis here.")
        
        # Feature importance
        st.markdown("### Feature Importance Analysis")
        
        # Get top features
        top_features = st.session_state.model.get_top_feature_importance(n=15)
        
        if top_features:
            # Split into positive and negative features
            positive_features = [(word, weight) for word, weight in top_features if weight > 0]
            negative_features = [(word, weight) for word, weight in top_features if weight < 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Positive Features:**")
                for word, weight in positive_features[:7]:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(0, 255, 0, {min(abs(weight)*5, 0.3)}); 
                             padding: 5px; margin: 2px; border-radius: 3px;">
                            {word}: +{weight:.3f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.markdown("**Top Negative Features:**")
                for word, weight in negative_features[:7]:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(255, 0, 0, {min(abs(weight)*5, 0.3)}); 
                             padding: 5px; margin: 2px; border-radius: 3px;">
                            {word}: {weight:.3f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("Train the model first to see feature importance.")
        
        # Model improvement recommendations
        st.markdown("### Model Improvement Recommendations")
        
        # Check if retraining is needed
        retraining_needed = len(corrected_examples) >= 3
        
        # Accuracy threshold
        good_accuracy = metrics['accuracy'] >= 0.8
        
        if not good_accuracy:
            st.warning("""
            **Model accuracy is below 80%**
            
            Consider these improvements:
            1. Add more training examples, especially for cases where the model is confused
            2. Retrain the model with more epochs
            3. Review the confused terms above and provide more examples of those
            """)
        
        if retraining_needed:
            st.warning("""
            **Multiple predictions have been corrected**
            
            The model would likely benefit from retraining with the new feedback.
            """)
            
            if st.button("Retrain Model with Feedback", key="retrain_button"):
                st.info("Retraining model with all feedback data...")
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Define callback for training progress
                def update_progress(epoch, total_epochs, accuracy, history):
                    progress = epoch / total_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch}/{total_epochs} - Accuracy: {accuracy*100:.1f}%")
                    st.session_state.training_history = history
                
                # Train the model
                st.session_state.model.train(epochs=20, callback=update_progress)
                
                st.success("Model retrained with all feedback incorporated!")
        
        # Learning cycle explanation
        st.markdown("""
        ### How This Demonstrates Machine Learning
        
        This demo illustrates the core **training-testing-improvement** cycle of supervised ML:
        
        1. **Training Phase** - The model learns patterns from labeled examples
        2. **Testing Phase** - The model makes predictions on new data
        3. **Feedback Loop** - Human feedback corrects errors and improves the model
        4. **Continuous Learning** - The model is retrained with new examples
        
        With each iteration of this cycle:
        
        - **Accuracy increases**: The model makes fewer mistakes as training examples grow
        - **Feature importance evolves**: The model learns which words are most predictive
        - **Confusion decreases**: The model learns from its mistakes 
        - **Generalizes better**: The model performs well on new, unseen text
        
        This simulates how real-world ML systems improve through human feedback, but simplified
        to make the process visible and interactive. In production systems, this feedback loop
        can involve millions of examples and complex deep learning models.
        """)