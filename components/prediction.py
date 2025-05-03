import streamlit as st

def show_prediction_section(model, submissions):
    st.header("2. Using the Model for Prediction")
    
    # Check if model is trained
    if not model.trained:
        st.warning("Please train the model first before making predictions.")
        return
    
    # Text input for prediction
    user_input = st.text_input("Enter text to analyze:", key="prediction_input")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        predict_button = st.button("Analyze")
    
    with col2:
        sample_button = st.button("Try a Sample")
    
    # Sample text examples
    sample_texts = [
        "I really enjoyed this presentation!",
        "This is confusing and hard to follow",
        "The examples are very clear and helpful",
        "I'm not learning anything new",
        "This demonstration is fascinating"
    ]
    
    # Process sample button
    if sample_button:
        import random
        random_sample = random.choice(sample_texts)
        st.session_state.prediction_input = random_sample
        user_input = random_sample
        predict_button = True
    
    # Make prediction
    if predict_button and user_input:
        prediction = model.predict(user_input)
        
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
            important_words = model.get_important_words(user_input)
            
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
                if st.button("✓ This is correct", key="correct_button"):
                    # Add to training data with current label
                    model.add_to_training(prediction['text'], prediction['sentiment'])
                    
                    # Update the submission's status
                    for i, submission in enumerate(st.session_state.submissions):
                        if submission['id'] == prediction['id']:
                            st.session_state.submissions[i]['added_to_training'] = True
                    
                    st.success("Added to training data with current label!")
            
            with col2:
                if prediction['sentiment'] == 'positive':
                    if st.button("✗ This is actually negative", key="wrong_button"):
                        # Add to training data with corrected label
                        model.add_to_training(prediction['text'], 'negative')
                        
                        # Update the submission's status
                        for i, submission in enumerate(st.session_state.submissions):
                            if submission['id'] == prediction['id']:
                                st.session_state.submissions[i]['added_to_training'] = True
                                st.session_state.submissions[i]['corrected'] = True
                                st.session_state.submissions[i]['originalSentiment'] = submission['sentiment']
                                st.session_state.submissions[i]['sentiment'] = 'negative'
                        
                        st.success("Added to training data with corrected label!")
                else:
                    if st.button("✗ This is actually positive", key="wrong_button"):
                        # Add to training data with corrected label
                        model.add_to_training(prediction['text'], 'positive')
                        
                        # Update the submission's status
                        for i, submission in enumerate(st.session_state.submissions):
                            if submission['id'] == prediction['id']:
                                st.session_state.submissions[i]['added_to_training'] = True
                                st.session_state.submissions[i]['corrected'] = True
                                st.session_state.submissions[i]['originalSentiment'] = submission['sentiment']
                                st.session_state.submissions[i]['sentiment'] = 'positive'
                        
                        st.success("Added to training data with corrected label!")