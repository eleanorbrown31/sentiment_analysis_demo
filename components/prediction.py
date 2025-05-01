import streamlit as st
import pandas as pd
import random

def show_prediction_section():
    """Display the prediction section of the app"""
    model = st.session_state.model
    
    st.header("2. Using the Model for Prediction")
    
    # Check if model is ready
    is_model_ready = bool(model.word_weights)
    
    if not is_model_ready:
        st.warning("Please train the model first before making predictions.")
        return
    
    # Initialize sample text in session state if it doesn't exist
    if 'sample_text' not in st.session_state:
        st.session_state.sample_text = ""
    
    # Sample button - MUST be placed BEFORE the text input widget
    if st.button("Try a Sample Text"):
        samples = [
            "I really enjoyed this presentation!",
            "This is confusing and hard to follow",
            "The examples are very clear and helpful",
            "I'm not learning anything new",
            "This demonstration is fascinating"
        ]
        st.session_state.sample_text = random.choice(samples)
    
    # Input area for prediction
    st.subheader("Enter text to analyse:")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Use the sample text as the default value
        user_input = st.text_input(
            "Type something to analyse...",
            value=st.session_state.sample_text,  # Use sample as the default
            key="user_input",
            label_visibility="collapsed"
        )
        
        # Clear the sample text after it's been used once
        if st.session_state.sample_text:
            st.session_state.sample_text = ""
    
    with col2:
        analyze_button = st.button("Analyse", use_container_width=True)
    
    # Make prediction if button is clicked or text is entered
    if analyze_button and user_input:
        st.session_state.current_prediction = model.predict(user_input)
        st.session_state.added_to_training = False
    
    # Display prediction result
    if st.session_state.current_prediction:
        prediction = st.session_state.current_prediction
        
        st.subheader("Prediction Result:")
        
        # Determine sentiment color
        sentiment_color = "green" if prediction["sentiment"] == "positive" else "red"
        
        # Create container
        with st.container(border=True):
            st.markdown(f'### "{prediction["text"]}"')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Sentiment:** {prediction['sentiment']}")
                st.markdown(f"**Confidence:** {prediction['confidence']:.1f}%")
            
            with col2:
                # Display important words
                if prediction["important_words"]:
                    st.markdown("**Important words:**")
                    
                    for word_info in prediction["important_words"]:
                        word = word_info["word"]
                        weight = word_info["weight"]
                        impact = word_info["impact"]
                        
                        bg_color = "rgba(0, 255, 0, 0.2)" if impact == "positive" else "rgba(255, 0, 0, 0.2)"
                        
                        st.markdown(
                            f"""
                            <span style="display: inline-block; padding: 2px 8px; 
                                        border-radius: 4px; margin: 2px;
                                        font-size: 0.8em; background-color: {bg_color};">
                                {word} ({'+' if weight > 0 else ''}{weight:.2f})
                            </span>
                            """,
                            unsafe_allow_html=True
                        )
            
            # Add to training data button
            if st.session_state.added_to_training or prediction.get('added_to_training', False):
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; color: green; margin-top: 10px;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                        <span style="margin-left: 5px;">Added to training data</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                if st.button("Add to training data", key="add_training_btn"):
                    # Add to training data
                    model.add_to_training(
                        prediction["text"], 
                        prediction["sentiment"]
                    )
                    
                    # Update state
                    st.session_state.added_to_training = True
                    prediction['added_to_training'] = True
                    
                    # Add to current prediction
                    st.session_state.current_prediction['added_to_training'] = True
                    
                    # Add to submission history if not already there
                    if prediction not in st.session_state.predictions:
                        st.session_state.predictions.append(prediction)
                    
                    # Show success message
                    st.success("Added to training data!")
                    st.experimental_rerun()