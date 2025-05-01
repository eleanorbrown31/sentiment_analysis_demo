import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def show_results_section():
    """Display the results and history section of the app"""
    predictions = st.session_state.predictions
    model = st.session_state.model
    
    st.header("3. Analysis Results")
    
    if not predictions:
        st.info("No predictions have been made yet. Use the prediction tab to analyse text.")
        return
    
    # Create two columns for metrics and history
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        
        # Count predictions by sentiment
        positive_count = sum(1 for p in predictions if p['sentiment'] == 'positive')
        negative_count = sum(1 for p in predictions if p['sentiment'] == 'negative')
        
        # Create a simple bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        
        categories = ['Positive', 'Negative']
        counts = [positive_count, negative_count]
        colors = ['#4CAF50', '#F44336']  # Green and red
        
        bars = ax.bar(categories, counts, color=colors)
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.set_ylabel('Number of Submissions')
        ax.set_title('Sentiment Distribution')
        
        # Display the chart
        st.pyplot(fig)
    
    with col2:
        st.subheader("Recent Submissions")
        
        # Display recent submissions with the most recent first
        recent_predictions = list(reversed(predictions))[:5]
        
        for item in recent_predictions:
            # Determine sentiment color
            sentiment_color = "green" if item["sentiment"] == "positive" else "red"
            
            # Create container
            with st.container(border=True):
                st.markdown(f"**{item['text']}**")
                
                confidence = item['confidence']
                
                st.markdown(
                    f"""
                    <p style="font-size: 0.8em; margin-top: 5px;">
                        Sentiment: <strong>{item['sentiment']}</strong>
                        ({confidence:.1f}% confidence)
                        {f"<span style='color: blue; margin-left: 10px;'>(Corrected from {item.get('original_sentiment', '')})</span>" if item.get('corrected') else ""}
                    </p>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show added to training indicator
                if item.get('added_to_training'):
                    st.markdown(
                        """
                        <div style="display: flex; align-items: center; color: green; margin-top: 5px; font-size: 0.8em;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                            <span style="margin-left: 5px;">Added to training data</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Feedback buttons - only show if not already added to training
                if not item.get('added_to_training'):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("This is positive", key=f"pos_{item['id']}"):
                            model.add_to_training(item['text'], 'positive')
                            
                            # Update item in list
                            for idx, pred in enumerate(predictions):
                                if pred['id'] == item['id']:
                                    predictions[idx] = {
                                        **pred,
                                        'sentiment': 'positive',
                                        'corrected': True,
                                        'original_sentiment': pred['sentiment'],
                                        'added_to_training': True
                                    }
                            
                            st.experimental_rerun()
                    
                    with col2:
                        if st.button("This is negative", key=f"neg_{item['id']}"):
                            model.add_to_training(item['text'], 'negative')
                            
                            # Update item in list
                            for idx, pred in enumerate(predictions):
                                if pred['id'] == item['id']:
                                    predictions[idx] = {
                                        **pred,
                                        'sentiment': 'negative',
                                        'corrected': True,
                                        'original_sentiment': pred['sentiment'],
                                        'added_to_training': True
                                    }
                            
                            st.experimental_rerun()
    
    # Display stats about corrections
    total_predictions = len(predictions)
    corrected_predictions = sum(1 for p in predictions if p.get('corrected'))
    added_to_training = sum(1 for p in predictions if p.get('added_to_training'))
    
    if corrected_predictions > 0:
        st.info(f"{corrected_predictions} out of {total_predictions} predictions were corrected by user feedback ({corrected_predictions/total_predictions:.1%}).")
    
    if added_to_training > 0:
        st.success(f"{added_to_training} predictions have been added to the training data to improve the model.")
        
        if st.button("Retrain Model with Feedback"):
            with st.spinner("Retraining model with feedback..."):
                model.train()
            st.success("Model retrained! Check the Training tab to see improved metrics.")