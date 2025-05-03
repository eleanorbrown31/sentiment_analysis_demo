import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

def show_results_section(model, submissions, training_history):
    if not submissions:
        return
        
    st.header("3. Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Distribution")
        
        # Count positives and negatives
        positive_count = sum(1 for s in submissions if s['sentiment'] == 'positive')
        negative_count = sum(1 for s in submissions if s['sentiment'] == 'negative')
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(4, 4))
        
        labels = ['Positive', 'Negative']
        sizes = [positive_count, negative_count]
        colors = ['#4CAF50', '#F44336']
        
        if positive_count > 0 or negative_count > 0:  # Only show chart if we have data
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            st.pyplot(fig)
        else:
            st.info("Make some predictions to see sentiment distribution")
        
        # Add a retraining button
        if len(model.training_data) > 25:  # Show only if we've added new data
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
                model.train(epochs=20, callback=update_progress)
                
                st.success("Model retrained with all feedback incorporated!")
    
    with col2:
        st.markdown("### Recent Submissions")
        
        if not submissions:
            st.info("No submissions yet")
        else:
            # Show the 5 most recent submissions in reverse order
            for item in list(reversed(submissions))[:5]:
                sentiment_color = "green" if item['sentiment'] == 'positive' else "red"
                
                # Create HTML for each submission
                html = f"""
                <div style="border: 1px solid {sentiment_color}; border-radius: 5px; padding: 10px; margin: 5px 0; 
                     background-color: rgba({0 if sentiment_color=='green' else 255}, {255 if sentiment_color=='green' else 0}, 0, 0.1)">
                    <p>{item['text']}</p>
                    <p style="font-size: smaller">Sentiment: <b>{item['sentiment']}</b> ({item['confidence']:.1f}% confidence)
                """
                
                # Add corrected indicator if applicable
                if item.get('corrected'):
                    original = item.get('originalSentiment', 'unknown')
                    html += f'<span style="color: blue; margin-left: 10px;">(Corrected from {original})</span>'
                
                # Add training indicator if applicable
                if item.get('added_to_training'):
                    html += f"""
                    <br><span style="color: green; display: flex; align-items: center; margin-top: 5px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" style="margin-right: 5px;">
                            <path fill="green" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path>
                        </svg>
                        Added to training data
                    </span>
                    """
                
                html += "</p></div>"
                
                st.markdown(html, unsafe_allow_html=True)