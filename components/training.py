import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

def show_training_section():
    """Display the training section of the app"""
    model = st.session_state.model
    
    st.header("1. Training the Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Training examples:** {len(model.training_data)}")
        st.markdown(f"**Vocabulary size:** {len(model.vocabulary)} words")
        
        epochs = st.number_input(
            "Training epochs:",
            min_value=1,
            max_value=100,
            value=model.epochs,
            step=5,
            disabled=model.is_training
        )
        
        # Update model epochs if changed
        if epochs != model.epochs:
            model.epochs = epochs
        
        if st.button("Train Model", disabled=model.is_training):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback function
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Training (Epoch {current}/{total})")
            
            # Train the model
            with st.spinner("Training model..."):
                important_words = model.train(update_progress)
            
            # Reset progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show success message
            st.success(f"Model trained successfully! Current accuracy: {model.accuracy:.1%}")
            
            # Display important words
            if important_words:
                st.markdown("### Top weighted words:")
                word_df = pd.DataFrame(
                    important_words[:10], 
                    columns=["Word", "Weight"]
                )
                st.dataframe(word_df)
    
    with col2:
        if model.is_training:
            st.markdown(f"**Current accuracy:** {model.accuracy:.1%}")
    
    # Display training history chart if available
    if model.training_history:
        st.subheader("Training Progress")
        
        # Create a figure and axis
        fig, ax1 = plt.subplots(figsize=(10, 4))
        
        # Prepare data
        epochs = [item['epoch'] for item in model.training_history]
        accuracy = [item['accuracy'] for item in model.training_history]
        loss = [item['loss'] for item in model.training_history]
        
        # Plot accuracy
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color=color)
        line1 = ax1.plot(epochs, accuracy, color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1)
        
        # Create second y-axis for loss
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Loss', color=color)
        line2 = ax2.plot(epochs, loss, color=color, label='Loss')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center')
        
        # Show the plot
        st.pyplot(fig)
    
    # Display sample training data
    st.subheader("Training Data Examples")
    
    # Create a 2-column grid for examples
    example_cols = st.columns(2)
    
    # Get a few random examples of each class
    positive_examples = [item for item in model.training_data if item["sentiment"] == 1]
    negative_examples = [item for item in model.training_data if item["sentiment"] == 0]
    
    # Show 3 examples of each class (6 total)
    np.random.shuffle(positive_examples)
    np.random.shuffle(negative_examples)
    
    samples = (positive_examples[:3] + negative_examples[:3])
    np.random.shuffle(samples)
    
    for i, example in enumerate(samples):
        col_idx = i % 2
        with example_cols[col_idx]:
            sentiment = "Positive" if example["sentiment"] == 1 else "Negative"
            color = "green" if example["sentiment"] == 1 else "red"
            st.markdown(
                f"""
                <div style="padding: 10px; border: 1px solid {color}; 
                           border-radius: 5px; margin-bottom: 10px;
                           background-color: rgba({0 if color=='green' else 255}, 
                                                {255 if color=='green' else 0}, 0, 0.1);">
                    <p>{example['text']}</p>
                    <p style="font-size: 0.8em; margin-top: 5px;">
                        Label: <strong>{sentiment}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )