import re
import json
from pathlib import Path

def load_initial_training_data():
    """Load initial training data or create default examples"""
    # Default training data if no file exists
    default_data = [
        # 50 examples as shown in the React version
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
        {"text": "The content is poorly organized", "sentiment": 0},
        {"text": "Very insightful and educational", "sentiment": 1},
        {"text": "The presenter seems unprepared", "sentiment": 0},
        {"text": "This makes machine learning accessible", "sentiment": 1},
        {"text": "I'm completely lost", "sentiment": 0},
        {"text": "The visualizations are excellent", "sentiment": 1},
        {"text": "This topic is way too advanced", "sentiment": 0},
        {"text": "I'm excited to learn more about ML", "sentiment": 1},
        {"text": "The pace is frustratingly slow", "sentiment": 0},
        # 20 additional examples
        {"text": "The practical examples really solidify the concepts", "sentiment": 1},
        {"text": "I can't relate this to real-world applications", "sentiment": 0},
        {"text": "The instructor's enthusiasm is contagious", "sentiment": 1},
        {"text": "There are too many technical terms without explanation", "sentiment": 0},
        {"text": "This presentation has transformed my understanding", "sentiment": 1},
        {"text": "The slides are cluttered and hard to read", "sentiment": 0},
        {"text": "I appreciate how complex ideas are broken down simply", "sentiment": 1},
        {"text": "The material feels outdated and irrelevant", "sentiment": 0},
        {"text": "The hands-on exercises are incredibly valuable", "sentiment": 1},
        {"text": "This presentation is putting me to sleep", "sentiment": 0},
        {"text": "The analogies used make difficult concepts easier to grasp", "sentiment": 1},
        {"text": "There's no logical flow to this information", "sentiment": 0},
        {"text": "I'm impressed by how well-researched this is", "sentiment": 1},
        {"text": "The speaker keeps going off on tangents", "sentiment": 0},
        {"text": "This workshop exceeded my expectations", "sentiment": 1},
        {"text": "The examples are too simplified to be useful", "sentiment": 0},
        {"text": "The interactive demonstrations enhance the learning experience", "sentiment": 1},
        {"text": "I feel like my time is being wasted", "sentiment": 0},
        {"text": "The Q&A session cleared up all my doubts", "sentiment": 1},
        {"text": "This presentation lacks depth and substance", "sentiment": 0}
    ]
    
    # Try to load from file if it exists
    data_path = Path('data/initial_training_data.json')
    if data_path.exists():
        try:
            with open(data_path, 'r') as f:
                return json.load(f)
        except:
            return default_data
    else:
        # Ensure data directory exists
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save default data for future use
        try:
            with open(data_path, 'w') as f:
                json.dump(default_data, f, indent=2)
        except:
            pass
            
        return default_data

def create_vocabulary(data, min_frequency=2):
    """Create vocabulary from training data"""
    # Track word frequencies
    word_freq = {}
    bigram_freq = {}
    
    # Collect frequencies
    for item in data:
        text = item['text']
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        
        # Count individual words
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Count bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
    
    # Create vocabulary with words above threshold
    vocab = {}
    
    # Add words with sufficient frequency
    for word, freq in word_freq.items():
        if freq >= min_frequency:
            vocab[word] = True
            
    # Add bigrams with sufficient frequency
    for bigram, freq in bigram_freq.items():
        if freq >= min_frequency:
            vocab[bigram] = True
            
    # Add special features
    vocab['__has_negation__'] = True
    vocab['__exclamation__'] = True
    
    return vocab

def extract_features(text, vocabulary):
    """Extract features from text for model input"""
    features = {}
    
    # Normalize text
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    
    # Add single words (unigrams)
    for word in words:
        if word in vocabulary:
            features[word] = features.get(word, 0) + 1
            
    # Add word pairs (bigrams)
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        if bigram in vocabulary:
            features[bigram] = features.get(bigram, 0) + 1
            
    # Add negation feature
    negation_words = ['not', 'no', "don't", "didn't", "doesn't", 
                      "isn't", "aren't", "wasn't", "weren't"]
    if any(word in negation_words for word in words):
        features['__has_negation__'] = 1
        
    # Add exclamation mark feature
    if '!' in text:
        features['__exclamation__'] = 1
        
    return features