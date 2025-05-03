import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// A simpler model without TensorFlow dependency
const MLSentimentAnalysisDemo = () => {
  // Training data - pre-labeled examples (expanded to 50)
  const [trainingData, setTrainingData] = useState([
    { text: "I love this presentation", sentiment: 1 },
    { text: "This is really helpful", sentiment: 1 },
    { text: "I don't understand this", sentiment: 0 },
    { text: "This is boring", sentiment: 0 },
    { text: "Great explanation of machine learning", sentiment: 1 },
    { text: "Too complicated for beginners", sentiment: 0 },
    { text: "I'm learning so much", sentiment: 1 },
    { text: "Unclear examples", sentiment: 0 },
    { text: "The visuals help a lot", sentiment: 1 },
    { text: "Moving too quickly", sentiment: 0 },
    // Original additional training examples
    { text: "This demonstration is amazing", sentiment: 1 },
    { text: "I'm confused by these concepts", sentiment: 0 },
    { text: "Excellent presentation style", sentiment: 1 },
    { text: "The examples don't make sense", sentiment: 0 },
    { text: "Very clear explanation", sentiment: 1 },
    { text: "I'm lost and can't follow along", sentiment: 0 },
    { text: "Best ML demo I've seen", sentiment: 1 },
    { text: "This is a waste of time", sentiment: 0 },
    { text: "The interactive elements are engaging", sentiment: 1 },
    { text: "Too much information too quickly", sentiment: 0 },
    { text: "I appreciate the step-by-step approach", sentiment: 1 },
    { text: "The content is poorly organized", sentiment: 0 },
    { text: "Very insightful and educational", sentiment: 1 },
    { text: "The presenter seems unprepared", sentiment: 0 },
    { text: "This makes machine learning accessible", sentiment: 1 },
    { text: "I'm completely lost", sentiment: 0 },
    { text: "The visualizations are excellent", sentiment: 1 },
    { text: "This topic is way too advanced", sentiment: 0 },
    { text: "I'm excited to learn more about ML", sentiment: 1 },
    { text: "The pace is frustratingly slow", sentiment: 0 },
    // 20 NEW EXAMPLES to reach 50 total
    { text: "The practical examples really solidify the concepts", sentiment: 1 },
    { text: "I can't relate this to real-world applications", sentiment: 0 },
    { text: "The instructor's enthusiasm is contagious", sentiment: 1 },
    { text: "There are too many technical terms without explanation", sentiment: 0 },
    { text: "This presentation has transformed my understanding", sentiment: 1 },
    { text: "The slides are cluttered and hard to read", sentiment: 0 },
    { text: "I appreciate how complex ideas are broken down simply", sentiment: 1 },
    { text: "The material feels outdated and irrelevant", sentiment: 0 },
    { text: "The hands-on exercises are incredibly valuable", sentiment: 1 },
    { text: "This presentation is putting me to sleep", sentiment: 0 },
    { text: "The analogies used make difficult concepts easier to grasp", sentiment: 1 },
    { text: "There's no logical flow to this information", sentiment: 0 },
    { text: "I'm impressed by how well-researched this is", sentiment: 1 },
    { text: "The speaker keeps going off on tangents", sentiment: 0 },
    { text: "This workshop exceeded my expectations", sentiment: 1 },
    { text: "The examples are too simplified to be useful", sentiment: 0 },
    { text: "The interactive demonstrations enhance the learning experience", sentiment: 1 },
    { text: "I feel like my time is being wasted", sentiment: 0 },
    { text: "The Q&A session cleared up all my doubts", sentiment: 1 },
    { text: "This presentation lacks depth and substance", sentiment: 0 }
  ]);
  
  // State variables for model
  const [wordWeights, setWordWeights] = useState({});
  const [vocabulary, setVocabulary] = useState({});
  const [isTraining, setIsTraining] = useState(false);
  const [epochs, setEpochs] = useState(30); // Changed from 10 to 30
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState([]);
  
  // User input
  const [userInput, setUserInput] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [submissions, setSubmissions] = useState([]);
  
  // NEW: Added state to track if prediction has been added to training data
  const [addedToTraining, setAddedToTraining] = useState(false);
  
  // Create vocabulary from training data
  useEffect(() => {
    createVocabulary(trainingData);
  }, [trainingData]);
  
  const createVocabulary = (data) => {
    // Track word frequencies to filter out rare words
    const wordFreq = {};
    const bigramFreq = {};
    
    // Collect frequencies
    data.forEach(item => {
      const words = item.text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
      
      // Count individual words
      words.forEach(word => {
        wordFreq[word] = (wordFreq[word] || 0) + 1;
      });
      
      // Count bigrams
      for (let i = 0; i < words.length - 1; i++) {
        const bigram = `${words[i]}_${words[i+1]}`;
        bigramFreq[bigram] = (bigramFreq[bigram] || 0) + 1;
      }
    });
    
    // Filter out uncommon words and bigrams
    const minFrequency = 2;
    const vocab = {};
    
    // Add words that appear at least minFrequency times
    Object.keys(wordFreq).forEach(word => {
      if (wordFreq[word] >= minFrequency) {
        vocab[word] = true;
      }
    });
    
    // Add bigrams that appear at least minFrequency times
    Object.keys(bigramFreq).forEach(bigram => {
      if (bigramFreq[bigram] >= minFrequency) {
        vocab[bigram] = true;
      }
    });
    
    // Add special features
    vocab['__has_negation__'] = true;
    vocab['__exclamation__'] = true;
    
    setVocabulary(vocab);
  };
  
  // Extract features from text (bag of words with some basic NLP enhancements)
  const extractFeatures = (text) => {
    const features = {};
    
    // Normalize text and split into words
    const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
    
    // Track unigrams (single words)
    words.forEach(word => {
      if (vocabulary[word]) {
        features[word] = features[word] ? features[word] + 1 : 1;
      }
    });
    
    // Add bigrams (pairs of adjacent words)
    for (let i = 0; i < words.length - 1; i++) {
      const bigram = `${words[i]}_${words[i+1]}`;
      if (vocabulary[bigram]) {
        features[bigram] = features[bigram] ? features[bigram] + 1 : 1;
      }
    }
    
    // Add some basic sentiment indicators
    const negationWords = ['not', 'no', "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't", "weren't"];
    const hasNegation = words.some(word => negationWords.includes(word));
    if (hasNegation) {
      features['__has_negation__'] = 1;
    }
    
    // Add exclamation mark feature
    if (text.includes('!')) {
      features['__exclamation__'] = 1;
    }
    
    return features;
  };
  
  // Simple logistic regression prediction
  const predictSentimentScore = (features, weights = wordWeights) => {
    let score = 0;
    
    // Add bias term
    score += weights['__bias__'] || 0;
    
    // Add weighted sum of features
    Object.keys(features).forEach(word => {
      if (weights[word]) {
        score += features[word] * weights[word];
      }
    });
    
    // Apply sigmoid function to get probability
    return 1 / (1 + Math.exp(-score));
  };
  
  // Train the model with logistic regression
  const trainModel = async () => {
    setIsTraining(true);
    setCurrentEpoch(0);
    setTrainingHistory([]);
    
    // Initialize weights with small random values
    const weights = { '__bias__': 0 };
    Object.keys(vocabulary).forEach(word => {
      weights[word] = Math.random() * 0.1 - 0.05; // Small random initial weights
    });
    
    // Parameters
    const baseLearningRate = 0.1;
    const history = [];
    
    // Split data into training (80%) and validation (20%) sets
    const shuffledData = [...trainingData].sort(() => Math.random() - 0.5);
    const splitIndex = Math.floor(shuffledData.length * 0.8);
    const trainingSet = shuffledData.slice(0, splitIndex);
    const validationSet = shuffledData.slice(splitIndex);
    
    // Train for specified number of epochs
    for (let epoch = 0; epoch < epochs; epoch++) {
      setCurrentEpoch(epoch + 1);
      
      // Adaptive learning rate that decreases with epochs
      const learningRate = baseLearningRate / (1 + epoch * 0.1);
      
      // Training loop with mini-batches
      // Shuffle training set for each epoch
      const shuffledTraining = [...trainingSet].sort(() => Math.random() - 0.5);
      
      let batchSize = 5;
      let totalLoss = 0;
      
      // Process in mini-batches
      for (let i = 0; i < shuffledTraining.length; i += batchSize) {
        const batch = shuffledTraining.slice(i, i + batchSize);
        let batchGradients = { '__bias__': 0 };
        
        // Compute gradients for batch
        for (const example of batch) {
          const features = extractFeatures(example.text);
          const prediction = predictSentimentScore(features, weights);
          const target = example.sentiment;
          const error = target - prediction;
          
          // Accumulate gradient for bias
          batchGradients['__bias__'] = (batchGradients['__bias__'] || 0) + error;
          
          // Accumulate gradients for features
          Object.keys(features).forEach(word => {
            batchGradients[word] = (batchGradients[word] || 0) + error * features[word];
          });
          
          // Compute loss for monitoring (squared error)
          totalLoss += error * error;
        }
        
        // Apply batch gradients
        weights['__bias__'] += learningRate * batchGradients['__bias__'] / batch.length;
        
        Object.keys(batchGradients).forEach(word => {
          if (word !== '__bias__') {
            weights[word] = (weights[word] || 0) + 
                            learningRate * batchGradients[word] / batch.length;
          }
        });
      }
      
      // Add L2 regularization to prevent overfitting
      const regularizationRate = 0.01;
      Object.keys(weights).forEach(word => {
        if (word !== '__bias__') {
          weights[word] = weights[word] * (1 - regularizationRate * learningRate);
        }
      });
      
      // Evaluate accuracy on validation set
      let correct = 0;
      for (const example of validationSet) {
        const features = extractFeatures(example.text);
        const prediction = predictSentimentScore(features, weights);
        const predictedClass = prediction > 0.5 ? 1 : 0;
        if (predictedClass === example.sentiment) {
          correct++;
        }
      }
      
      const epochAccuracy = correct / validationSet.length;
      setAccuracy(epochAccuracy);
      
      history.push({
        epoch: epoch + 1,
        accuracy: epochAccuracy,
        loss: totalLoss / trainingSet.length
      });
      
      setTrainingHistory([...history]);
      
      // Pause to show progress visually
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    
    setWordWeights(weights);
    setIsTraining(false);
    
    // Sort words by weight for feature importance
    const sortedWords = Object.keys(weights)
      .filter(word => word !== '__bias__')
      .sort((a, b) => Math.abs(weights[b]) - Math.abs(weights[a]))
      .slice(0, 15);
      
    console.log("Top weighted words:", sortedWords.map(word => ({ word, weight: weights[word] })));
  };
  
  // Predict sentiment for new text
  const predictSentiment = () => {
    if (Object.keys(wordWeights).length === 0 || !userInput.trim()) return;
    
    const features = extractFeatures(userInput);
    const score = predictSentimentScore(features);
    
    const predictedSentiment = score > 0.5 ? 'positive' : 'negative';
    const confidence = score > 0.5 ? score * 100 : (1 - score) * 100;
    
    const prediction = {
      id: Date.now(),
      text: userInput,
      sentiment: predictedSentiment,
      confidence: confidence,
      score: score,
      // NEW: Track if this prediction has been added to training data
      addedToTraining: false
    };
    
    setPrediction(prediction);
    setSubmissions([...submissions, prediction]);
    setUserInput('');
    setAddedToTraining(false); // Reset the added to training status
    
    return prediction;
  };
  
  // Add example to training data
  const addToTraining = (text, sentiment) => {
    const newExample = {
      text: text,
      sentiment: sentiment === 'positive' ? 1 : 0
    };
    
    const updatedTrainingData = [...trainingData, newExample];
    setTrainingData(updatedTrainingData);
    
    // Update vocabulary with new words
    createVocabulary(updatedTrainingData);
    
    // NEW: Mark the prediction as added to training data
    if (prediction && prediction.text === text) {
      setPrediction({
        ...prediction,
        addedToTraining: true
      });
      setAddedToTraining(true);
    }
    
    // Also update in submissions list
    const updatedSubmissions = submissions.map(sub => {
      if (sub.text === text) {
        return {
          ...sub,
          addedToTraining: true
        };
      }
      return sub;
    });
    setSubmissions(updatedSubmissions);
  };
  
  // Correct prediction and add to training data
  const correctPrediction = (item, correctSentiment) => {
    // First add to training data with correct label
    const newExample = {
      text: item.text,
      sentiment: correctSentiment === 'positive' ? 1 : 0
    };
    
    const updatedTrainingData = [...trainingData, newExample];
    setTrainingData(updatedTrainingData);
    
    // Update vocabulary with new words
    createVocabulary(updatedTrainingData);
    
    // Then update the submission in the list
    const updatedSubmissions = submissions.map(sub => {
      if (sub.id === item.id) {
        return {
          ...sub,
          sentiment: correctSentiment,
          corrected: true,
          originalSentiment: sub.sentiment,
          addedToTraining: true // Mark as added to training
        };
      }
      return sub;
    });
    
    setSubmissions(updatedSubmissions);
  };
  
  // Get important words for prediction
  const getImportantWords = (text) => {
    if (Object.keys(wordWeights).length === 0) return [];
    
    const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
    const wordScores = words
      .filter(word => vocabulary[word] && wordWeights[word])
      .map(word => ({
        word,
        weight: wordWeights[word],
        impact: wordWeights[word] > 0 ? 'positive' : 'negative'
      }))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 5);
    
    return wordScores;
  };
  
  // Generate sample predictions for demonstration
  const generateSamplePrediction = () => {
    const samples = [
      "I really enjoyed this presentation!",
      "This is confusing and hard to follow",
      "The examples are very clear and helpful",
      "I'm not learning anything new",
      "This demonstration is fascinating"
    ];
    
    const randomSample = samples[Math.floor(Math.random() * samples.length)];
    setUserInput(randomSample);
    
    setTimeout(() => {
      const prediction = predictSentiment();
      if (prediction) {
        // Highlight the important words that influenced this prediction
        console.log("Important words for prediction:", getImportantWords(randomSample));
      }
    }, 500);
  };
  
  // Statistics
  const positiveCount = submissions.filter(s => s.sentiment === 'positive').length;
  const negativeCount = submissions.filter(s => s.sentiment === 'negative').length;
  
  const chartData = [
    { name: 'Positive', count: positiveCount },
    { name: 'Negative', count: negativeCount }
  ];

  // Check if model is ready
  const isModelReady = Object.keys(wordWeights).length > 0;
  
  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Machine Learning Sentiment Analysis</h1>
      
      {/* Training Section */}
      <div className="mb-6 p-4 border rounded">
        <h2 className="text-xl mb-4">1. Training the Model</h2>
        
        <div className="mb-4">
          <p className="mb-2">Training examples: {trainingData.length}</p>
          <p className="mb-2">Vocabulary size: {Object.keys(vocabulary).length} words</p>
          <div className="mb-2">
            <label className="mr-2">Training epochs:</label>
            <input
              type="number"
              min="1"
              max="100"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              className="border p-1 w-16"
              disabled={isTraining}
            />
          </div>
          
          <button
            onClick={trainModel}
            disabled={isTraining}
            className={`p-2 rounded text-white ${isTraining ? 'bg-gray-400' : 'bg-blue-500'}`}
          >
            {isTraining ? `Training (Epoch ${currentEpoch}/${epochs})` : 'Train Model'}
          </button>
          
          {isTraining && (
            <div className="mt-2">
              <p>Current accuracy: {(accuracy * 100).toFixed(1)}%</p>
            </div>
          )}
        </div>
        
        {trainingHistory.length > 0 && (
          <div className="h-64 mb-4">
            <h3 className="text-lg mb-2">Training Progress</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                <YAxis 
                  yAxisId="left"
                  label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} 
                  domain={[0, 1]} 
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  label={{ value: 'Loss', angle: 90, position: 'insideRight' }} 
                />
                <Tooltip 
                  formatter={(value, name) => {
                    if (name === 'Accuracy') return (value * 100).toFixed(1) + '%';
                    return value.toFixed(3);
                  }} 
                />
                <Legend />
                <Line 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="accuracy" 
                  name="Accuracy" 
                  stroke="#8884d8" 
                  activeDot={{ r: 8 }} 
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="loss" 
                  name="Loss" 
                  stroke="#ff7300" 
                  activeDot={{ r: 8 }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        
        <div>
          <h3 className="text-lg mb-2">Training Data Examples:</h3>
          <div className="grid grid-cols-2 gap-2">
            {trainingData.slice(0, 6).map((item, index) => (
              <div 
                key={index}
                className={`p-2 rounded border ${item.sentiment === 1 ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'}`}
              >
                <p className="text-sm">{item.text}</p>
                <p className="text-xs mt-1">
                  Label: {item.sentiment === 1 ? 'Positive' : 'Negative'}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Prediction Section */}
      <div className="mb-6 p-4 border rounded">
        <h2 className="text-xl mb-4">2. Using the Model for Prediction</h2>
        
        {!isModelReady ? (
          <p className="text-orange-500">Please train the model first before making predictions.</p>
        ) : (
          <>
            <div className="mb-4">
              <p className="mb-2">Enter text to analyze:</p>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  placeholder="Type something to analyze..."
                  className="flex-1 p-2 border rounded"
                  onKeyPress={(e) => e.key === 'Enter' && predictSentiment()}
                />
                <button
                  onClick={predictSentiment}
                  className="bg-green-500 text-white p-2 rounded"
                >
                  Analyze
                </button>
              </div>
              <button
                onClick={generateSamplePrediction}
                className="mt-2 bg-gray-200 p-1 rounded text-sm"
              >
                Try a Sample Text
              </button>
            </div>
            
            {prediction && (
              <div className="mb-4">
                <h3 className="text-lg mb-2">Prediction Result:</h3>
                <div
                  className={`p-3 rounded border ${
                    prediction.sentiment === 'positive' ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'
                  }`}
                >
                  <p>"{prediction.text}"</p>
                  <p className="mt-2">
                    Sentiment: <span className="font-bold">{prediction.sentiment}</span>
                  </p>
                  <p>
                    Confidence: <span className="font-bold">{prediction.confidence.toFixed(1)}%</span>
                  </p>
                  
                  {/* Show important words that influenced the prediction */}
                  <div className="mt-2">
                    <p className="text-sm font-medium">Important words:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {getImportantWords(prediction.text).map((item, index) => (
                        <span 
                          key={index} 
                          className={`text-xs px-2 py-1 rounded ${
                            item.impact === 'positive' ? 'bg-green-200' : 'bg-red-200'
                          }`}
                        >
                          {item.word} ({item.weight > 0 ? '+' : ''}{item.weight.toFixed(2)})
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  {/* Modified add to training data button */}
                  <div className="mt-2">
                    {addedToTraining || prediction.addedToTraining ? (
                      <div className="flex items-center text-green-600">
                        <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                        <span className="text-sm">Added to training data</span>
                      </div>
                    ) : (
                      <button
                        onClick={() => addToTraining(prediction.text, prediction.sentiment)}
                        className="text-sm p-1 bg-blue-100 hover:bg-blue-200 rounded"
                      >
                        Add to training data
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
      
      {/* Results Section */}
      {submissions.length > 0 && (
        <div className="mb-6 p-4 border rounded">
          <h2 className="text-xl mb-4">3. Analysis Results</h2>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="text-lg mb-2">Sentiment Distribution</h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" name="Number of Submissions" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            <div>
              <h3 className="text-lg mb-2">Recent Submissions</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {submissions.slice().reverse().slice(0, 5).map(item => (
                  <div
                    key={item.id}
                    className={`p-2 rounded border text-sm ${
                      item.sentiment === 'positive' ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'
                    }`}
                  >
                    <p>{item.text}</p>
                    <p className="text-xs mt-1">
                      Sentiment: <span className="font-semibold">{item.sentiment}</span> 
                      ({item.confidence.toFixed(1)}% confidence)
                      {item.corrected && (
                        <span className="ml-2 text-blue-600">
                          (Corrected from {item.originalSentiment})
                        </span>
                      )}
                    </p>
                    
                    {/* Show added to training indicator */}
                    {item.addedToTraining && (
                      <div className="flex items-center text-green-600 mt-1 text-xs">
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                        <span>Added to training data</span>
                      </div>
                    )}
                    
                    {/* Feedback buttons */}
                    {!item.addedToTraining && (
                      <div className="mt-2 flex gap-2">
                        <button 
                          onClick={() => correctPrediction(item, 'positive')}
                          className="text-xs p-1 bg-green-100 hover:bg-green-200 rounded"
                        >
                          This is positive
                        </button>
                        <button 
                          onClick={() => correctPrediction(item, 'negative')}
                          className="text-xs p-1 bg-red-100 hover:bg-red-200 rounded"
                        >
                          This is negative
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="p-4 border rounded bg-gray-50">
        <h2 className="text-xl mb-2">How This Shows Machine Learning</h2>
        <ul className="list-disc pl-6">
          <li>We start with labelled training data (examples of positive and negative text)</li>
          <li>The model learns patterns from this data during training</li>
          <li>With each epoch, the model's accuracy typically improves</li>
          <li>The trained model can then analyse new text it hasn't seen before</li>
          <li>The model gives both a prediction and confidence score</li>
          <li><strong>Feedback loop:</strong> When predictions are wrong, you can correct them</li>
          <li><strong>Model improvement:</strong> Corrections are added to training data to improve future predictions</li>
          <li><strong>Continuous learning:</strong> Retrain the model with the expanded dataset to improve accuracy</li>
        </ul>
        
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
          <h3 className="text-lg mb-1">Interactive Learning Cycle</h3>
          <ol className="list-decimal pl-6">
            <li>Make predictions on new text</li>
            <li>Identify incorrect predictions</li>
            <li>Provide feedback by marking correct sentiment</li>
            <li>Add corrections to training data</li>
            <li>Retrain model with enhanced dataset</li>
            <li>Observe improved accuracy on similar future inputs</li>
          </ol>
          <div className="mt-3 flex flex-col gap-2">
            <button 
              onClick={trainModel}
              className="p-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
              disabled={isTraining}
            >
              {isTraining ? `Training (Epoch ${currentEpoch}/${epochs})` : 'Retrain With Feedback'}
            </button>
            
            {trainingData.length > 50 && (
              <div className="text-sm text-blue-800">
                <p>Added {trainingData.length - 50} feedback examples to the training data.</p>
                {trainingHistory.length > 0 && (
                  <p>Last training accuracy: {(trainingHistory[trainingHistory.length - 1].accuracy * 100).toFixed(1)}%</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLSentimentAnalysisDemo;