# IMDB-Movie-Reviews-Sentiment-Analysis-using-LSTM-Neural-Network
The Model Uses the Kaggle IMDB movie reviews dataset - Text preprocessing and tokenization - Word embeddings with Keras' Embedding layer - Deep learning model using LSTM (Long Short-Term Memory) layers - Training with validation and accuracy/loss visualization - Evaluation on test data with predictions

The notebook includes a deep learning project on **IMDB Sentiment Analysis using LSTM**, with steps like:

* Importing dependencies
* Accessing data via the Kaggle API
* Data preprocessing
* Tokenization & padding
* Building and training an LSTM model
* Evaluation and prediction

Uncovering the Aspects:
* Project title & description
* Features
* Installation & setup
* How to run the notebook
* Project structure
* Dataset source
* Model architecture
* Results
* Future work


This project performs sentiment analysis on IMDB movie reviews using an LSTM-based deep learning model. The goal is to classify reviews as either **positive** or **negative** by leveraging natural language processing (NLP) and deep learning techniques.

## Features

- Uses the **Kaggle IMDB movie reviews dataset**
- Text preprocessing and tokenization converting 5000 most frequent words into tokens while ignoring other
- padding sequence to same length of 200 tokens in each datapoint.
- Word embeddings with Keras' Embedding layer
- Deep learning model using LSTM (Long Short-Term Memory) layers
- Training with validation and accuracy/loss visualization
- Evaluation on test data with predictions

##  Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/imdb-lstm-sentiment.git
cd imdb-lstm-sentiment
````

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

> Or manually install:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn kaggle
```

3. **Kaggle API Setup**:

* Create a Kaggle account
* Generate your API token from your [Kaggle account settings](https://www.kaggle.com/account)
* Place the `kaggle.json` file in your working directory or `~/.kaggle/`
* Set environment variables in the notebook or terminal:
bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```
 Model Architecture

The model consists of:

* Embedding layer to learn word representations
* LSTM layer to capture sequential patterns
* Dense layers with dropout for classification
* Binary output (positive or negative sentiment)
* Adam optimizer, Binary cross entropy and Accuracy metrics

python
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    LSTM(units=128, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

##  Results

* Achieved high accuracy on validation and test sets,test accuracy= 0.8771,validation accuracy=0.8754
* Validation accuracy and loss
* Demonstrated example predictions with review text

---

##  Dataset

* **Source**: [Kaggle IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Size**: 50,000 labeled reviews (25k training, 25k testing)
* Balanced dataset: 50% positive and 50% negative reviews

##  How to Run

1. Open the notebook:

```
IMDB_Reviews_Sentiment_Analysis_Using_LSTM_Neural_Network.ipynb
```

2. Run all cells in order. Make sure:

   * Kaggle API is set up
   * All libraries are installed
   * Dataset is downloaded and available

##  Future Improvements

* Use bidirectional LSTM or GRU layers
* Try pretrained embeddings (e.g., GloVe or Word2Vec)
* Deploy the model via a web API (Flask/FastAPI)
* Add sentiment visualization with word clouds


This project is open source and available.

##  Acknowledgements

* [Kaggle Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* TensorFlow/Keras for deep learning


