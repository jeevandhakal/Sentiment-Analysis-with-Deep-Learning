# Sentiment Analysis with Deep Learning

This project is a sentiment analysis model that uses deep learning techniques to predict the sentiment (positive or negative) of movie reviews. It's based on the IMDB movie review dataset, containing 50,000 movie reviews.

## Project Overview

Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique that determines the sentiment or emotion expressed in text data. In this project, we've built a deep-learning model to classify movie reviews as positive or negative.

## Dataset

The dataset used in this project is the [IMDB Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle. It consists of 50,000 movie reviews labeled as "positive" or "negative."

## Project Components

1. **Data Preprocessing**: We cleaned and preprocessed the text data, including removing duplicates and encoding labels.

2. **Word Embeddings**: We used pre-trained word embeddings (GloVe) to convert words into numerical vectors to capture semantic meanings.

3. **Deep Learning Model**: We built a deep learning model with an embedding layer, LSTM layer, and dense layer for sentiment classification.

4. **Training**: The model was trained on a portion of the dataset and evaluated on a test set.

5. **Inference**: We provided examples of how to use the trained model to predict sentiment on new movie reviews.

## Dependencies

Make sure you have the following Python libraries installed:

- TensorFlow
- Pandas
- NumPy

## Usage

- You can train the model by running the provided Jupyter Notebook or Python script.
- To use the model for prediction, follow the example provided in the script, where you can input your own movie reviews.

## Model Saving

The trained sentiment analysis model is saved as `sentiment_analysis_model.h5`. You can load this model to make predictions on new data without retraining.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code.

## Acknowledgments

- The GloVe word embeddings used in this project are available at https://nlp.stanford.edu/projects/glove/.
- The IMDB Movie Reviews dataset is available on Kaggle.

For more details and code, please refer to the project files.
