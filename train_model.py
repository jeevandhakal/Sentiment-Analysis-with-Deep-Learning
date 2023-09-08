from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

from opendatasets import download 
from numpy import array, zeros, save
from pandas import read_csv

# Download the data from kaggle.com
dataset_url = 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews'
download(dataset_url)

# Read the csv file
df = read_csv('./imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

# Drop the dublicates since they are of no use
df.drop_duplicates(subset='review', keep='first', inplace=True)

# Split data into features (reviews) and labels (sentiments)
reviews = df['review'].values
sentiments = df['sentiment'].values

# Encode labels (e.g., 'positive' -> 1, 'negative' -> 0)
label_map = {'positive': 1, 'negative': 0}
labels = array([label_map[s] for s in sentiments])

# Hyperparameters
embedding_dim = 100  # Should match the dimension of your GloVe vectors (e.g., 100d)
max_length = 200  # Adjust based on your data
num_epochs = 5

# Initialize GloVe embeddings
glove_model = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = array(values[1:], dtype='float32')
        glove_model[word] = vector

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Prepare word embeddings matrix
embedding_matrix = zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = glove_model.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Split the data into training and testing sets
train_size = int(0.8 * len(reviews))
X_train, X_test = padded_sequences[:train_size], padded_sequences[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# Create a sequential model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the trained model to a file
model.save('sentiment_analysis_model.h5')

# Save the tokenizer's word index to a Numpy file
save('tokenizer_word_index.npy', tokenizer.word_index)
