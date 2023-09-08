from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

from opendatasets import download 
from numpy import array, zeros
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

# These reviews are the one I found on the internet. 
new_reviews = [
    "My favorite film is Zodiac (2007). It is a representation of the case of the zodiac murderer, who terrorized the streetsof Nothern California in 1968. I like it for the suspense and the victims, and the investigation is very detailed and important. I love this movie (5 stars)",
    """This film has got to be the epitome of terrible writing and should be a classroom example of 'what not to do' when writing a screenplay. Why would Joshua take on (clearly) amateur writer Adam Gaines script is beyond me. Even his good directing and excellent cinematography could not save this disaster.
    Aside from the super obvious plot holes and very poor story overall, the dragged-out unnecessary dialogue made this film unbearable and extremely boring. The way too long 1h 39min film length felt like 4 hours and I found myself saying "get on with it already, who cares!" when the two leads would just ramble on about nothing relevant. This movie may have been interesting if it was a 30 min short film (which oddly enough is the only minimal writing experience Adam Gaines has).
    The acting was decent and Katia Winter was very easy on the eyes to look at, but her chemistry with Simon Quarterman was very unconvincing. Maybe it was the boring dialogue they had that made their chemistry absent.
    Even the maybe total of 10 minutes of action scenes were overly dragged out. The rest of the film was primarily useless garbage dialogue with absolutely no point to the story - start to finish.
    Don't waste your time with this one. See the trailer, and that's all the good and interesting parts you'll need to see.
    This gets a 3/10 strictly for the directing and cinematography.""",
    "Crap, crap and totally crap. Did I mention this film was totally crap? Well, it's totally crap",
    """I can see what they were trying to pull off here, and they almost did it. Emma Paunil , and Brianna Roy don't have a lot of experience between them, but there is potential for both of their careers. This venture however fell just a little short of being a complete effort though. Mostly it was the sound that had me cringing. Up until the party scene, it was horrible. There was an tinny sound happening throughout until then. I don't know why the sound engineers didn't clue into it until that party scene.
    The zany, offbeat dialogue, and story line kept me entertained enough to get through the sound issues. It fell off the rails a bit during the party scene. Was it too much to ask for the solo cups to at least appear filled with any sort of beverage? Aside from establishing an alibi, the whole scene felt disjointed.
    Overall it was a good venture for a crew with limited experience.""",
    """The visual effects are amazing. But Cameron should be better than this.
    I'm a big fan of Mr Cameron. Not only for his directing skills, but also for his screenwriting skills. But this time he seemed to have missed the goal. I know he rarely does sequels. But is this the best he could do? The story has not changed anything compared to the previous episode. Repeated crises, repeated enemies, repeated conflicts,and wait a minute,WHAT? Even repeated Titanic. Are you serious?
    As a director, he also did not reach his previous level. For a long time, the pace of the film felt too slow. Yes, the underwater scenes are phenomenal. But this is not the Blue Planet, this is a sci-fi action movie. At least it's what most audiences expect from the film, isn't it?"""
]

sequences = tokenizer.texts_to_sequences(new_reviews)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
predictions = model.predict(padded_sequences)

for i, prediction in enumerate(predictions):
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    print(f"Review: {new_reviews[i]}")
    print("Actual prediction:", prediction)
    print(f"Predicted Sentiment: {sentiment}")

# Save the trained model to a file
model.save('sentiment_analysis_model.h5')
