from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import load


# Load the saved model
model = load_model('./models/sentiment_analysis_model.h5')

max_length = 200  

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

# Initialize a tokenizer with the same settings used during training
tokenizer = Tokenizer()

# Load the vocabulary created during training
tokenizer.word_index = load('./tokens/tokenizer_word_index.npy', allow_pickle=True).item()

# Tokenize the new reviews
sequences = tokenizer.texts_to_sequences(new_reviews)

# Pad the sequences to the same length as used during training
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Make predictions
predictions = model.predict(padded_sequences)

# Interpret predictions
sentiments = ['positive' if prediction > 0.5 else 'negative' for prediction in predictions]

# Print the results
for i, review in enumerate(new_reviews):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiments[i]}\n")