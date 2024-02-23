import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)
# data.head()

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

# sales.head()


df = pd.DataFrame()

df['Questions'] = cust[1].reset_index(drop = True)
df['Answer'] = sales[1].reset_index(drop = True)

# df

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the df
    sentences = nltk.sent_tokenize(text)

    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric
        # The code above does the following:
        # Identifies every word in the sentence
        # Turns it to a lower case
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)

    return ' '.join(preprocessed_sentences)


df['tokenized Questions'] = df['Questions'].apply(preprocess_text)
# df.head()


corpus = df['tokenized Questions'].to_list()
# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()

vectorised_corpus = tfidf_vectorizer.fit_transform(corpus)
# TDIDF is a numerical statistic used to evaluate how important a word is to a document in a collection or corpus.
# The TfidfVectorizer calculates the Tfidf values for each word in the corpus and uses them to create a matrix where each row represents a document and each column represents a word.
# The cell values in the matrix correspond to the importance of each word in each document.


def get_response(user_input):
    user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

    user_input_vector = tfidf_vectorizer.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

    similarity_scores = cosine_similarity(user_input_vector, vectorised_corpus) # .. Calculate the score of similarity between the user input vector and the corpus (df) vector

    most_similar_index = similarity_scores.argmax() # ..... Find the index of the most similar question in the corpus (df) based on cosine similarity

    return data['Answers'].iloc[most_similar_index] # ... Retrieve the corresponding answer from the df DataFrame and return it as the chatbot's response


# create greeting list
greetings = ["Hey There.... I am a creation of SEYI OLORUNHUNDO.... How can I help",
            "Hi Human.... How can I help",
            'Twale baba nla, wetin dey happen nah',
            'How far Alaye, wetin happen'
            "Good Day .... How can I help",
            "Hello There... How can I be useful to you today",
            "Hi Student.... How can I be of use"]

exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
farewell = ['Thanks....see you soon', 'Babye, See you soon', 'Bye... See you later', 'Bye... come back soon']

random_farewell = random.choice(farewell) # ---------------- Randomly select a farewell message from the list
random_greetings = random.choice(greetings) # -------- Randomly select greeting message from the list

# Test your chatbot
# # while True:
#     user_input = input("You: ")
#     if user_input.lower() in exits:
#         print(f"\nChatbot: {random_farewell}!")
#         break
#     if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
#         print(f"\nChatbot: {random_greetings}!")
#     else:
#         response = get_response(user_input)
#         print(f"\nChatbot: {response}")

 # --------------------- STREAMLIT IMPLEMENTATION  ------------
st.markdown("<h1 style = 'text-align: center; color: #176B87'>SAMSUNG CUSTOMER CARE</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>BUILT BY Seyi Olorunhundo", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('pngwing.com (2).png', caption = 'Samsung Related Chats')


def bot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vectorizer.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, vectorised_corpus)
    most_similar_index = most_similar.argmax()
    
    return df['Answer'].iloc[most_similar_index]

chatbot_greeting = [
    "Hello there, welcome to Orpheus Bot. pls ejoy your usage",
    "Hi user, This bot is created by oprheus, enjoy your usage",
    "Hi hi, How you dey my nigga",
    "Alaye mi, Abeg enjoy your usage",
    "Hey Hey, pls enjoy your usage"    
]

user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']


user_q = col2.text_input('Pls ask your Samsung related question: ')
if user_q in user_greeting:
    col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
    col2.write('Thank you for your usage. Bye')
elif user_q == '':
    st.write('')
else:
    responses = bot_response(user_q)
    col2.write(f'ChatBot:  {responses}')