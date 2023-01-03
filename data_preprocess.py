import re
import string
import pickle
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = stopwords.words('english')
#lemmatizing objects
lemm = nltk.stem.WordNetLemmatizer()
spell = Speller()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
def lemmatize(text):
    return [lemm.lemmatize(w, get_wordnet_pos(w)) for w in w_tokenizer.tokenize(text)]

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

def text_preprocess(df_in):
    df_in['text'] = df_in['text'].str.lower()  # lowercase
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('@[^\s]+', '', x))  # remove mentions
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('http[^\s]+', '', x))  # remove http links
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('www.[^\s]+', '', x))  # remove www links
    df_in['text'] = df_in['text'].str.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    df_in['text'] = df_in['text'].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(text=x))
    #print(df_in['text'].head())
    df_in['text'] = df_in['text'].apply(lemmatize)
    df_in['text'] = df_in['text'].apply(lambda x: [spell(w) for w in x]) #autocorrect
    df_in['text'] = df_in['text'].apply(lambda x: [w for w in x if w not in stop_words])
    df_in['tokens'] = df_in['text'].apply(lambda x: detokenize(x))
    data = df_in['tokens'].to_numpy()
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    test_data = tokenizer.texts_to_sequences(data)
    data_padded = pad_sequences(test_data, maxlen=20)
    return data_padded
