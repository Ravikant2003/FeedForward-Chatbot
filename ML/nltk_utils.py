import nltk
import string

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = nltk.stem.porter.PorterStemmer()
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # Create a bag of words vector
    bag = [0] * len(all_words)
    for w in tokenized_sentence:
        if w in all_words:
            bag[all_words.index(w)] = 1
    return bag
