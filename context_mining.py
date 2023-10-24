import numpy as np
import unicodedata
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter

# Text Preprocessing class
class TextPreprocessing:
    def __init__(self):
        pass
    def clean_line(self, line): 
        line = line.replace('\n', '')
        line = line.replace('@ @ @ @ @ @ @ @ @ @', '')
        line = self.transform_accented_chars(line)
        line = self.remove_stopwords_from_line(line)
        line = line.lower()
        return line
    def remove_stopwords_from_line(self, line):
        stop_words = set(stopwords.words('english'))
        return " ".join([word for word in line.split() if word not in stop_words])
    def transform_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text
    def preprocess(self, file):
        with open(file, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()
            clean_lines = [self.clean_line(line) for line in lines]
            tokenized_lines = [nltk.word_tokenize(line) for line in clean_lines]

            tokens_with_pos_tagging = [nltk.pos_tag(line) for line in tokenized_lines]
            lemmatized_pos_tagged_tokens = [[(self.lemmatize_word(word), pos_tag) for (word, pos_tag) in line] for line in tokens_with_pos_tagging]
            # filter stopwords
            return  [[(word, pos_tag) for (word, pos_tag) in line if word not in set(stopwords.words('english'))] for line in lemmatized_pos_tagged_tokens]
    def lemmatize_word(self, word):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return lemmatizer.lemmatize(word)

    def get_vocabulary_from_tokenized_lines(self, tokenized_lines):
        vocabulary = set()
        for line in tokenized_lines:
            for word in line.split():
                vocabulary.add(word)
        return vocabulary

class ContextMining:
    def __init__(self):
        pass
    def get_context(self, tokens, word, window_size):
        
        
    def get_context_matrix(self, tokenized_lines, vocabulary, window_size):
        context_matrix = np.zeros((len(vocabulary), len(vocabulary)))
        for i in range(len(vocabulary)):
            for j in range(len(vocabulary)):
                if i != j:
                    context_matrix[i][j] = self.get_context(tokenized_lines, vocabulary[i], window_size).count(vocabulary[j])
        return context_matrix
    

ORIGINAL_FILE_NAME = './inputs/original.txt'

TextPreprocessing = TextPreprocessing()
tokens = TextPreprocessing.preprocess(ORIGINAL_FILE_NAME)
ContextMining = ContextMining()
word_context
vocabulary = TextPreprocessing.get_vocabulary_from_tokenized_lines(tokens)