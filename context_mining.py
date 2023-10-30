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

    def get_vocabulary_from_tokens(self, token_list):
        set_of_tokens = set()
        for line in token_list:
            for token in line:
                set_of_tokens.add(token[0])
        return [token for token in token_list if token not in set_of_tokens]

class ContextMining:
    def __init__(self):
        pass
    def get_context(self, tokens, word, window_size):
        context = []
        for i in range(len(tokens)):
            if tokens[i] == word:
                for j in range(i-window_size, i+window_size+1):
                    if j >= 0 and j < len(tokens) and j != i:
                        context.append(tokens[j])
        return context 
        
    def get_context_matrix(self, tokenized_lines, vocabulary, window_size):
        context_matrix = np.zeros((len(vocabulary), len(vocabulary)))
        for i in range(len(vocabulary)):
            for j in range(len(vocabulary)):
                if i != j:
                    context_matrix[i][j] = self.get_context(tokenized_lines, vocabulary[i], window_size).count(vocabulary[j])
        return context_matrix
    
class Checkpoint:
    def __init__(self): 
        pass
    def save(self, file_name, data, create_directory=False):
        if create_directory:
            import os
            directory = os.path.dirname(file_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
        with open(file_name, 'w') as file:
            for line in data:
                for token in line:
                    file.write(f'{token[0]} ')
                file.write('\n')
    
    def load(self, file_name):
        try:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                return [list(line.split()) for line in lines]
        except Exception:
            return None

ORIGINAL_FILE_NAME = './inputs/original.txt'


TextPreprocessing = TextPreprocessing()
CheckpointManager = Checkpoint()

if CheckpointManager.load('./checkpoints/tokens.txt') is None:
    tokens = TextPreprocessing.preprocess(ORIGINAL_FILE_NAME)
    CheckpointManager.save('./checkpoints/tokens.txt', tokens, create_directory=True)
else:
    tokens = CheckpointManager.load('./checkpoints/tokens.txt')

print(tokens[:50])
vocabulary = TextPreprocessing.get_vocabulary_from_tokens(tokens)
ContextMining = ContextMining()
context_matrix = ContextMining.get_context_matrix(tokens, vocabulary, 8)

