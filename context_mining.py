import numpy as np
import unicodedata
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet
import pickle

# Text Preprocessing class
class TextPreprocessing:
    def __init__(self):
        pass
    def clean_line(self, line): 
        line = line.replace('\n', '')
        line = line.replace(' @ @ @ @ @ @ @ @ @ @', '')
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
    
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN
    def lemmatize_pos_tagged_word(self, pos_tagged_word):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        try:
            tag = pos_tagged_word[1]
            wntag = self.get_wordnet_pos(tag)
            return lemmatizer.lemmatize(pos_tagged_word[0], wntag)
        except Exception:
            print("Exception: ", Exception)
            return pos_tagged_word[0]
    def preprocess(self, file):
        with open(file, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()
            clean_lines = [self.clean_line(line) for line in lines]
            tokenized_lines = [nltk.word_tokenize(line) for line in clean_lines]
            token_list = [word for line in tokenized_lines for word in line]
            tokens_with_pos_tagging = nltk.pos_tag(token_list)
            print("first 50 tokens with pos tagging: ", tokens_with_pos_tagging[:50])
            lemmatized_pos_tagged_tokens = [self.lemmatize_pos_tagged_word(token) for token in tokens_with_pos_tagging]
            print("lemmatized pos tagged tokens length: ", len(lemmatized_pos_tagged_tokens))
            print("first 50 lemmatized pos tagged tokens: ", lemmatized_pos_tagged_tokens[:50])
            # filter stopwords
            lemmatized_pos_tagged_tokens_without_stop_words = [token for token in lemmatized_pos_tagged_tokens if token not in stopwords.words('english')]
            print("lemmatized pos tagged tokens without stop words length: ", len(lemmatized_pos_tagged_tokens_without_stop_words))
            return lemmatized_pos_tagged_tokens_without_stop_words
    def lemmatize_word(self, word):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return lemmatizer.lemmatize(word)

    def get_vocabulary_from_tokens(self, token_list):
        set_of_tokens = {}
        for i in range(len(token_list)):
            if token_list[i] not in set_of_tokens:
                set_of_tokens[token_list[i]] = [i]
            else:
                set_of_tokens[token_list[i]].append(i)
        return set_of_tokens

class ContextMining:
    def __init__(self):
        pass
    
    def get_context_per_token(self, tokens, vocabulary, window_size):
        context_per_token = {}
        for word in vocabulary:
            context = []
            # for each index of the word in the tokens list
            # get the context of the word. Do not include the word itself
            for index in vocabulary[word]:
                # get the left context
                left_context = tokens[max(0, index - window_size):index]
                # get the right context
                right_context = tokens[index + 1:min(len(tokens), index + window_size + 1)]
                context.extend(left_context)
                context.extend(right_context)
                # eliminate the word itself from the context
            context = [token for token in context if token != word]
            context_per_token[word] = context
        return context_per_token

    # def get_context_matrix(self, tokens, vocabulary, window_size):
    #     context_matrix = np.zeros((len(vocabulary), len(vocabulary)))
    #     context_per_token = self.get_context_per_token(tokens, vocabulary, window_size)
    #     vacabulary_list = list(vocabulary.keys()).sort()
    #     for i in range(len(vacabulary_list)):
    #         print("Processing word: ", i, " / ", len(vacabulary_list))
    #         for j in range(len(vacabulary_list)):
    #             if i != j:
    #                 current_word = vacabulary_list[i]
    #                 current_context = context_per_token[vacabulary_list[j]]
    #                 # count the number of times the current word appears in the context of the current context word
    #                 context_matrix[i][j] = current_context.count(current_word)
    #     return context_matrix
    
    def make_bm_25_vectors(self, context_per_token, vocabulary):
        try:
            return pickle.load(open("checkpoints/bm25.pkl","rb"))
        except Exception:
            print("BM25 not found. Creating BM25 vectors...")
        
        s = 0

        context_per_token_values = context_per_token.values()
        for token_context in context_per_token_values:
            s += len(token_context)
        avgdl = s / len(token_context)

        vectors_bm25 = {}
        k = 1.2
        b = 0.5

        for counter, word in enumerate(vocabulary, start=1):
            print("Processing word: ", counter, " / ", len(vocabulary))
            context = context_per_token[word]
            vector_bm25 = [context.count(word) for word in vocabulary]
            counts = np.array(vector_bm25)
            numerator = counts * (k + 1)
            denominator = counts + k * (1 - b + b * len(context) / avgdl)
            bm_25 = numerator / denominator
            summation = np.sum(bm_25)
            bm_25 = bm_25 / summation
            vectors_bm25[word] = bm_25
        
        # save bm25 vectors as pkl 
        pickle.dump(vectors_bm25, open("checkpoints/bm25.pkl","wb"))

        return vectors_bm25
    
    def show_word_similarities(self, word, bm25_vectors):
        word_vector = bm25_vectors[word]
        similarities = {
            vec: np.dot(word_vector, bm25_vectors[vec]) for vec in bm25_vectors
        }
        # get the top 100 most similar words
        top_100_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:100]
        print("Top 100 most similar words to ", word, ": ", top_100_similarities)
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
            for word in data:
                file.write(word + '\n')
    def save_matrix(self, file_name, matrix, create_directory=False):
        if create_directory:
            import os
            directory = os.path.dirname(file_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
        with open(file_name, 'w') as file:
            col_separator = '|'
            for row in matrix:
                for col in row:
                    file.write(str(col) + col_separator)
                file.write('\n')
    def load_matrix(self, file_name):
        try:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                return [[int(col) for col in line.split('|')] for line in lines]
        except Exception:
            return None
    def load(self, file_name):
        try:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                return [line.replace('\n', '') for line in lines]
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
print("Tokens extracted. Length: ", len(tokens))

vocabulary = TextPreprocessing.get_vocabulary_from_tokens(tokens)
# print the first 50 words in voc 
print("Vocabulary extracted. Length: ", len(vocabulary))

ContextMining = ContextMining()
try :
    context_per_token = pickle.load(open("checkpoints/context_per_token.pkl","rb"))
    print("Context per token loaded. Length: ", len(context_per_token))
except Exception:
    print("Context per token not found. Creating context per token...")
    context_per_token = ContextMining.get_context_per_token(tokens, vocabulary, 8)
    # save to pkl
    pickle.dump(context_per_token, open("checkpoints/context_per_token.pkl","wb"))
    print("Context per token extracted. Length: ", len(context_per_token))
context_matrix = ContextMining.make_bm_25_vectors(context_per_token, vocabulary)

ContextMining.show_word_similarities('student', context_matrix)


