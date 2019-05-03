#!/usr/bin/env python
# coding: utf-8

# ## Problem 2 

# In[129]:


import sys
import requests
import wget
from scraper import PaperParser
import os
from pdf_parser import PDFParser
import subprocess
from heapq import heappush, heappop

def count_words(run_once=False):
    txt_directory = '/Users/samuelkaeser/Documents/University/Classes/EE_460J/Homework/Lab5/txts'
    txts = [f for f in os.listdir(txt_directory)]
    parser = PDFParser()
    for txt in txts:
        txt_path = os.path.join(txt_directory, txt)
        with open(txt_path, 'r') as f:
            for line in f:
                parser.parse(line)
        if run_once:
            break
    return parser.word_counts, parser.total_words

def heap_sort_words(word_dict):
    h = []
    for key in word_dict.keys():
        heappush(h, (word_dict[key], key))
    return [heappop(h) for i in range(len(h))]
        
def print_top_n_words(words, n):
    print('The most frequent {0} words are : '.format(n))
    for i in range(len(words) - n, len(words)):
        val = words[i]
        print('{0}. {1}'.format(len(words) - i, val))
        
def get_word_probabilities(word_dict, total_words):
    word_probs = {}
    for word in word_dict.keys():
        word_probs[word] = word_dict[word] / total_words
    return word_probs


# In[130]:


word_dict, total_words = count_words(run_once=False)


# ### 2.1

# In[131]:


h = heap_sort_words(word_dict)
print_top_n_words(h, 10)


# In[132]:


word_probs = get_word_probabilities(word_dict, total_words)
h = heap_sort_words(word_probs)
print_top_n_words(h, 10)


# ### 2.2

# In[135]:


import math

def get_entropy(h):
    entropy = 0.0
    for prob, word in h:
        entropy += (-1.0)*prob*math.log(prob,2)
    return entropy


# In[136]:


entropy = get_entropy(h)
print('The entropy of a word in the dataset is {0}'.format(entropy))


# ### 2.3

# In[11]:


import random
import time

random.seed(time.time())

def select_word(h):
    thresh = random.random()
    prob_sum = 0.0
    for prob, word in h:
        prob_sum += prob
        if thresh < prob_sum:
            return word
        
def generate_paragraph(h, n=30):
    words = []
    for _ in range(n):
        words.append(select_word(h))
    paragraph = ' '.join(words)
    return paragraph


# In[22]:


paragraph = generate_paragraph(h)
print(paragraph)


# ### 2.4

# In[113]:


class NGramLayer():
    def __init__(self, layer):
        self.word_dict = {}
        self.word_probs = {}
        self.sorted_words = []
        self.total_words = 0
        self.layer = layer

    def add_words(self, word_window):
        # Recursively increments a word counter in its given word context
        word = word_window[0]
        if len(word_window) != self.layer:
            return
        if len(word_window) == 1:
            if word in self.word_dict.keys():
                self.word_dict[word] += 1
            else:
                self.word_dict[word] = 1
        else:
            if word in self.word_dict.keys():
                self.word_dict[word].add_words(word_window[1:])
            else:
                self.word_dict[word] = NGramLayer(layer=self.layer - 1)
                self.word_dict[word].add_words(word_window[1:])
        self.total_words += 1

    def calculate_probs(self):
        # Recursively calcuates the marginal probability of each word in the layers word context
        if self.layer == 1:
            for word in self.word_dict.keys():
                self.word_probs[word] = self.word_dict[word] / self.total_words
        else:
            for word in self.word_dict.keys():
                self.word_probs[word] = self.word_dict[word].total_words / self.total_words
                self.word_dict[word].calculate_probs()

    def get_sorted_probs(self):
        # Recursively generates lists of words sorted by their probability in the current context
        if self.layer == 1:
            h = []
            for key in self.word_probs.keys():
                heappush(h, (self.word_probs[key], key))
            self.sorted_words = [heappop(h) for i in range(len(h))]
            return self.sorted_words
        else:
            h = []
            for key in self.word_probs.keys():
                heappush(h, (self.word_probs[key], key))
                self.word_dict[key].get_sorted_probs()
            self.sorted_words = [heappop(h) for i in range(len(h))]
            return self.sorted_words

    def select_word(self, prev):
        # Selects a new word to follow a word context
        if len(prev) == 0:
            thresh = random.random()
            prob_sum = 0.0
            for prob, word in reversed(self.sorted_words):
                prob_sum += prob
                if thresh < prob_sum:
                    return word
        elif len(prev) > 1:
            word = prev[0]
            return self.word_dict[word].select_word(prev[1:])
        elif len(prev) == 1:
            word = prev[0]
            return self.word_dict[word].select_word([])


class NGramModel():
    ALPHABET = set('abcdefghijklmnopqrstuvwxyz \'')
    STRIP_SET = '*!?[]{}/\'\":;\n,'
    PUNC_SET = {'.'}

    def __init__(self, layers=2):
        # Initializes an NGramModel with N layers
        self.layers = layers
        self.word_dict = {}
        self.word_window = []
        self.sorted_words = []
        self.total_words = 0
        self.word_probs = {}

    def parse(self, text):
        # Parses through a string and counts the number of times a word appears within a word context
        words = text.split(' ')
        words_punc = self.get_words_and_punc(words)
        # Try and use '.' as a word

        for word in words_punc:
            if len(word) <= 1 and word not in NGramModel.PUNC_SET:
                continue
            lword = self.format_word(word)
            if self.check_valid_word(lword):
                self.shift_word_into_window(lword)
                self.update()

    def update(self):
        # Shifts word window in model, or adds it to model if not present
        word = self.word_window[0]
        if word in self.word_dict.keys():
            self.word_dict[word].add_words(self.word_window[1:])
        else:
            self.word_dict[word] = NGramLayer(layer=self.layers - 1)
            if len(self.word_window) > 1:
                self.word_dict[word].add_words(self.word_window[1:])
        self.total_words += 1

    def get_words_and_punc(self, words):
        words_with_punc = []
        for word in words:
            found_punc = False
            for punc in NGramModel.PUNC_SET:
                if punc in word:
                    word_witho_punc = word.strip(punc)
                    words_with_punc.append(word_witho_punc)
                    words_with_punc.append(punc)
                    found_punc = True
                    break
            if not found_punc:
                words_with_punc.append(word)
        return words_with_punc


    def shift_word_into_window(self, word):
        # Shifts a word into the word context
        if len(self.word_window) == self.layers:
            self.word_window.pop(0)
            self.word_window.append(word)
        else:
            self.word_window.append(word)

    def format_word(self, word):
        # Lower cases the word and removes punctuation from the edges
        lword = word.lower()
        lword = lword.strip(NGramModel.STRIP_SET)
        return lword

    def check_valid_word(self, word):
        # Checks that word only contains letters
        if word in NGramModel.PUNC_SET:
            return True
        if len(word) <= 1:
            return False
        for c in word:
            if c not in NGramModel.ALPHABET:
                return False
        return True

    def calculate_probs(self):
        # Iterates through each layer and calculates the marginal probability at each layer
        for word in self.word_dict.keys():
            self.word_probs[word] = self.word_dict[word].total_words / self.total_words
            self.word_dict[word].calculate_probs()

    def get_sorted_probs(self):
        # Uses a min-heap to create a sorted list of probabilities and words
        h = []
        for key in self.word_probs.keys():
            heappush(h, (self.word_probs[key], key))
            self.word_dict[key].get_sorted_probs()
        self.sorted_words = [heappop(h) for i in range(len(h))]
        return self.sorted_words

    def select_word(self, prev=[]):
        # Randomly selects a new word given a word context
        if len(prev) == 0:
            thresh = random.random()
            prob_sum = 0.0
            for prob, word in reversed(self.sorted_words):
                prob_sum += prob
                if thresh < prob_sum:
                    return word
        elif len(prev) > 0:
            word = prev[0]
            return self.word_dict[word].select_word(prev[1:])
        


# In[114]:


def parse_n_gram_model(model):
    txt_directory = '/Users/robmisasi/Documents/University/Classes/EE_460J/Homework/Lab5/txts'
    txts = [f for f in os.listdir(txt_directory)]
    for txt in txts:
        txt_path = os.path.join(txt_directory, txt)
        with open(txt_path, 'r') as f:
            for line in f:
                model.parse(line)
    model.calculate_probs()
    model.get_sorted_probs()
    return model


# In[115]:


model_layers = 4
model = NGramModel(layers=model_layers)
model = parse_n_gram_model(model)


# In[116]:


print(model.total_words)


# In[125]:


def create_n_gram_paragraph(model,context=[], size=30):
    prev = []
    prev.extend(context)
    paragraph = []
    paragraph.extend(context)
    for i in range(size):
        word = model.select_word(prev)
        if len(prev) < model.layers-1:
            prev.append(word)
        else:
            prev.pop(0)
            prev.append(word)
        paragraph.append(word)
    return ' '.join(paragraph)


# In[126]:


p = create_n_gram_paragraph(model, size=60)
print(p)


# In[127]:


p2 = create_n_gram_paragraph(model, context = ['the'], size=60)
print(p2)


# In[128]:


p3 = create_n_gram_paragraph(model, context = ['we'], size=60)
print(p3)


# In[ ]:




