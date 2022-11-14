import glob
import math 
import os
import re
import sys
from collections import defaultdict
from functools import reduce
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class Vector_Space_Model(object):
    def __init__(self, corpus) :
        self.corpus = corpus

        self.documents = dict()
        self.vocabulary = set()
        self.postings = defaultdict(dict)

        self.Stemmer = PorterStemmer()
        self.Stopwords = set(stopwords.words("english"))

        self.golbal_terms_frequency = defaultdict(int)
        self.documents_norm = defaultdict(float) 
        self.__preprocesing_corpus()

    
    def __preprocesing_corpus(self):
        index =1
        for filename in glob.glob(self.corpus):
            #read the document
            with open(filename,"r") as file:
                document_text = file.read()
            document_text = self.__remove_special_characters(document_text)
            document_text = self.__remove_digits(document_text)

            #tokens
            document_text = word_tokenize(document_text)

            #removing stopwords
            document_text = [word.lower() for word in document_text if word not in self.Stopwords]

            #Stemming
            document_text = [self.Stemmer.stem(word) for word in document_text]

            unique_terms = set(document_text)
            self.vocabulary = self.vocabulary.union(unique_terms)


            for term in unique_terms:
                self.postings[term][index] = document_text.count(term)
                self.golbal_terms_frequency[term] += 1

            self.documents[index] = os.path.basename(filename)
            index = index +1
        return     

    def __remove_special_characters(self,text):
        regex = re.compile(r"[^a-zA-Z0-9\s]") 
        return re.sub(regex,"",text)

    def __remove_digits(self,text):
        # Regex pattern for a word
        regex = re.compile(r"\d")

        # Replace and return
        return re.sub(regex, "", text)

    def query(self,query):
        query = self.__preprocesing_query(query)
        return query


    def __preprocesing_query(self, query):
        query = self.__remove_special_characters(query)
        query = self.__remove_digits(query)    

         #tokens
        query = word_tokenize(query)

            #removing stopwords
        query = [word.lower() for word in query if word not in self.Stopwords]

            #Stemming
        query = [self.Stemmer.stem(word) for word in query]

        # falta ver lo del vector q

        return query





        



