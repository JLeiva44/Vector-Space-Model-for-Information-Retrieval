import glob
import math
from operator import invert 
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
        self.documents_vector = defaultdict(list)
        self.vocabulary = set()
        self.postings = defaultdict(dict)

        self.Stemmer = PorterStemmer()
        self.Stopwords = set(stopwords.words("english"))

        self.golbal_terms_frequency = defaultdict(int)
        self.__preprocesing_corpus()
        self.documents_norm = defaultdict(float) 
        self.__initialize_norms()

    def __initialize_norms(self):
        for id in self.documents_vector:
            norm=0
            for term in self.documents_vector[id]:
                norm += self.__calculate_weigth(term,self.documents_vector[id],None,id)**2
            norm = math.sqrt(norm)   
            self.documents_norm[id] = norm  
                
    
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
            self.documents_vector[index] = unique_terms
            self.vocabulary = self.vocabulary.union(unique_terms)
            self.documents_vector[index] = unique_terms


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
        query,query_postings = self.__preprocesing_query(query)
        scores = defaultdict(float)
        for id in range(1,len(self.documents)):
            scores[id] = self.similarity(query,query_postings,id)

        return scores

    def similarity(self, query,query_postings,document_id):    
        similarity = 0.0
        query_norm = 0

        for term in query:
            weigth_for_term_in_query = self.__calculate_weigth(term,query,query_postings,-1)
            query_norm += weigth_for_term_in_query**2
            weigth_for_term_in_document = self.__calculate_weigth(term,self.documents_vector[document_id],query_postings, document_id)
            similarity += (weigth_for_term_in_query * weigth_for_term_in_document)

        query_norm = math.sqrt(query_norm)
        similarity = similarity / (query_norm * self.documents_norm[document_id])   
        return similarity 

    def __calculate_weigth(self,term,document_vector,query_postings, index):
        if index == -1 :
            return (0.5 + (1-0.5) * self.__tf(term,document_vector,query_postings,index)) * self.__idf(term)
        return self.__tf(term,document_vector,query_postings,index) * self.__idf(term)

    def __tf(self,term, document_vector,query_postings,index):
        max_freq = 0
        for term in document_vector:
            if index == -1:
                freq = query_postings[term]
            else:
                freq = self.postings[term][index]
            if max_freq < freq:
                 max_freq = freq
        if index == -1:
            term_freq = query_postings[term]
        else:    
            term_freq = self.postings[term][index]
        return term_freq/max_freq

    def __idf(self,term):
        return math.log((len(self.documents_vector) +1) /(self.golbal_terms_frequency[term] + 1),2)            



    def __preprocesing_query(self, query):
        query = self.__remove_special_characters(query)
        query = self.__remove_digits(query)    

         #tokens
        query = word_tokenize(query)

            #removing stopwords
        query = [word.lower() for word in query if word not in self.Stopwords]

            #Stemming
        query = [self.Stemmer.stem(word) for word in query]
        query_postings = defaultdict(int)
        for term in query:
            query_postings[term] = query.count(term)

        query = list(set(query))

        return query,query_postings



# model = Vector_Space_Model("./corpus/*")
# f = model.golbal_terms_frequency
# print(model.query("couple"))


        



