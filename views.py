from django.shortcuts import render
import json
# Create your views here.
from django.http import  HttpResponse , JsonResponse,HttpResponseBadRequest


def say_hello(request):
    return render(request ,'index.html',{'name':'halim zaaim'})

def serve_ajax(request):
    data = json.load(request)
    sentence = data.get('sentence')
    disamb = main(sentence)
    return JsonResponse({"entities": disamb})
      


######################## business Logic ########################
import nltk
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from itertools import chain
from nltk.corpus import wordnet
from nltk import FreqDist
import string
import re
import math
import numpy as np

from rdflib import Graph ,URIRef
from SPARQLWrapper import  SPARQLWrapper ,JSON, N3
from pprint import pprint
from collections import Counter
from string import digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  


####################################################
class Similarity:
    def __init__(self):
        self.translation_table = str.maketrans(string.punctuation+string.ascii_uppercase,
                                     " "*len(string.punctuation)+string.ascii_lowercase)
        self.stemmer =PorterStemmer()
        self.tfidf_vectorizer = TfidfVectorizer(max_df=20,ngram_range=(1,20),smooth_idf=False)
        self.lemmatizer = WordNetLemmatizer()

    def get_words_from_line_list(self,text): 
      
        text = self.remove_stop_words(text)
        word_list = text.split()
      
        return word_list

    def remove_stop_words(self,txt):
        encoded_text = txt.encode("ascii", "ignore")
        text = encoded_text.decode()
        text = txt.translate(self.translation_table)## more cleaning 
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\x00-\x7f]',r'', text) 
        stop_words = list(get_stop_words('en'))         #About 900 stopwords
        nltk_words = list(stopwords.words('english')) #About 150 stopwords
        stop_words.extend(nltk_words)
        output = [w for w in text.split(" ") if not w in stop_words and not len(w)<=3 ]
        output = [self.lemmatizer.lemmatize(token) for token in output]
        output = [self.stemmer.stem(token) for token in output]
        sentence =  ' '.join(output)
        return sentence
  
    
    def count_frequency(self,word_list): 
      
        D = {}
      
        for new_word in word_list:
          
            if new_word in D:
                D[new_word] = D[new_word] + 1
              
            else:
                D[new_word] = 1
              
        return D
  
    def word_frequencies_for_txt(self,text): 
      
        line_list = text
        word_list = self.get_words_from_line_list(line_list)
        freq_mapping = self.count_frequency(word_list)
        return freq_mapping
  
  
    # returns the dot product of two documents
    def two_docs_dot_product(self,D1, D2): 
        Sum = 0.0
      
        for key in D1:
          
            if key in D2:
                Sum += (D1[key] * D2[key])
              
        return Sum
  
    
    def vector_angle_between_2docs(self,D1, D2): 
        numerator = self.two_docs_dot_product(D1, D2)
        denominator = math.sqrt(self.two_docs_dot_product(D1, D1)*self.two_docs_dot_product(D2, D2))
      
        return math.acos(numerator / denominator)
    
    def doc_similarity_to_doc(self,doc1, doc2):
      
        sorted_word_list_1 = self.word_frequencies_for_txt(doc1)
        sorted_word_list_2 = self.word_frequencies_for_txt(doc2)
        distance = self.vector_angle_between_2docs(sorted_word_list_1, sorted_word_list_2)
        return distance
      
    def tfIdf_cosin_similarity(self,doc ,train_set,arr):
        ## train_set is list that conatain [sentence ,abstarct1 ,abstract1 ...]
       
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(train_set)
        cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
        ## the similarity is the biggest when it is 1
        del tfidf_matrix 
        cosine = cosine.reshape(-1)
    
        cosine = (0.8 * cosine) + (arr * 0.2)
        temp = cosine.copy()
        cosine[::-1].sort()
        
        dis = cosine[1]
        index = temp.tolist().index(dis)
        index = index - 1
        return (dis,index)
        
####################################################

class Preprocess:
    def __init__(self,sentence=None,IsLocal=True):
        if IsLocal:
            self.original_sentence = sentence
            self.sentence = sentence
        
        
    def process(self,sentence=None,IsLocal=True):
        if IsLocal:
            self.__clean_sentence(self.sentence)
            self.__remove_stop_words()
            tuple_word_tag = self.apply_part_of_speech_tagging(self.sentence)
            entities = self.get_only_entities(tuple_word_tag)
            print("entities in processs :" ,entities)
            return entities
        else:
            self.__clean_sentence(sentence)
            tuple_word_tag = self.apply_part_of_speech_tagging(sentence)
            entities = self.get_only_entities(tuple_word_tag)
            print("entities in processs :" ,entities)
            return entities

    def __remove_stop_words(self):
        stop_words = list(get_stop_words('en'))         #About 900 stopwords
        nltk_words = list(stopwords.words('english')) #About 150 stopwords
        stop_words.extend(nltk_words)
        output = [w for w in self.sentence.split() if not w in stop_words]
        self.sentence = ' '.join(output)
        
    def __clean_sentence(self,sentence,IsLocal=True):
        #sentence = sentence.lower()
        #sentence = re.sub('\[.*?\]', '', sentence)
        sentence = re.sub('https?://\S+|www\.\S+', '', sentence)
        #sentence = re.sub('<.*?>+', '', sentence)
        sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
        sentence = re.sub('\n', '', sentence)
        #sentence = re.sub('\w*\d\w*', '', sentence)
        #sentence = re.sub(r'[-\(\)\"#\/@;:<>\{\}\-=~|\.\?]', '',  self.sentence)
        if IsLocal:
            self.sentence = sentence
        return sentence
    def apply_part_of_speech_tagging(self,sentence):
        return nltk.pos_tag(sentence.split())

    
    def get_only_entities(self, tuple_list,IsLocal=True):
        entities = []
        #print(tuple_list)
        for i in range(len(tuple_list)):
            
            if self.__isEntity(tag=tuple_list[i][1]):
                enn = tuple_list[i][0]
                if not enn[0].isupper():
                    enn = enn.title()

                entities.append(enn)

        """add if NN preceeded by NNS or NNP .. or the inverse """
        if IsLocal:
            original_sentence_tags = self.__clean_sentence(self.original_sentence)
            original_sentence_tags = self.apply_part_of_speech_tagging(original_sentence_tags)
        
            for j in range(len(original_sentence_tags)-1):
                if self.__isEntity(tag=original_sentence_tags[j][1]) and self.__isEntity(tag=original_sentence_tags[j+1][1]):
                    enn = original_sentence_tags[j][0]
                    if not enn[0].isupper():
                        enn = enn.title()
                    entities.append(enn +"_"+original_sentence_tags[j+1][0].title())

        return entities


    def __isEntity(self ,tag=""):
        if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
            return True
        else:
            return False

    def get_sentence(self):
        return self.sentence
    
    
    def get_original_sentence(self):
        return self.original_sentence
        

############################# Disambiguation  #########################################

class EntityDiambiguation:
    def __init__(self,entities,sentence):
        self.original_sentence = sentence
        self.entities = entities
        self.__sparql__= SPARQLWrapper("https://dbpedia.org/sparql")
        self.similarily = Similarity()
        
        #self.do_more_entity_filtring()
    
    def get_entities(self):
        return self.entities
   
    def do_more_entity_filtring(self):
         
        for entity in self.entities:
            if "_" in entity:
                if self.check(entity):
                    self.entities.remove(entity)

    def check(self,complex_word):
        self.__sparql__.setQuery('''
        PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>
        SELECT Distinct COUNT(?y) 
        WHERE { <http://dbpedia.org/resource/'''+complex_word+'''>
        dbo:wikiPageWikiLink ?y . 
        }
        ''')
        self.__sparql__.setReturnFormat(JSON)
        qres = self.__sparql__.query().convert()
            
        if qres['results']['bindings'][0]["y"]["value"] == 0:
            return True
        else:
            return False
    def filter_links(self,list_of_link, entities):
        """ the idea is to keep only links that contains words 
        fron the given sentence
        returns a tuple of (link , resourceName)"""
        disired_links = []
        for link in list_of_link:
            l = link[0]
            resource_name = l.replace("http://dbpedia.org/resource/", "")
            for entity in entities:
                if entity.contains(resource_name) or resource_name.contains(entity):
                    disired_links.append((l,resource_name))
        
        """ guess it's always to add the first element desipte all the odds"""
        first_Link = list_of_link[0][0].replace("http://dbpedia.org/resource/", "")
        disired_links.append(list_of_link[0][0],first_Link)
        return disired_links

    def get_entity_abstract_by_entity(self,entity):
        self.__sparql__.setQuery('''
        prefix dbpedia: <http://dbpedia.org/resource/>
        prefix dbpedia-owl: <http://dbpedia.org/ontology/>
        select ?abstract  where { 
        dbpedia:'''+entity+''' dbpedia-owl:abstract ?abstract  .                  
        filter(langMatches(lang(?abstract),"en"))
        }

        ''')
        self.__sparql__.setReturnFormat(JSON)
        qres = self.__sparql__.query().convert()
        if len(qres['results']['bindings'])!=0:
            return qres['results']['bindings'][0]["abstract"]["value"]
        else:
            return False
    
    
    def get_entity_abstract_by_link(self,link):
        link = link.replace("http","https")
         
        self.__sparql__.setQuery('''
        prefix dbpedia: <http://dbpedia.org/resource/>
        prefix dbpedia-owl: <http://dbpedia.org/ontology/>
        select distinct ?abstract  where { 
        '''+link+''' 
        dbo:abstract ?abstract .
        filter(langMatches(lang(?abstract),"en"))
        }
       
        ''')
        self.__sparql__.setReturnFormat(JSON)
        qres = self.__sparql__.query().convert()
        return qres['results']['bindings'][0]["abstract"]["value"]
        
    
    ######################################### ALGO 0 ####################################
    def check_if_enenty_isAmbiguious(self,entity):
        self.__sparql__.setQuery('''
        SELECT DISTINCT COUNT(?y) WHERE {
        dbr:'''+entity+''' dbo:wikiPageDisambiguates ?y }
        ''')
        self.__sparql__.setReturnFormat(JSON)
        qres = self.__sparql__.query().convert()
        if int(qres['results']['bindings'][0]["callret-0"]["value"]) == 0:
            return True
        else:
            return False
    
    def get_lenght_of_entity_disamb(self,entity):

        self.__sparql__.setQuery('''
        SELECT DISTINCT COUNT(?y) WHERE {
        dbr:'''+entity+'''_\(disambiguation\) dbo:wikiPageDisambiguates ?y }
        ''')
        self.__sparql__.setReturnFormat(JSON)
        qres = self.__sparql__.query().convert()

        if int(qres['results']['bindings'][0]["callret-0"]["value"]) > 0:
            return True
        else:
            return False
    
    def entity_vs_entities_disamb_byType(self, entity):
        self.__sparql__.setQuery('''
        SELECT distinct ?y ?b WHERE {
        dbr:'''+entity+'''_\(disambiguation\) dbo:wikiPageDisambiguates ?y.
        ?y rdf:type ?b .}

        ''')
        self.__sparql__.setReturnFormat(JSON)
        qres = self.__sparql__.query().convert()

        queryRes = []
         
        for result in qres['results']['bindings']:

            yy = result['y']["value"].replace("http://dbpedia.org/resource/", "")
            bb = result['b']["value"]
            lastIndex = bb.rindex("/")
            b = bb[lastIndex+1:len(bb)]
            b = ''.join(i for i in b if i.isalpha())
            if b == "owlThing" or len(b) < 2:
                continue

            for ent in self.entities:
                if ent in yy:
                    queryRes.append((yy,b))
               
        """count and filter"""
        final_list_count = []
        for tup in queryRes:
            final_list_count.append(tup[1])
        
        return (entity , list(set(final_list_count[0:50])))
    
    def disambiguate_alg0(self):
        """ here the immplementation of the first step in the algo 0"""
        ANE = []
        DNE = []
        for en in self.entities:
            
            if self.check_if_enenty_isAmbiguious(en):
                
                c = self.get_lenght_of_entity_disamb(en)
                if c:
                    ANE.append(en)
            
            else:
                #ANE.append(en)
                pass
        print(ANE)
        return ANE
##        for ne in ANE:
##            dd = self.entity_vs_entities_disamb_byType(ne)
##            if len(dd) != 0:
##                DNE.append(dd)
##
##        """ return a list of entites each one is (oracle , [(Corporation ,7),(Database,6)])"""
##        return DNE
    ######################################################################

    
                
            
############################## Algo 1 #######################################
    def get_list_of_conditates(self , entity):
        self.__sparql__.setQuery('''
        SELECT distinct ?y WHERE {
        dbr:'''+entity+'''_\(disambiguation\) dbo:wikiPageDisambiguates ?y.
        }
        ''')
        self.__sparql__.setReturnFormat(JSON)
        qres = self.__sparql__.query().convert()

        queryRes = []
        for result in qres['results']['bindings']:
            yy = result['y']["value"].replace("http://dbpedia.org/resource/", "")
            if ","in yy or "'" in yy or "!" in yy or "?" in yy or "&" in yy or "+" in yy or "*" in yy:
                continue
            if "(" in yy:
                yy=yy.replace("(", "\(")
                yy=yy.replace(")", "\)")
            if "/" in yy:
                yy=yy.replace("/", "\/")
            if "." in yy:
                yy=yy.replace(".", "\.")
            if "-" in yy:
                yy=yy.replace("-", "\-")
                
            queryRes.append(yy)
        return queryRes

    def get_list_of_conditates_wiki(self , entity):
        self.__sparql__.setQuery('''
        SELECT distinct ?y WHERE {
        dbr:'''+entity+''' dbo:wikiPageWikiLink ?y.
        }
        ''')
        self.__sparql__.setReturnFormat(JSON)
        qres = self.__sparql__.query().convert()

        queryRes = []
        for result in qres['results']['bindings']:
            yy = result['y']["value"].replace("http://dbpedia.org/resource/", "")
            if ","in yy or "'" in yy or "!" in yy or "?" in yy or "&" in yy or "+" in yy or "*" in yy or "@" in yy:
                continue
            if "(" in yy:
                yy=yy.replace("(", "\(")
                yy=yy.replace(")", "\)")
            if "/" in yy:
                yy=yy.replace("/", "\/")
            if "." in yy:
                yy=yy.replace(".", "\.")
            if "-" in yy:
                yy=yy.replace("-", "\-")
                
            queryRes.append(yy)
        return queryRes
        
################################### ifidf_onto aproche : Algo 2 #####################
    def get_all_entity_cond_abstracts(self, entity):
        list_condidates = self.get_list_of_conditates(entity)
        
        #wiki_conditats = self.get_list_of_conditates_wiki(entity)
        print("started with :",entity)
        #list_condidates.extend(wiki_conditats)
        list_condidates = list(set(list_condidates))
        List_abstracts = []
        list_no_found = []
        ## get all abstracts
        for i in range(len(list_condidates)):
            abstract = self.get_entity_abstract_by_entity(list_condidates[i])
            if abstract == False:
                list_no_found.append(list_condidates[i])
            else:
                abstractt = self.similarily.remove_stop_words(abstract)
                List_abstracts.append(abstractt)
                
        ## remove the ones without abstract
        for ll in list_no_found:
            list_condidates.remove(ll)
        print("done with : ",entity)
        return (list_condidates,List_abstracts)

    def frequency_dist(self ,entity , list_abstract):
        list_freq = np.empty(shape=(len( list_abstract),),dtype=float)
        for i in range(len(list_abstract)):
            text_list = list_abstract[i].lower().split(" ")
            freqDist = FreqDist(text_list)
            #words = list(freqDist.keys())
            freq = freqDist[entity.lower()]
            if float(freq) == 0.0:
                freq = 0.5
            list_freq[i] = float(freq)
        return list_freq
        

    def linking(self,entity,list_onto):
        list_condidates,List_abstracts = self.get_all_entity_cond_abstracts(entity)
        if self.original_sentence.split().count(entity) > 1:
            self.original_sentence.replace(entity,"")
        
        lesk_dfini = lesk(self.original_sentence, entity.lower()).definition()
        p = Preprocess(IsLocal=False)
        lesk_dfini_pos = p.apply_part_of_speech_tagging(lesk_dfini)
        lesk_dfini_pos = p.get_only_entities(lesk_dfini_pos,IsLocal=False)
        
        sentence = self.original_sentence +" " +' '.join(lesk_dfini_pos)
        sentence = self.similarily.remove_stop_words(sentence)
        List_abstracts.insert(0,sentence)
        arr = self.frequency_dist(entity,List_abstracts)
        simm , indext = self.similarily.tfIdf_cosin_similarity(sentence ,List_abstracts,arr)
        del List_abstracts[0]
        dist = 8 ;
        index = 0;
        for i in range(len(List_abstracts)):
            d = self.similarily.doc_similarity_to_doc(sentence,List_abstracts[i])
            if d < dist:
                index = i
                dist = d
        if 2 - dist > simm:
            index = index
        else:
            index = indext
            
        entity_disamd = list_condidates[index]
        del list_condidates
        del List_abstracts
        return(entity_disamd, d)
        
    def disambiguate1(self,DNE):############# algo 1 main Method
        tuples = []
        result = []
        for d in DNE:
            entity = d
            res = self.linking(entity,d)
            tuples.append((entity,res))
        print("FINALE TUPLE : ",tuples)  
        return tuples

    def break_onto_to_words(self,onto):
        index = 0 ;
        boolean = False
        for element in range(0, len(onto)):
            if onto[element].isupper() and element != 0:
                index = element
                boolean = True
        if boolean:
             onto =  onto[:index] + ' ' + onto[index:]
        return onto

        
################################## TESTS##############
def main(sentence):
        p = Preprocess(sentence=sentence)
        entities = p.process()
        algo = EntityDiambiguation(entities,p.get_original_sentence())
        DNE = algo.disambiguate_alg0()
        result = ""
        
        result = algo.disambiguate1(DNE)
        dismab = {}
        for en in result:
            entity = en[0]
            
            entity = entity.replace("\\","")
            print(entity)
            
            dismab[entity] = en[1][0]
            values = [*dismab]
        """for e in entities:
            if e not in values:
                dismab[e]= e"""

        print (dismab)
        return dismab


