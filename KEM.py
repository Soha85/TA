from flask import Flask,request,jsonify
from flask_restful import Api, Resource, reqparse
import sys
from nltk.stem import PorterStemmer
from itertools import groupby
import spacy
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
api = Api(app)

class KEM(Resource):
    

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument("std_txt")
        parser.add_argument("gld_txt")
        params = parser.parse_args()
        self.__nlp = spacy.load("en_core_web_lg")
        #load ideal text
        self.__Golden_txt=request.args.get("gld_txt")
        self.__gld_doc=self.__nlp(self.__Golden_txt)
        
        #apply nlp spacy on student text
        self.__std_doc=self.__nlp(request.args.get("std_txt"))
        return {'tag':self.__std_GetTagInfo(),
                'OOV':self.__std_GetOOVInfo(),
                'wrd-punc':self.__std_GetWrdInfo(),
                'stp':self.__std_GetStopInfo(),
                'sent':self.__std_GetSentInfo(),
                'longsent':self.__std_GetLongSentInfo(),
                'freq':self.__frequent_matched_unmatched(),
                'unique':self.__uique_matched_unmatched(),
                'pos':self.__special_pos_matched_unmatched(),
                'yule':self.__Yule1Measure()}

    def __std_GetTagInfo(self):
        return dict(Counter(([token.pos_ for token in self.__std_doc])))
    
    def __std_GetOOVInfo(self):
        return dict(Counter(([token.is_oov for token in self.__std_doc])))

    def __std_GetWrdInfo(self):
        return dict(Counter(([token.is_punct for token in self.__std_doc])))
    
    def __std_GetStopInfo(self):
        return dict(Counter(([token.is_stop for token in self.__std_doc])))

    def __std_GetSentInfo(self):
        return {'Sentences number':len(list(self.__std_doc.sents))}

    def __std_GetLongSentInfo(self):
        i=0
        lst=[]
        for sent in self.__std_doc.sents:
            i=0
            for w in self.__nlp(sent.text):
                if(not(w.is_punct)):
                    i=i+1
            lst.append(i)
        return {'Long Sentences number':sum(1 for obj in lst if obj>9)}
  

    def __frequent_matched_unmatched(self):
        gld_frq=self.__get_frequent_words(self.__gld_doc)
        std_frq=self.__get_frequent_words(self.__std_doc)

        #exact match
        match_result=self.__get_matched_unmatched_lists(std_frq,gld_frq)
        #similar match using threshold greater than 0.6
        similar_result=self.__most_similar(match_result.get("Unmatched"),gld_frq)
        
        return {'Match Results':match_result,'Similar_Results':similar_result}

    def __uique_matched_unmatched(self):
        unq_gld=self.__get_doc_unique(self.__gld_doc)
        unq_std=self.__get_doc_unique(self.__std_doc)

        #exact match
        match_result=self.__get_matched_unmatched_lists(unq_std,unq_gld)
        #similar match using threshold greater than 0.6
        similar_result=self.__most_similar(match_result.get("Unmatched"),unq_gld)
        
        return {'Match Results':match_result,'Similar_Results':similar_result}
    
    def __special_pos_matched_unmatched(self):
        pos_gld=self.__extract_POS(self.__gld_doc)
        pos_std=self.__extract_POS(self.__std_doc)

        #exact match
        match_result=self.__get_matched_unmatched_lists(pos_std,pos_gld)
        #similar match using threshold greater than 0.6
        similar_result=self.__most_similar(match_result.get("Unmatched"),pos_gld)

        return {'Match Results':match_result,'Similar_Results':similar_result}

    def __Yule1Measure(self):
        yule_gld=self.__yule(self.__gld_doc)
        yule_std=self.__yule(self.__std_doc)
        return {'std-Yule':yule_std,'gld_std':yule_gld}

    #get most similar tokens between unmatched list and the whole golden set
    def __most_similar(self,unmatched_lst,gld_lst):
        sim_lst=[]
        for tkn1 in unmatched_lst:
            for tkn2 in gld_lst:
                if (self.__std_doc.vocab[tkn1].similarity(self.__gld_doc.vocab[tkn2]) > 0.6 and (tkn1 is not tkn2) ):
                    sim_lst.append((tkn1,tkn2,str(self.__std_doc.vocab[tkn1].similarity(self.__gld_doc.vocab[tkn2]))))
        #df= pd.DataFrame(lst,columns=['tkn1','tkn2','similarity'])
        #df=df[df['similarity']>0.7]
        return sim_lst
    

    
    #define the match function between the two lists
    def __matching_keywords(self,stdlst,gldlst):
            #matched list
            res=[]
            #unmatched list
            tmpres=[]
            for x in stdlst:
                if (x in gldlst):
                    res.append(x)
                else:
                    tmpres.append(x)
            return res,tmpres

    #define a function to get the frequent words in a text which aren't stop words or punctuations
    def __get_frequent_words(self,sample_doc):
        words = [token.text.lower() for token in sample_doc if not token.is_stop and not token.is_punct]
        word_freq = Counter(words)
        return word_freq

    def __get_doc_unique(self,sample_doc):
        #extract unique words with only 1 occurrence
        wrd_frq=self.__get_frequent_words(sample_doc)
        uniq_wrds=[word for word in wrd_frq if (wrd_frq.get(word)==1)]
        return uniq_wrds

    def __get_matched_unmatched_lists(self,gld_lst,std_lst):
        #apply match function
        matched,unmatched=self.__matching_keywords(std_lst,gld_lst)
        return {'Matched':matched,'Count-M':len(matched),'Unmatched':unmatched,'Count-U':len(unmatched)}

    #third method extract nouns chuncks and special tags like proper noun, noun, adjective
    def __extract_POS(self,sample_doc):
        res=[]
        for chk in sample_doc.noun_chunks:
            tmp=""
            for tkn in chk:
                if (tkn.pos_ in ['NOUN','PROPN','ADJ'] ):
                    if (not(tkn.is_stop) and not(tkn.is_punct)):
                        tmp = tmp + tkn.text.lower() + " "
            if(tmp.strip()!=""):
                res.append(tmp.strip())
        return list(dict.fromkeys(res))

    
    def __yule(self,sample_doc):
        # yule's I measure (the inverse of yule's K measure)
        # higher number is higher diversity - richer vocabulary
        d = {}
        ps = PorterStemmer()
        for w in sample_doc:
            if(not(w.is_punct)):
                try:
                    w = ps.stem(w.text.lower())
                except:
                    print(w)
                try:
                    d[w] = d[w] + 1
                except KeyError:
                    d[w] = 1
     
            M1 = float(len(d))
            M2 = float(sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(d.values()))]))
     
        try:
            return float((M1*M1)/(M2-M1))
        except ZeroDivisionError:
            return float(0)

api.add_resource(KEM, "/ta", "/ta/")
     
if __name__=="__main__":
    
    app.run(debug=True)
