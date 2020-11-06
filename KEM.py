import language_check
#import language_tool_python
from nltk.stem import PorterStemmer
from itertools import groupby
import spacy
from collections import Counter
import json
import re
from nltk.corpus import stopwords
from textblob import TextBlob
from lexicalrichness import LexicalRichness
from lexical_diversity import lex_div as ld
import string
nlp=spacy.load("en_core_web_md")
stopwrds=set(stopwords.words('english'))
#LT_tool=language_tool_python.LanguageTool('en-us')
gr_tool=language_check.LanguageTool('en-us')
class KEM():
    def __init__(self,std_txt=None,gld_lst=None):
        
        if (std_txt!=None):
             #apply nlp spacy on student text
             self.__std_txt=std_txt
             self.__std_txt=self.__std_txt.replace("\n"," ")
             self.__std_doc=nlp(self.__std_txt, disable=['textcat','merge_entities','merge_subtokens','merge_noun_chunks'])
        if(gld_lst!=None):
            self.__gld_lst=gld_lst
            
    def ExtractVA(self):
        try:
            std_key=self.__get__keywords(self.__std_doc)
            return {'documents similarities':self.__doc_cosine_similarity(self.__std_doc,self.__gld_lst),
                    'Keywords list':self.__Find_match_and_similarity(std_key.get('keywords'),self.__gld_lst),
                    'yule':self.__yule(self.__std_doc),
                    'Other Yules':self.__get_yules(self.__std_doc),
                    'lexical diversity':self.__lexical_diversity(self.__std_doc)}
        except Exception as e:
            return {'error':str(e)}
        
        
    def Extractonly(self):
        try:
            key_dic=self.__get__keywords(self.__std_doc)
            key_dic["keywords"]=str(key_dic.get("keywords")).replace("', '","','")
            return json.dumps({'All Caps':self.__std_GetALLCAPS(self.__std_doc),
                    'tag':self.__std_GetTagInfo(self.__std_doc),
                    'OOV':self.__std_GetOOVInfo(self.__std_doc),
                    'tenses':self.__std_GetVerbTenses(self.__std_doc),
                    #'Verb dependencies':self.__std_GetDependencies(self.__std_doc),
                    'Active':self.__std_GetActiveVoices(self.__std_doc),
                    'Passive':self.__std_GetPassiveVoices(self.__std_doc),
                    'words':self.__std_GetWrdInfo(self.__std_doc),
                    'stop':self.__std_GetStopInfo(self.__std_doc),
                    'Punct':self.__std_GetPunctInfo(self.__std_doc),
                    'sent':self.__std_GetSentInfo(self.__std_doc),
                    'longsent':self.__std_GetLongSentInfo(self.__std_doc),
                    'sent info':self.__std_GetwrdsSentInfo(self.__std_doc),
                    'sentences similarity':self.__std_coisne_similarity(self.__std_doc),
                    'grammar errors':self.__grammar_check(self.__std_doc),
 #                   'grammar errors LT':self.__grammar_check_LT(self.__std_doc),
                    'Keywords list':key_dic,
                    'yule':self.__yule(self.__std_doc),
                    'Other Yules':self.__get_yules(self.__std_doc),
                    'lexical diversity':self.__lexical_diversity(self.__std_doc)})
        except Exception as e:
            return json.dumps({'error':str(e)})
    
    def Extract_and_match(self):
        try:
            std_key=self.__get__keywords(self.__std_doc)
            return json.dumps({'All Caps':self.__std_GetALLCAPS(self.__std_doc),
                    'tag':self.__std_GetTagInfo(self.__std_doc),
                    'OOV':self.__std_GetOOVInfo(self.__std_doc),
                    'tenses':self.__std_GetVerbTenses(self.__std_doc),
                    #'Verb dependencies':self.__std_GetDependencies(self.__std_doc),
                    'Active':self.__std_GetActiveVoices(self.__std_doc),
                    'Passive':self.__std_GetPassiveVoices(self.__std_doc),
                    'words':self.__std_GetWrdInfo(self.__std_doc),
                    'stop':self.__std_GetStopInfo(self.__std_doc),
                    'Punct':self.__std_GetPunctInfo(self.__std_doc),
                    'sent':self.__std_GetSentInfo(self.__std_doc),
                    'longsent':self.__std_GetLongSentInfo(self.__std_doc),
                    'sent info':self.__std_GetwrdsSentInfo(self.__std_doc),
                    'sentences similarity':self.__std_coisne_similarity(self.__std_doc),
                    'documents similarities':self.__doc_cosine_similarity(self.__std_doc,self.__gld_lst),
                    'grammar errors':self.__grammar_check(self.__std_doc),
#                    'grammar errors LT':self.__grammar_check_LT(self.__std_doc),
                    'Keywords list':self.__Find_match_and_similarity(std_key.get('keywords'),self.__gld_lst),
                    'yule':self.__yule(self.__std_doc),
                    'Other Yules':self.__get_yules(self.__std_doc),
                    'lexical diversity':self.__lexical_diversity(self.__std_doc)})
        except Exception as e:
            return json.dumps({'error':str(e)})
        
    def __grammar_check(self,sample_doc):
        res=[]
        
        for sent in sample_doc.sents:
            matches=gr_tool.check(sent.text)
            res.append(len(matches))
        return res
    def __grammar_check_LT(self,sample_doc):
        res=[]
        
        for sent in sample_doc.sents:
            matches=LT_tool.check(sent.text)
            res.append(len(matches))
        return res

    
    def __lexical_diversity(self,sample_doc):
        lst=[token.text for token in sample_doc]
        text=' '.join(lst)
        flt=ld.flemmatize(text)
        return {"TTR":ld.ttr(flt),"RTTR":ld.root_ttr(flt),"LTTR":ld.log_ttr(flt),"massTTR":ld.maas_ttr(flt),"MSTTR":ld.msttr(flt),
                "MATTR":ld.mattr(flt),"HDD":ld.hdd(flt),"MTLD":ld.mtld(flt),"MTLD_MW":ld.mtld_ma_wrap(flt),"MTLD_MA":ld.mtld_ma_bid(flt)}
    
    def __lexical_richness(self,sample_doc):
        lst=[token.text for token in sample_doc]
        lex = LexicalRichness(' '.join(lst),use_TextBlob=True)
        hdd=0
        if(len(lst)<=10 and len(lst)>=0):
            hdd=lex.hdd(draws=2)
        else:
            hdd=lex.hdd(draws=10)
        return {"TTR":lex.ttr,"RTTR":lex.rttr,"CTTR":lex.cttr,"MSTTR":lex.msttr(segment_window=5),"MATTR":lex.mattr(window_size=5)
                ,"MTLD":lex.mtld(threshold=0.72),"HDD":hdd}
    
    def __doc_cosine_similarity(self,sampledoc,gld_lst):
        res=[]
        for gld in gld_lst:
            gld_txt=' '.join(gld)
            res.append(sampledoc.similarity(nlp(gld_txt, disable=['textcat','merge_entities','merge_subtokens','merge_noun_chunks'])))
        return res
    
    def __std_coisne_similarity(self,sampledoc):
        prevsent=""
        results=[]
        
        for sent in sampledoc.sents:
            txt=' '.join([re.sub(r'['+string.punctuation+']+', '', w.lower()) for w in sent.text.split(' ') if (re.sub(r'['+string.punctuation+']+', '', w.lower()) not in  stopwrds) and (w not in string.punctuation)])
            txt=nlp(txt, disable=['textcat','merge_entities','merge_subtokens','merge_noun_chunks'])
            if prevsent!="":
                val=prevsent.similarity(txt)
                results.append(val)
            prevsent=txt
        return results
                
    def __std_GetALLCAPS(self,sample_doc):
        
        lst=[line.text for line in sample_doc if line.text.strip().isupper() and line.text.strip()]
        return list(dict.fromkeys(lst))
        

    def __std_GetPassiveVoices(self,sample_doc):
        pass_tkn=[tok.text for tok in sample_doc if (tok.dep_ == "nsubjpass")] 
        return len(pass_tkn)
    
    def __std_GetActiveVoices(self,sample_doc):
        Act_tkn=[tok.text for tok in sample_doc if (tok.dep_ == "nsubj")]
        return len(Act_tkn)

    def __std_GetVerbTenses(self,sample_doc):
        tenses=dict({})
        #tenses['present']=len([nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense") for tkn in sample_doc if tkn.pos_=="VERB" if (nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense") and nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense") =="pres")])
        tenses['present']=len([tkn for tkn in sample_doc if tkn.tag_ in ["VBP","VBZ","VBG"]])
        #tenses['past']=len([nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense") for tkn in sample_doc if tkn.pos_=="VERB" if (nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense") and nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense") =="pas")])
        tenses['past']=len([tkn for tkn in sample_doc if tkn.tag_ in ["VBD","VBN"]])
        #tenses['future']=len([tkn.text for tkn in sample_doc if tkn.text.lower() in ("shall","will") and tkn.dep_.lower()=="aux"])
        tenses['future']=len([tkn for tkn in sample_doc if tkn.tag_ in ["MD"]])
        #return [(tkn.lemma_.lower(),nlp.vocab.morphology.tag_map[tkn.tag_]) for tkn in sample_doc if tkn.pos_=="VERB"]
        return tenses
    def __std_GetDependencies(self,sample_doc):
        dep=[]
        for tkn in sample_doc:
            if (tkn.pos_ =="VERB"):
                dep.append([tkn.text,tkn.pos_,tkn.dep_,nlp.vocab.morphology.tag_map[tkn.tag_]])
        return dep
    
    def __std_GetTagInfo(self,sample_doc):
        return dict(Counter(([token.pos_ for token in sample_doc])))
    
    def __std_GetOOVInfo(self,sample_doc):
        return len([token.is_oov for token in sample_doc if (token.is_oov)])

    def __std_GetWrdInfo(self,sample_doc):
        return len([token.is_punct for token in sample_doc if not(token.is_stop) and not(token.is_punct)])
    
    def __std_GetStopInfo(self,sample_doc):
        return len([token.is_stop for token in sample_doc if (token.lemma_.lower() in stopwrds)])
    def __std_GetPunctInfo(self,sample_doc):
        return len([token.is_punct for token in sample_doc if (token.is_punct)])

    def __std_GetSentInfo(self,sample_doc):
        return len(list(sample_doc.sents))

    def __std_GetLongSentInfo(self,sample_doc):
        i=0
        lst=[]
        for sent in sample_doc.sents:
            i=0
            for w in nlp(sent.text,disable=['textcat','ner','merge_entities','merge_subtokens','merge_noun_chunks']):
                if(not(w.is_punct)):
                    i=i+1
            lst.append(i)
        return sum(1 for obj in lst if obj>15)
    def __std_GetwrdsSentInfo(self,sample_doc):
        res=[]
        for sent in sample_doc.sents:
            res.append(sum(1 for w in nlp(sent.text, disable=['textcat','ner','merge_entities','merge_subtokens','merge_noun_chunks']) if not(w.is_punct)))
        return res

    #remove duplicate keys
    def __sumup_Match_Similarity(self, sample_lst):
        tmplst=[]
        for lst in sample_lst:
            tmplst.extend(lst)
        return list(dict.fromkeys(tmplst))
  
    #get most similar tokens between unmatched list and the whole golden set
    def __most_similar(self,unmatched_lst,gld_lst):
        sim=[]
        #similar match using threshold greater than 0.6
        for tkn1 in unmatched_lst:
            for tkn2 in gld_lst:
                num=nlp.vocab[tkn1].similarity(nlp.vocab[tkn2])
                if(num>0.6):
                    sim.append(tkn2)
                            
        return sim
    
    #define the match function between the two lists
    def __matching_keywords(self,stdlst,gldlst):
            #matched list
            res=[]
            #unmatched list
            tmpres=[]
            for x in stdlst:
                if x.lower() in gldlst:
                    res.append(x.lower())
                else:
                    tmpres.append(x.lower())
            return res,tmpres

    def __get__keywords(self, sample_doc):
        #define a function to get the frequent words in a text which aren't stop words or punctuations
        words = [token.lemma_.strip().lower() if token.lemma_ != '-PRON-' else token.lower_ for token in sample_doc if token.lemma_.strip().lower() and not (token.lemma_.lower() in stopwrds) and not token.is_punct and token.is_alpha]
        word_coun = Counter(words)
        wrd_frq=[wrd for wrd in word_coun]
        
        #extract unique words with only 1 occurrence
        uniq_wrds=[word for word in word_coun if (word_coun.get(word)==1)]
        
        #extract nouns chuncks and special tags like proper noun, noun, adjective
        res=[]
        for chk in sample_doc.noun_chunks:
            tmp=""
            for tkn in chk:
                if (tkn.pos_ in ['NOUN','PROPN','ADJ'] and tkn.lemma_.strip().lower() ):
                    if (not(tkn.is_stop) and not(tkn.is_punct) and tkn.is_alpha):
                        if tkn.lemma_ !='-PRON-':
                            tmp = tmp + tkn.lemma_.strip().lower() + " "
                        else:
                            tmp = tmp + tkn.lower_ + " "
            if(tmp.strip()!=""):
                res.append(tmp.strip())
        pos_lst= list(dict.fromkeys(res))
        
        key=self.__sumup_Match_Similarity([wrd_frq,uniq_wrds,pos_lst])
        return {'keywords': key, 'Count': len(key)}

    def __Find_match_and_similarity(self,std_key,gld_lst):
        res=[]
        #apply match function
        for gld in gld_lst:
            matched,unmatched=self.__matching_keywords(std_key,gld)
            similar=self.__most_similar(unmatched,gld)
            tmp=self.__sumup_Match_Similarity([matched,similar])
            res.append({'keywords': tmp,'Count':len(tmp),'Exact keywords':matched,'Count-E':len(matched),'Count-G':len(gld)})
        return res

    
    
    def __yule(self,sample_doc):
        # yule's I measure (the inverse of yule's K measure)
        # higher number is higher diversity - richer vocabulary
        d = {}
        ps = PorterStemmer()
        M1,M2=0,0
        for w in sample_doc:
            if(not(w.is_punct) and w.is_alpha):
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
    def __get_yules(self,sample_doc):
        """ 
        Returns a tuple with Yule's K and Yule's I.
        (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
        International Journal of Applied Linguistics, Vol 10 Issue 2)
        In production this needs exception handling.
        """
        ps = PorterStemmer()
        token_counter = Counter(ps.stem(tok.text.lower()) for tok in sample_doc if not(tok.is_punct) and tok.is_alpha)
        m1 = sum(token_counter.values())
        m2 = sum([freq ** 2 for freq in token_counter.values()])

        try:
            i = (m1*m1) / (m2-m1)
            k = 1/i * 10000
            return {'yule-k':k, 'yule-i':i}
        except ZeroDivisionError:
            return {'yule-k':float(0), 'yule-i':float(0)}
        

        
