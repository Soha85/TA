import difflib
import sys
import nltk
import json
import string
import csv

#initilaize empty dictionary variable
uk2us_dic=dict([])
with open('uk2us.csv','r') as data: 
   for line in csv.reader(data):
       uk2us_dic[line[0]]=line[1] 
          
    
#define function to replace uk to us
def replace_all(text):
    for tkn in text.split(' '):
        try:
            text = text.replace(tkn, uk2us_dic[tkn])
        except:
            err=0
    return text


def QA_Match(q,a):

    #replace uk to us
    q=replace_all(q.lower())
    a=replace_all(a.lower())
    
    #tokenize input sentences to tokens
    s1=nltk.word_tokenize(q)
    s2=nltk.word_tokenize(a)

    #initialize arrays
    omitted,added=[],[]

    #remove punctuations words
    s1=[wrd for wrd in s1 if wrd not in string.punctuation]
    s2=[wrd for wrd in s2 if wrd not in string.punctuation]
    

    
        
    #get omitted and added tokens using difflib
    for line in difflib.unified_diff(s1, s2):
        if(line[0]=='-' and not("\n") in line[1:]):
            omitted.append(line[1:])
        if(line[0]=='+' and not("\n") in line[1:]):
            added.append(line[1:])    
    
    #return json tags
    return json.dumps({"omitted":omitted,"omitted-len":len(omitted),
                       "added":added,"added-len":len(added),
                       "Q len":len(s1),"A len":len(s2)})


if __name__=="__main__":
    try:
        if(len(sys.argv)>2):
            #load question argument
            q=sys.argv[1]
            #load answer argument
            a=sys.argv[2]
            #call function to get matches
            obj=QA_Match(q,a)
            #print the result
            print(obj)
        else:
            print({"error":"arguments incorrect"})
    except Exception as err:
        print({"error":err})
