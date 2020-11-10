# TA
Keyword extraction and keyword matching from paragraphs 

#Installation requirements
•	Pip install nltk
•	Pip install numpy
•	Pip install spacy
•	Pip install language-tool-python
•	Pip install --upgrade language-check
•	Pip install --upgrade 3to2
•	Pip install lexicalrichness
•	Pip install lexical-diversity
•	Pip install textblob
•	Pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.3.0/en_core_web_md-2.3.0.tar.gz#egg=en_core_web_md
You can check by trying the above given url on http browser to know if it works or not.
•	If you have this error ReadTimeoutError
Then try to fix it by
•	Pip install -U –timeout 1000 https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.3.0/en_core_web_md-2.3.0.tar.gz#egg=en_core_web_md
Or
•	you can get out from the scripts folder using cd.. command to then main python folder then type the following command:
Python -m spacy download en_core_web_md
•	move to python.exe path using command line and type
python -m nltk.downloader stopwords
python -m textblob.download_corpora lite

#How to call scripts from prompt
Type the following command to call keyword extraction module for a paragraph
          Python myapp.py “some paragraph here between these quotes”   
Type the following command to the keyword extraction and matching module, you must take care of spaces as shown in the example
          Python myapp.py “student paragraph here” [“keyword 1 for ideal 1”,”keyword 2 for ideal 1”] [“keyword1 for ideal 2”,”keyword2 for ideal2”]
Type the following command to call two sentences match to detect omitted and added words
          Python QAmatch.py “sentence1” “sentence2”
          
#Json return tags are:

A - Keyword extraction module, 
The input is just one paragraph to extract its information, the returned json tags are:
1-	'All Caps': number Caps lock words in paragraph
2-	'tag': number of tags used in the student paragraph
a.	It consists different tags in paragraph 
i.	“VERB”: number of verbs
ii.	“NOUN”: number of nouns
iii.	“PROPN”: number of proper nouns like countries, cities, etc.
iv.	There are other tags that you may not need, you can ignore it or display it.
3-	'OOV': number of misspelled words in the student paragraph
4-	'tenses': number of tenses in the student paragraph
a.	It consists of the following tags:
i.	“past”: number of past verbs
ii.	“present”: number of present verbs
iii.	“future”: number of future verbs
5-	'Active': number of active verbs
6-	'Passive': number of passive verbs 
7-	'words': number of words that are not stop words or punctuation
8-	'stop': number of stop words
9-	'Punct': number of punctuations
10-	'sent': number of all sentences in the input paragraph
11-	'longsent': number of long sentences in the input paragraph that exceed 15 words
12-	‘sent info’: number of words in each sentence
13-	‘sentences similarity’: list of values represents cosine similarity between each two successive sentences in the input paragraph.
14-	'grammar errors': number of grammar errors using grammar-checker api.
15-	‘Keywords list’: it extracts the most frequent, unique and noun phrases keywords in one list without redundancy. 
It consists of the following tags:
o	“keywords”: It shows the extracted keywords list
o	“Count”: the count of total keywords 
16-	'yule': number represents the richness of vocabulary in the paragraph, high numbers represent high richness and diversity between words in the input paragraph and low numbers show low diversity between words.
17-	“Other Yules”: another calculation equation to measure vocabulary richness
a.	“yule-k”: yule-k measures richness of vocabulary, the lowest value is the richest in vocabulary.
b.	“yule-i”: yule-i measures richness of vocabulary, the highest value is the richest in vocabulary.
18-	Lexical diversity:
a.	"TTR": simple type token ratio
b.	"RTTR": root type token ratio
c.	"LTTR": log type token ratio
d.	"massTTR": mass type token ratio
e.	"MSTTR": mean segmental token type ratio
f.	"MATTR": move average type token ratio
g.	"HDD": hyperdemotric distribution diversity
h.	"MTLD": lexical textual diversity
i.	"MTLD_MW": moving average lexical textual diversity
j.	"MTLD_MA": moving average lexical textual diversity using moving window

B - Keyword extraction and Matching module
The input parameters are one student paragraph and different number of ideal keywords lists, lists are separated by spaces, the returned json tags are:
1-	'All Caps': number Caps lock words in the student paragraph
2-	'tag': number of tags used in the student paragraph
It consists different tags in paragraph 
i.	“VERB”: number of verbs
ii.	“NOUN”: number of nouns
iii.	“PROPN”: number of proper nouns like countries
3-	'OOV': number of misspelled words in the student paragraph
4-	'tenses': number of tenses in the student paragraph
It consists of the following tags:
i.	“past”: number of past verbs
ii.	“present”: number of present verbs
iii.	“future”: number of future verbs
5-	'Active': number of active verbs
6-	'Passive': number of passive verbs 
7-	'words': number of words that are not stop words or punctuation in the student paragraph
8-	'stop': number of stop words used by the student
9-	'Punct': number of punctuations used by the student
10-	'sent': number of all sentences in the student paragraph
11-	'longsent': number of long sentences in the student paragraph that exceed 150 words
12-	‘sent info’: number of words in each sentence
13-	‘sentences similarity’: list of values represents cosine similarity between each two successive sentences in the input paragraph.
14-	‘documents similarities’: list of values represents cosine similarity between student document and each ideal keyword list.
15-	'grammar errors': number of grammar errors using grammar-checker api.
16-	‘Keywords list’: it extracts keywords from student paragraph and compare it with multiple ideal keywords lists that are extracted before. It takes the exact match keywords and the highest similar keyword vectors which are greater than 0.6 cosine similarity. 
It consists of the following tags for each golden document keywords:
o	“keywords”: It shows the whole keyword list
o	“Count”: the count of total keywords 
o	“Exact keywords”: its shows exact matched keywords only
o	“Count-E”: the exact keywords matched count
o	“Count-G”: the golden keywords list count
17-	'yule': number represents the richness of vocabulary in the student paragraph, high numbers represent high richness and diversity between words in the input paragraph and low numbers show low diversity between words.
19-	“Other Yules”: another calculation equation to measure vocabulary richness of student paragraph
a.	“yule-k”: yule-k measures richness of vocabulary, the lowest value is the richest in vocabulary.
b.	“yule-i”: yule-i measures richness of vocabulary, the highest value is the richest in vocabulary.
20-	Lexical diversity:
a.	"TTR": simple type token ratio
b.	"RTTR": root type token ratio
c.	"LTTR": log type token ratio
d.	"massTTR": mass type token ratio
e.	"MSTTR": mean segmental token type ratio
f.	"MATTR": move average type token ratio
g.	"HDD": hyperdemotric distribution diversity
h.	"MTLD": lexical textual diversity
i.	"MTLD_MW": moving average lexical textual diversity
j.	"MTLD_MA": moving average lexical textual diversity using moving window


C – Question and Answer Matching
The input parameters are two sentences: the first is question sentence, and the second is the answer. The returned json tags are:
1-	'omitted': omitted words from question sentence
2-	‘added’: added words in answer sentence
3-	“’omitted-len”: length of omitted words
4-	“added-len”: length of added words
5-	“Q len”: total length of question
6-	“A len”: total length of answer


