import nltk
nltk.download('brown')
nltk.download('punkt')
nltk.download('universal_tagset')
from nltk.corpus import brown
#from nltk.tokenize import word_tokenize
import numpy as np
#import pandas as pd
import re
import copy


#l = brown.tagged_words(tagset='universal')
l = []
sentences = list(brown.tagged_sents(tagset='universal'))

for i in range(len(sentences)):
    test = sentences[i]
    test.insert(0, ("^", "START"))
    test.append(("\\", "END"))
    sentences[i] = test
    l.append(("^", "START"))
    l.append(("\\", "END"))
    for j in sentences[i]:
        l.append(j)
    

tags = {}
for i in l:
    if i[1] not in tags:
        tags[i[1]] = [i[0]]
    else:
        tags[i[1]].append(i[0])

words = {}
for i in l:
    wrd = i[0].lower()
    if wrd not in words:
        words[wrd] = [i[1]]
    else:
        words[wrd].append(i[1])

tag_list = list(tags.keys())

'''
Discussions

1) A quick check between the number of tagged words in corpus and total words over all sentences
come out to be 1161192 both meaning nothing is in excess/less.
2) tags['.'] contains punctuations, but ^, \\ not included so use as start and end token
4) tags['X'] has 'personnel', implying not all words are sorted correctly.
5) words['fox'] has only NOUNs, implying corpus might be old.
6) 7994 sentences don't end with a full stop.
'''

#Calculating P(tag|word)
p_tag_given_word = {}
for tag in tag_list:
    for wrd in words:
        p_tag_given_word[( tag, wrd )] = words[wrd].count(tag)/len(words[wrd]) #Word with given tag count / total count of the word 'wrd' in the corpus

p_word_given_tag = {}
for tag in p_tag_given_word:
    p_word_given_tag[ tag[::-1] ] = p_tag_given_word[tag] * ( len(words[ tag[1] ])/len(l) ) / ( len(tags[ tag[0] ])/len(l) )

#Transition Probabilities P(tag -> tag)
p_tag_to_tag = {}

for item in sentences:
    for g in range(len(item)-1):
        if ( item[g][1], item[g+1][1] ) not in p_tag_to_tag:
            p_tag_to_tag[( item[g][1], item[g+1][1] )] = 1.0
        else:
            p_tag_to_tag[( item[g][1], item[g+1][1] )] += 1.0 #P(noun -> verb) += 1, we store count here

for t in p_tag_to_tag:
    p_tag_to_tag[t] /= len(tags[t[0]]) #P(noun -> verb) /= num(nouns)



#---------------------PREPROCESSING-------------------------

given_sent = input("Enter a sentence: \n")#.lower()

#example -
token_ex = re.findall(r"([\w'-]+|[.,!?\[\]\(\);:]|[`]+)", given_sent) #Counts multiple instances of `` for quotes, have to convert `` '' to `` `` in preprocess.
token_ex.insert(0, "^")
token_ex.append("\\")

#https://www.thoughtco.com/common-suffixes-in-english-1692725
#Using above for suffix analysis and some more common verb ones..
def morph_unknown(word):
    noun_suffixes = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism',
                     'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion',
                     'tion', 'ct']
    verb_suffixes = ['es', 'ed', 'ate', 'en', 'ify', 'fy', 'ize', 'ise',
                     'ers']
    adj_suffixes = ['able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical',
                    'ious', 'ous', 'ish', 'ive', 'less', 'y']

    possible_tag = ""
    for i in noun_suffixes:
        if len(re.findall(i+"$", word)) > 0:
            possible_tag = "NOUN"

    for i in verb_suffixes:
        if len(re.findall(i+"$", word)) > 0:
            possible_tag = "VERB"

    for i in adj_suffixes:
        if len(re.findall(i+"$", word)) > 0:
            possible_tag = "ADJ"
    
    if len(re.findall("^[A-Z]", word)):#Capital Letter, usually nouns.
        possible_tag = "NOUN"

    try:
        float(word) #If it works then assign NUM tag
        possible_tag = "NUM"
    except:
        pass

    if possible_tag == "":
        possible_tag = "X" #Or X?

    return possible_tag
    

for index in range(len(token_ex)):
    wrd = token_ex[index]
    if not wrd.lower() in words:
        assign = morph_unknown(wrd) #has possible assignable tag.
        for tag in tag_list:
            p_word_given_tag[(wrd.lower(),tag)] = 1 if tag == assign else 0
            words[wrd.lower()] = [assign]
    token_ex[index] = wrd.lower()

print(token_ex)

#-----------------------VITERBI-----------------------------

all_tokens = list(words.keys())
all_tags = tag_list 

transition_matrix=np.random.rand(len(all_tags),len(all_tags))
for i in range(len(all_tags)):
    for j in range(len(all_tags)):
        try:
            transition_matrix[i][j] = p_tag_to_tag[(all_tags[i], all_tags[j])]
        except:
            transition_matrix[i][j]=0.0
transition_matrix=transition_matrix.astype('float64')#(transition_matrix)

lexical_probs=np.random.rand(len(all_tags),len(all_tokens))
for i in range(len(all_tags)):
    for j in range(len(all_tokens)):
        try:
            lexical_probs[i][j] = p_word_given_tag[(all_tokens[j], all_tags[i])]
        except:
            lexical_probs[i][j]=0.0
lexical_probs=lexical_probs.astype('float64')#(lexical_probs)

output_tags=['START']
prob_matrix=np.array([[0.0]*len(all_tags)]*len(all_tags))
for i in range(len(token_ex)-1):
    if i==0:
        prev_tag=output_tags[-1]
        transition_probs=np.array([transition_matrix[all_tags.index(prev_tag)][j] for j in range(len(all_tags))])
        obs_prob=lexical_probs[all_tags.index(prev_tag)][all_tokens.index(token_ex[i])]
        net_probs=obs_prob*transition_probs
        seq=[[output_tags[-1],all_tags[k]] for k in range(0,len(all_tags))]

    else:
        for k in range(0,len(all_tags)):
            transition_probs=np.array([transition_matrix[k][j] for j in range(len(all_tags))])
            obs_prob=lexical_probs[k][all_tokens.index(token_ex[i])]
            prob_matrix[k]=obs_prob*net_probs[k]*transition_probs

        new_seq=[[] for s in range(len(all_tags))]
        for k in range(0,len(all_tags)):
            a=prob_matrix[:,k].max
            idx=np.argmax(prob_matrix[:,k])
            new_seq[k]=copy.deepcopy(seq[idx])
            new_seq[k].append(all_tags[k])
            net_probs[k]=prob_matrix[idx][k]
        seq=copy.deepcopy(new_seq)

print(seq[1])

#input("Press [ENTER] to exit")
