import nltk
from nltk.corpus import brown
#from nltk.tokenize import word_tokenize
import numpy as np
#import pandas as pd
import re
import copy
#import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import random #for shuffling list

#from evaluation import *
nltk.download('punkt')
nltk.download('brown')
nltk.download('universal_tagset')

#l = brown.tagged_words(tagset='universal')

def morph_unknown(word):
    noun_suffixes = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism',
                     'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion',
                     'tion', 'ct']
    verb_suffixes = ['es', 'ed', 'ate', 'en', 'ify', 'fy', 'ize', 'ise',
                     'ers', 'ingly']
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
        possible_tag = "X"

    return possible_tag

all_sentences = list(brown.tagged_sents(tagset='universal'))

def sentence_decoding(training_sentences, testing_sentences):
    l = []
    #sentences = list(brown.tagged_sents(tagset='universal'))
    sentences = training_sentences


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
    1) A quick check between the number of words in l and total words over all sentences
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

    #for tag in tag_list:
    #    p_tag_to_tag[(tag, 'START')] = 0.0
    #    p_tag_to_tag[('END', tag)] = 0.0


    #p_tag_to_tag[('START', 'START')] = 0.0
    #p_tag_to_tag[('END', 'END')] = 0.0
    #p_tag_to_tag[('START', 'END')] = 0.0
        


    #---------------------VITERBI-------------------------

    actual_tags = []
    predicted_tags = []

    #print(len(all_sentences))
    all_tags = tag_list 

    transition_matrix=np.random.rand(len(all_tags),len(all_tags))
    for i in range(len(all_tags)):
        for j in range(len(all_tags)):
            try:
                transition_matrix[i][j] = p_tag_to_tag[(all_tags[i], all_tags[j])]
            except:
                transition_matrix[i][j]=0.0
    transition_matrix=transition_matrix.astype('float64')#(transition_matrix)

    all_tokens = list(words.keys())

    lexical_probs=np.random.rand(len(all_tags),len(all_tokens))
    for i in range(len(all_tags)):
        for j in range(len(all_tokens)):
            try:
                lexical_probs[i][j] = p_word_given_tag[(all_tokens[j], all_tags[i])]
            except:
                lexical_probs[i][j]=0.0
    lexical_probs=lexical_probs.astype('float64')#(lexical_probs)

    for iter in range( len(testing_sentences) ):

        #given_sent = ' '.join([str(elem[0]) for elem in testing_sentences[iter]])
        actual_tag_sent = []
        actual_tag_sent = testing_sentences[iter]

        #print(iter)
        #print(given_sent)
        #print(actual_tag_sent)
        actual_tag_sent.insert(0, ("^", "START"))
        actual_tag_sent.append(("//", "END"))
        actual_tags.append([i[1] for i in actual_tag_sent])

        #print(given_sent)
        
        #given_sent = given_sent.lower()

#---------------------PREPROCESSING-------------------------

        #example -
        #token_ex = re.findall(r"([\w'-]+|[.,!?\[\]\(\);:]|[`]+)", given_sent)
        token_ex = [i[0] for i in actual_tag_sent]
        #print(token_ex)
        

        for index in range(len(token_ex)):
            wrd = token_ex[index]
            if not wrd.lower() in words:
                assign = morph_unknown(wrd) #has possible assignable tag.
                for tag in tag_list:
                    p_word_given_tag[(wrd.lower(),tag)] = 1/len(tags[tag]) if tag == assign else 0
                    words[wrd.lower()] = [assign]
                all_tokens.append(wrd.lower())
                new_arr=np.array([p_word_given_tag[(wrd.lower(),alpha)] for alpha in tag_list])
                new_arr=new_arr.reshape(len(new_arr))
                #print(new_arr[:, None].shape)
                #print(lexical_probs.shape)
                
                lexical_probs=np.append(lexical_probs,new_arr[:, None],axis=1)
            token_ex[index] = wrd.lower()

        #print(token_ex)

#-----------------------VITERBI-----------------------------

        

        prob_matrix=np.array([[0.0]*len(all_tags)]*len(all_tags))
        for i in range(len(token_ex)-1):
            if i==0:
                prev_tag="START"
                transition_probs=np.array([transition_matrix[all_tags.index(prev_tag)][j] for j in range(len(all_tags))])
                obs_prob=lexical_probs[all_tags.index(prev_tag)][all_tokens.index(token_ex[i])]
                net_probs=obs_prob*transition_probs
                seq=[["START",all_tags[k]] for k in range(0,len(all_tags))]

            else:
                for k in range(0,len(all_tags)):
                    transition_probs=np.array([transition_matrix[k][j] for j in range(len(all_tags))])
                    obs_prob=lexical_probs[k][all_tokens.index(token_ex[i])]
                    prob_matrix[k]=obs_prob*net_probs[k]*transition_probs

                new_seq=[[] for s in range(len(all_tags))]
                for k in range(0,len(all_tags)):
                    #a=prob_matrix[:,k].max
                    idx=np.argmax(prob_matrix[:,k])
                    new_seq[k]=copy.deepcopy(seq[idx])
                    new_seq[k].append(all_tags[k])
                    net_probs[k]=prob_matrix[idx][k]
                seq=copy.deepcopy(new_seq)

        predicted_tags.append(seq[1])
        #print(seq[1])
        #print(actual_tags[iter])
        #print(predicted_tags[iter])


        #if( len(actual_tags[iter]) == len(predicted_tags[iter])):
        #    print(accuracy(actual_tags[iter], predicted_tags[iter])[0])
        #else:
        #    print(0)

    actual_total=[]
    predicted_total=[]
    for i in range(len(actual_tags)):
        for j in range(len(actual_tags[i])):
            actual_total.append(actual_tags[i][j])
            predicted_total.append(predicted_tags[i][j])

    my_matrix=confusion_matrix(predicted_total, actual_total, labels=all_tags)
    print(my_matrix)
    

    return actual_tags, predicted_tags, actual_total, predicted_total, all_tags

def accuracy_per_fold(training_sentences, test_sentences):
    actual_tags, predicted_tags, actual_total, predicted_total, all_tags = sentence_decoding(training_sentences, test_sentences)

    f1 = 0
    prec = 0
    recall = 0


    #for i in range( len(actual_tags) ):
    #    f1 += f1_score(actual_tags[i], predicted_tags[i], average='weighted')
    #    prec += precision_score(actual_tags[i], predicted_tags[i], average='weighted')
    #    recall += recall_score(actual_tags[i], predicted_tags[i], average='weighted')
    per_pos=[[] for i in range(0,len(all_tags))]
    #per_pos_act=[[] for i in range(0,12)]
    for i in range(len(actual_total)):
        per_pos[all_tags.index(actual_total[i])].append(all_tags.index(predicted_total[i]))
    print("my_fold------------------------")
    for i in range(len(per_pos)):
        print(all_tags[i],"-----")
        true_tag=[i for j in range(len(per_pos[i]))]
        print("precision",precision_score(per_pos[i],true_tag, average='weighted'))
        print("recall",recall_score(per_pos[i],true_tag, average='weighted'))
        print("f1",f1_score(per_pos[i],true_tag, average='weighted'))
    print("overall for the batch:")
    precision=precision_score(actual_total,predicted_total, average='weighted')
    recall=recall_score(actual_total,predicted_total, average='weighted')
    f1=f1_score(actual_total,predicted_total, average='weighted')
    print("precision",precision)
    print("recall",recall)
    print("f1",f1)

        
    return [precision, recall, f1]
    #return [f1/len(actual_tags), prec/len(actual_tags), recall/len(actual_tags)]

#total_acc_for_split1 = accuracy_per_fold(all_sentences[:57000], all_sentences[56000:56010])
#total_acc_for_split2 = accuracy_per_fold(all_sentences[:57000], all_sentences[57000:57010])

#print("\nTotal accuracy for the above split1: " + str(total_acc_for_split1))
#print("Total accuracy for the above split2: " + str(total_acc_for_split2) + "\n")
#exit(0)

def cross_validation_accuracy(all_sentences = all_sentences, folds = 5):
    all_accuracies = []
    total_acc = [0, 0, 0]

    num_sentences = len(all_sentences)

    fold_size = int(num_sentences / folds)

    print(fold_size)

    for fold in range(folds):
        random.shuffle(all_sentences)
        train = all_sentences[:fold_size*4]
        test = all_sentences[fold_size*4:]

        all_accuracies.append(accuracy_per_fold(train, test))
        
        print("Accuracy for fold ", str(fold), " is : ", str(all_accuracies[fold]))
        total_acc[0] += all_accuracies[fold][0]
        total_acc[1] += all_accuracies[fold][1]
        total_acc[2] += all_accuracies[fold][2]

    print("Cross validation accuracy is : ", str(np.array(total_acc) / folds))

cross_validation_accuracy()
