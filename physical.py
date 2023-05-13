import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn 
import csv
import json
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import itertools
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.corpus import wordnet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
import tkinter as tk
from tkinter import simpledialog

bot_name = "HEAL"


data={"users":[]}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)

def write_json(new_data, filename='DATA.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["users"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)


df_tr=pd.read_csv('Medical_dataset/Training.csv')
df_tt=pd.read_csv('Medical_dataset/Testing.csv')


symp=[]
disease=[]
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i]==1].to_list())
    disease.append(df_tr.iloc[i,-1])



# ## I- GET ALL SYMPTOMS

all_symp_col=list(df_tr.columns[:-1])
def clean_symp(sym):
    return sym.replace('_',' ').replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow') 


all_symp=[clean_symp(sym) for sym in (all_symp_col)]

## get all symptoms which do not have a synset
ohne_syns=[]
mit_syns=[]
for sym in all_symp:
    if not wn.synsets(sym) :
        ohne_syns.append(sym)
    else:
        mit_syns.append(sym)


# ## II- Preprocess text
nlp = spacy.load('en_core_web_sm')

def preprocess(doc):
    nlp_doc=nlp(doc)
    d=[]
    for token in nlp_doc:
        if(not token.text.lower()  in STOP_WORDS and  token.text.isalpha()):
            d.append(token.lemma_.lower() )
    return ' '.join(d)


def preprocess_sym(doc):
    nlp_doc=nlp(doc)
    d=[]
    for token in nlp_doc:
        if(not token.text.lower()  in STOP_WORDS and  token.text.isalpha()):
            d.append(token.lemma_.lower() )
    return ' '.join(d)

all_symp_pr=[preprocess_sym(sym) for sym in all_symp]

#associe chaque symp pretraite au non de sa colonne originale
col_dict = dict(zip(all_symp_pr, all_symp_col))


# ## III- Syntactic Similarity
def jaccard_set(str1, str2):
    list1=str1.split(' ')
    list2=str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

#similarite syn avec ts le corpus
def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(symp_t):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None


#Returns all the subsets of this set. This is a generator.
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

#Sort list based on length
def sort(a):
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if len(a[j])>len(a[i]):
                a[i],a[j]=a[j],a[i]
    a.pop()
    return a

# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return([' '.join(permutation) for permutation in permutations])

def DoesExist(txt):
    txt=txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations :
        #print(permutations(comb))
        for sym in permutations(comb):
            if sym in all_symp_pr:
                #print(sym)
                return sym
    return False

def check_pattern(inp,dis_list):
    import re
    pred_list=[]
    ptr=0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return ptr,None


# ## IV- Semantic Similarity

def WSD(word, context):
    sens=lesk(context, word)
    return sens


def semanticD(doc1,doc2):
    doc1_p=preprocess(doc1).split(' ')
    doc2_p=preprocess_sym(doc2).split(' ')
    score=0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1,doc1)
            syn2 = WSD(tock2,doc2)
            #syn1=wn.synset(t)
            if syn1 is not None and syn2 is not None :
                x=syn1.wup_similarity(syn2)
                if x is not None and x>0.1:
                    score+=x
    return score/(len(doc1_p)*len(doc2_p))


anxiety_synsets = wn.synsets("brittle") 
nervous_synsets = wn.synsets("nervous") 
path=[]
wup=[]
lch=[]


for s1 in anxiety_synsets:
    for s2 in nervous_synsets:
        path.append(s1.path_similarity(s2))
        wup.append(s1.wup_similarity(s2))
        #lch.append(s1.lch_similarity(s2))
        

pd.DataFrame([path,wup],["path","wup"])

#similarite sem avec ts le corpus
def semantic_similarity(symp_t,corpus):
    max_sim=0
    most_sim=None
    for symp in corpus:
        d=semanticD(symp_t,symp)
        if d>max_sim:
            most_sim=symp
            max_sim=d
    return max_sim,most_sim

all_symp_pr.sort()


def suggest_syn(sym):
    symp=[]
    synonyms = wordnet.synsets(sym)
    lemmas=[word.lemma_names() for word in synonyms]
    lemmas = list(set(chain(*lemmas)))
    for e in lemmas:
        res,sym1=semantic_similarity(e,all_symp_pr)
        if res!=0:
            symp.append(sym1)
    return list(set(symp))

#recoit client_symptoms et renvoit un dataframe avec 1 pour les symptoms associees
def OHV(cl_sym,all_sym):
    l=np.zeros([1,len(all_sym)])
    for sym in cl_sym:
        l[0,all_sym.index(sym)]=1
    return pd.DataFrame(l, columns =all_symp)
    
def contains(small, big):
    a=True
    for i in small:
        if i not in big:
            a=False
    return a

def possible_diseases(l):
    poss_dis=[]
    for dis in set(disease):
        if contains(l,symVONdisease(df_tr,dis)):
            poss_dis.append(dis)
    return poss_dis

set(disease)

#recoit une maladie renvoit tous les sympts
def symVONdisease(df,disease):
    ddf=df[df.prognosis==disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()


# ##  VI- SEVERITY / DESCRIPTION / PRECAUTION

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

def getDescription():
    global description_list
    with open('Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('Medical_dataset/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Medical_dataset/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


getSeverityDict()
getprecautionDict()
getDescription()


def calc_condition(exp,days):
    sum=0
    for item in exp:
        if item in severityDictionary.keys():
            sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp))>13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")

# # Chat

def getInfo():
    # name=input("Name:")
    print("Hi, what's your name\n\t\t\t\t\t\t",end="=>")
    name=input("")
    print("Nice to see you here today ",name)
    return str(name)


def related_sym(psym1):
    if len(psym1)==1:
        return psym1[0]
    print("searches related to input: ")
    for num,it in enumerate(psym1):
        print(num,")",clean_symp(it))
    if num!=0:
        print(f"Select the one you meant (0 - {num}):  ", end="")
        conf_inp = int(input(""))
    else:
        conf_inp=0

    disease_input=psym1[conf_inp]
    return disease_input


def main_sp(msg):
    #main Idea: At least two initial sympts to start with
    
    #get the 1st syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)
    print("Enter the main symptom you are experiencing, "+"  \n\t\t\t\t\t\t",end="=>")
    
    sym1 = msg
    sym1=preprocess_sym(sym1)
    sim1,psym1=syntactic_similarity(sym1,all_symp_pr)
    if sim1==1:
        psym1=related_sym(psym1)
    
    #get the 2nd syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)

    print("Enter a second symptom you are experiencing, "+"  \n\t\t\t\t\t\t",end="=>")
    sym2=input("")
    sym2=preprocess_sym(sym2)
    sim2,psym2=syntactic_similarity(sym2,all_symp_pr)
    if sim2==1:
        psym2=related_sym(psym2)
        
    #if check_pattern==0 no similar syntaxic symp1 or symp2 ->> try semantic similarity
    
    if sim1==0 or sim2==0:
        sim1,psym1=semantic_similarity(sym1,all_symp_pr)
        sim2,psym2=semantic_similarity(sym2,all_symp_pr)
        
        #if semantic sim syp1 ==0 (no symp found) ->> suggest possible data symptoms based on all data and input sym synonymes
        if sim1==0:
            sugg=suggest_syn(sym1)
            print('Are you experiencing any ')
            for res in sugg:
                print(res)
                inp=input('')
                if inp=="yes":
                    psym1=res
                    sim1=1
                    break
                
        #if semantic sim syp2 ==0 (no symp found) ->> suggest possible data symptoms based on all data and input sym synonymes
        if sim2==0:
            sugg=suggest_syn(sym2)
            for res in sugg:
                inp=input('Do you feel '+ res+" ?(yes or no) ")
                if inp=="yes":
                    psym2=res
                    sim2=1
                    break
        #if no syntaxic semantic and suggested sym found return None and ask for clarification

        if sim1==0 and sim2==0:
            return None,None
        else:
            # if at least one sym found ->> duplicate it and proceed
            if sim1==0:
                psym1=psym2
            if sim2==0:
                psym2=psym1
    #create patient symp list
    all_sym=[col_dict[psym1],col_dict[psym2]]
    #predict possible diseases
    diseases=possible_diseases(all_sym)
    stop=False
    print("Are you experiencing any ")
    for dis in diseases:
        if stop==False:
            for sym in symVONdisease(df_tr,dis):
                if sym not in all_sym:
                    print(clean_symp(sym)+' ?')
                    while True:
                        inp=input("")
                        if(inp=="yes" or inp=="no"):
                            break
                        else:
                            print("provide proper answers i.e. (yes/no) : ",end="")
                    if inp=="yes":
                        all_sym.append(sym)
                        diseases=possible_diseases(all_sym)
                        if len(diseases)==1:
                            stop=True 
    return knn_clf.predict(OHV(all_sym,all_symp_col)),all_sym
    
    
def chat_sp():
    while True:
        #name=getInfo()
        
        result,sym=main_sp(all_symp_col)
        if result == None :
            ans3=input("can you specify more what you feel or tap q to stop the conversation")
            if ans3=="q":
                break
            else:
                continue

        else:
            print("you may have "+result[0])
            print(description_list[result[0]])
            an=input("how many day do you feel those symptoms ?")
            if calc_condition(sym,int(an))==1:
                print("you should take the consultation from doctor")
            else : 
                print('Take following precautions : ')
                for e in precautionDictionary[result[0]]:
                    print(e)
            print("do you need another medical consultation (yes or no)? ")
            ans=input()
            if ans!="yes":
                print("!!!!! THANKS FOR YOUR VISIT :) !!!!!! ")
                break


i=1
def get_response(inp):
    global i
    global sim1,psym1,sym1,sym2,sim2,psym2,all_sym,all_symp_pr,result,sym,all_symp_col
    
    if i==1:
        sym1 = inp
        sym1=preprocess_sym(sym1)
        sim1,psym1=syntactic_similarity(sym1,all_symp_pr)
        if sim1==1:
            psym1=related_sym(psym1)
        i=i+1
        return "Enter a second symptom you are experiencing,\n\n"
    
    elif i==2:
        sym2=inp
        sym2=preprocess_sym(sym2)
        sim2,psym2=syntactic_similarity(sym2,all_symp_pr)
        if sim2==1:
            psym2=related_sym(psym2)
         #if check_pattern==0 no similar syntaxic symp1 or symp2 ->> try semantic similarity
        if sim1==0 or sim2==0:
            sim1,psym1=semantic_similarity(sym1,all_symp_pr)
            sim2,psym2=semantic_similarity(sym2,all_symp_pr)
        else:
            # if at least one sym found ->> duplicate it and proceed
            if sim1==0:
                psym1=psym2
            if sim2==0:
                psym2=psym1
        #create patient symp list
        all_sym=[col_dict[psym1],col_dict[psym2]]
        #predict possible diseases
        result,sym=knn_clf.predict(OHV(all_sym,all_symp_col)),all_sym
        if result == None :
            i=3
            return "can you specify more what you feel or tap q to stop the conversation"
        
        else:
            i=4
            return "you may have "+result[0] + '\n' + description_list[result[0]] + "how many day do you feel those symptoms ?"
    
    elif i==3:
            ans3=inp
            if ans3=="q":
                return
            else:
                i==1
                return "Enter the main symptom"
            
    elif i==4:
        an = inp
        if calc_condition(sym,int(an))==1:
                i==5
                return "you should take the consultation from doctor" + '\n' + "do you need another medical consultation (yes or no)? "
        else : 
                st = 'Take following precautions : '
                for e in precautionDictionary[result[0]]:
                    st = st + '\n'+ e
                i==5
                return st + '\n' + "do you need another medical consultation (yes or no)? "
    
    elif i==5:
            ans=inp
            if ans!="yes":
                return  "!!!!! THANKS FOR YOUR VISIT :) !!!!!! "

    

#knn_clf=joblib.load('model/knn.pkl')  

#chat_sp()





