from math import log
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import os

#positive data

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])

my_files_pos = []
my_files_neg = []
test_pos = []
test_neg = []

cnt_pos = 0
cnt_neg = 0

cnt_pos_lis = {}
cnt_neg_lis = {}
cnt_lis = {}

cnt_wrds_pos = 0
cnt_wrds_neg = 0
cnt_wrds = 0

for (dirpath, dirnames, filenames) in os.walk(os.join(train_path,'/pos')):
    for filename in filenames:
        cnt_pos +=1
        my_files_pos.append(os.join(train_path,'/pos/',filename))
    break

for (dirpath, dirnames, filenames) in os.walk(os.join(test_path,'/pos')):
    for filename in filenames:
        test_pos.append(os.join(test_path,'/pos/' ,filename))
    break

for (dirpath, dirnames, filenames) in os.walk(os.join(train_path,'/neg')):
    for filename in filenames:
        cnt_neg+=1
        my_files_neg.append(os.join(train_path,'/neg/',filename))
    break

for (dirpath, dirnames, filenames) in os.walk(os.join(test_path, '/neg')):
    for filename in filenames:
        test_neg.append(os.join(test_path,'/neg/' ,filename))
    break

pos_prob = cnt_pos/(cnt_pos+cnt_neg)
neg_prob = cnt_neg/(cnt_pos+cnt_neg)

for file in my_files_pos:
    txt = open(file,'r').read()
    words = txt.split()
    for w in words:
        if cnt_pos_lis.get(w) == None:
            cnt_pos_lis[w]=1
        else:
            cnt_pos_lis[w] = cnt_pos_lis[w]+1
        
        cnt_wrds_pos += 1
        cnt_wrds += 1
        
        if cnt_lis.get(w) == None:
            cnt_lis[w]=1
        else:
            cnt_lis[w] = cnt_lis[w]+1

for file in my_files_neg:
    txt = open(file,'r').read()
    words = txt.split()
    for w in words:
        if cnt_neg_lis.get(w) == None:
            cnt_neg_lis[w]=1
        else:
            cnt_neg_lis[w] = cnt_neg_lis[w]+1
        
        cnt_wrds_neg += 1
        cnt_wrds += 1
        
        if cnt_lis.get(w) == None:
            cnt_lis[w]=1
        else:
            cnt_lis[w] = cnt_lis[w]+1


output_pos = ''
output_neg = ''

for item0 in cnt_pos_lis:
    for c in str(cnt_pos_lis[item0]):
        output_pos += item0 + ' '

for item0 in cnt_neg_lis:
    for c in str(cnt_neg_lis[item0]):
        output_neg += item0 + ' '

stopwords = set()
stopwords.add('br')

wc_pos = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

wc_pos.generate(output_pos)

fig = plt.figure()
fig.set_figwidth(14) 
fig.set_figheight(18)

plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
fig.savefig('pos_wc.png')

wc_neg = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

wc_neg.generate(output_neg)

fig = plt.figure()
fig.set_figwidth(14) 
fig.set_figheight(18)

plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
fig.savefig('neg_wc.png')

#accuracy for a .txt file

#accuracy for train data

accu = 0

def calc_accu(files,type):
    global cnt_pos_lis,cnt_wrds_neg,cnt_wrds_pos,cnt_pos_lis,cnt_wrds,cnt_lis,cnt_neg_lis,pos_prob,neg_prob
    accu = 0
    for file in files:

        test_file = open(file,'r').read()
        words = test_file.split()
        log_p_pos = 0
        log_p_neg = 0

        for w in words:
            if cnt_pos_lis.get(w) == None:
                log_p_pos += log((1)/(cnt_wrds_pos + len(cnt_lis)))
            else:
                log_p_pos += log((1+cnt_pos_lis[w])/(cnt_wrds_pos + len(cnt_lis)))
            
            if cnt_neg_lis.get(w) == None:
                log_p_neg += log((1)/(cnt_wrds_neg + len(cnt_lis)))
            else:
                log_p_neg += log((1+cnt_neg_lis[w])/(cnt_wrds_neg + len(cnt_lis)))

        log_p_pos += log(pos_prob)
        log_p_neg += log(neg_prob)

        if (log_p_pos>log_p_neg and type == 'pos') or (log_p_neg>log_p_pos and type == 'neg'):
            accu += 1
    return accu

accu += calc_accu(my_files_pos,'pos') + calc_accu(my_files_neg,'neg')

accu = accu / (len(my_files_pos)+len(my_files_neg))
print("Accuracy of train data : "+ str(accu))

#accuracy of test data

accu = 0

accu += calc_accu(test_pos,'pos') + calc_accu(test_neg,'neg')

accu = accu / (len(test_pos)+len(test_neg))
print("Accuracy of test data : "+ str(accu))

