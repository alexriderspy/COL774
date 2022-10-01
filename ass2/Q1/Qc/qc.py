from math import log
import random
import sys
import os
from tabulate import tabulate

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

for (dirpath, dirnames, filenames) in os.walk(os.path.join(train_path,'/pos')):
    for filename in filenames:
        cnt_pos +=1
        my_files_pos.append(os.path.join(train_path,'/pos/',filename))
    break

for (dirpath, dirnames, filenames) in os.walk(os.path.join(test_path,'/pos')):
    for filename in filenames:
        test_pos.append(os.path.join(test_path,'/pos/' ,filename))
    break

for (dirpath, dirnames, filenames) in os.walk(os.path.join(train_path,'/neg')):
    for filename in filenames:
        cnt_neg+=1
        my_files_neg.append(os.path.join(train_path,'/neg/',filename))
    break

for (dirpath, dirnames, filenames) in os.walk(os.path.join(test_path, '/neg')):
    for filename in filenames:
        test_neg.append(os.path.join(test_path,'/neg/' ,filename))
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

#accuracy of test data

pos_accu = calc_accu(test_pos,'pos')
neg_accu = calc_accu(test_neg,'neg')

tp = pos_accu
tn = neg_accu
fp = len(test_neg)-neg_accu
fn = len(test_pos)-pos_accu

table = [["Positive (predicted)", tp,fp],["Negative (predicted)",fn,tn]]
print("Confusion matrix for Naive Bayes model")
print(tabulate(table, headers=["Positive(actual)","Negative(actual)"]))

def calc_accu_rand(files,type):
    accu = 0
    for file in files:
        num = random.randint(0,1)
        if (num == 0 and type == 'neg') or (num == 1 and type == 'pos'):
            accu+=1
    return accu

#accuracy of test data

pos_accu = calc_accu_rand(test_pos,'pos')
neg_accu = calc_accu_rand(test_neg,'neg')

tp = pos_accu
tn = neg_accu
fp = len(test_neg)-neg_accu
fn = len(test_pos)-pos_accu

table = [["Positive (predicted)", tp,fp],["Negative (predicted)",fn,tn]]
print("Confusion matrix for random model")
print(tabulate(table, headers=["Positive(actual)","Negative(actual)"]))

tp = len(test_pos)
tn = 0
fp = len(test_neg)
fn = 0

table = [["Positive (predicted)", tp,fp],["Negative (predicted)",fn,tn]]
print("Confusion matrix for all positive model")
print(tabulate(table, headers=["Positive(actual)","Negative(actual)"]))
