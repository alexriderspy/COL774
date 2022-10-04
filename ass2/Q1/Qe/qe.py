from math import log
import sys,os
from wordcloud import STOPWORDS
from nltk.stem import SnowballStemmer

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
cnt2_pos_lis = {}
cnt2_neg_lis = {}
cnt2_lis = {}
cnt3_pos_lis = {}
cnt3_neg_lis = {}
cnt3_lis = {}

cnt_wrds_pos = 0
cnt_wrds_neg = 0
cnt_wrds = 0
cnt2_wrds_pos = 0
cnt2_wrds_neg = 0
cnt2_wrds = 0
cnt3_wrds_pos = 0
cnt3_wrds_neg = 0
cnt3_wrds = 0

for (dirpath, dirnames, filenames) in os.walk(train_path + '/pos/'):
    for filename in filenames:
        cnt_pos +=1
        my_files_pos.append(train_path + '/pos/' + filename)
    break

for (dirpath, dirnames, filenames) in os.walk(test_path + '/pos/'):
    for filename in filenames:
        test_pos.append(test_path+'/pos/' +filename)
    break

for (dirpath, dirnames, filenames) in os.walk(train_path+'/neg/'):
    for filename in filenames:
        cnt_neg+=1
        my_files_neg.append(train_path+'/neg/'+filename)
    break

for (dirpath, dirnames, filenames) in os.walk(test_path+'/neg/'):
    for filename in filenames:
        test_neg.append(test_path+'/neg/' +filename)
    break

pos_prob = cnt_pos/(cnt_pos+cnt_neg)
neg_prob = cnt_neg/(cnt_pos+cnt_neg)

snowball = SnowballStemmer(language='english')
stopwords = STOPWORDS
stopwords.add('br')
stopwords.add('th')
stopwords.add('h')
stopwords.add('b')
stopwords.add('thi')
stopwords.add('re')
stopwords.add('sh')
stopwords.add('lik')

for file in my_files_pos:
    txt = open(file,'r').read()
    words = txt.split()
    new_words = []
    for w in words:
        if len(w) > 5:
            w = snowball.stem(w)
        if w in stopwords:
            continue
        new_words.append(w)
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
    
    for i in range(len(new_words)-1):
        w0 = new_words[i]
        w1 = new_words[i+1]
        if cnt2_pos_lis.get((w0,w1))== None:
            cnt2_pos_lis[(w0,w1)]=1
        else:
            cnt2_pos_lis[(w0,w1)] = cnt2_pos_lis[(w0,w1)]+1
        
        cnt2_wrds_pos += 1
        cnt2_wrds += 1

        if cnt2_lis.get((w0,w1)) == None:
            cnt2_lis[(w0,w1)]=1
        else:
            cnt2_lis[(w0,w1)] = cnt2_lis[(w0,w1)]+1

    for i in range(len(new_words)-2):
        w0 = new_words[i]
        w1 = new_words[i+1]
        w2 = new_words[i+2]

        if cnt3_pos_lis.get((w0,w1,w2))== None:
            cnt3_pos_lis[(w0,w1,w2)]=1
        else:
            cnt3_pos_lis[(w0,w1,w2)] = cnt3_pos_lis[(w0,w1,w2)]+1
        
        cnt3_wrds_pos += 1
        cnt3_wrds += 1

        if cnt3_lis.get((w0,w1,w2)) == None:
            cnt3_lis[(w0,w1,w2)]=1
        else:
            cnt3_lis[(w0,w1,w2)] = cnt3_lis[(w0,w1,w2)]+1

for file in my_files_neg:
    txt = open(file,'r').read()
    words = txt.split()
    new_words = []
    for w in words:
        if len(w) > 5:
            w = snowball.stem(w)
        if w in stopwords:
            continue
        new_words.append(w)
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

    for i in range(len(new_words)-1):
        w0 = new_words[i]
        w1 = new_words[i+1]
        if cnt2_neg_lis.get((w0,w1))== None:
            cnt2_neg_lis[(w0,w1)]=1
        else:
            cnt2_neg_lis[(w0,w1)] = cnt2_neg_lis[(w0,w1)]+1
        
        cnt2_wrds_neg += 1
        cnt2_wrds += 1

        if cnt2_lis.get((w0,w1)) == None:
            cnt2_lis[(w0,w1)]=1
        else:
            cnt2_lis[(w0,w1)] = cnt2_lis[(w0,w1)]+1

    for i in range(len(new_words)-2):
        w0 = new_words[i]
        w1 = new_words[i+1]
        w2 = new_words[i+2]

        if cnt3_neg_lis.get((w0,w1,w2))== None:
            cnt3_neg_lis[(w0,w1,w2)]=1
        else:
            cnt3_neg_lis[(w0,w1,w2)] = cnt3_neg_lis[(w0,w1,w2)]+1
        
        cnt3_wrds_neg += 1
        cnt3_wrds += 1

        if cnt3_lis.get((w0,w1,w2)) == None:
            cnt3_lis[(w0,w1,w2)]=1
        else:
            cnt3_lis[(w0,w1,w2)] = cnt3_lis[(w0,w1,w2)]+1

#accuracy for train data

accu = 0

def calc_accu(files,type,part='a'):
    global cnt_pos_lis,cnt_wrds_neg,cnt_wrds_pos,cnt_pos_lis,cnt_wrds,cnt_lis,cnt_neg_lis,cnt2_wrds_pos,cnt2_wrds_neg,cnt2_pos_lis,cnt2_neg_lis,cnt2_lis,cnt2_wrds,cnt3_wrds_pos,cnt3_wrds_neg,cnt3_pos_lis,cnt3_neg_lis,cnt3_lis,cnt3_wrds,pos_prob,neg_prob

    accu = 0
    for file in files:

        test_file = open(file,'r').read()
        words = test_file.split()

        log_p_pos = 0
        log_p_neg = 0

        new_words = []
        for w in words:
            if len(w) > 5:
                w = snowball.stem(w)
            if w in stopwords:
                continue
            new_words.append(w)
            if cnt_pos_lis.get(w) == None:
                log_p_pos += log((1)/(cnt_wrds_pos + len(cnt_lis)))
            else:
                log_p_pos += log((1+cnt_pos_lis[w])/(cnt_wrds_pos + len(cnt_lis)))
            
            if cnt_neg_lis.get(w) == None:
                log_p_neg += log((1)/(cnt_wrds_neg + len(cnt_lis)))
            else:
                log_p_neg += log((1+cnt_neg_lis[w])/(cnt_wrds_neg + len(cnt_lis)))

        
        for i in range(len(new_words)-1):
            w0 = new_words[i]
            w1 = new_words[i+1]

            if cnt2_pos_lis.get((w0,w1))== None:
                log_p_pos += log((1)/(cnt2_wrds_pos + len(cnt2_lis)))
            else:
                log_p_pos += log((1+cnt2_pos_lis[(w0,w1)])/(cnt2_wrds_pos + len(cnt2_lis)))
            
            if cnt2_neg_lis.get((w0,w1)) == None:
                log_p_neg += log((1)/(cnt2_wrds_neg + len(cnt2_lis)))
            else:
                log_p_neg += log((1+cnt2_neg_lis[(w0,w1)])/(cnt2_wrds_neg + len(cnt2_lis)))

        if part == 'b':
            for i in range(len(new_words)-2):
                w0 = new_words[i]
                w1 = new_words[i+1]
                w2 = new_words[i+2]

                if cnt3_pos_lis.get((w0,w1,w2))== None:
                    log_p_pos += log((1)/(cnt3_wrds_pos + len(cnt3_lis)))
                else:
                    log_p_pos += log((1+cnt3_pos_lis[(w0,w1,w2)])/(cnt3_wrds_pos + len(cnt3_lis)))
                
                if cnt3_neg_lis.get((w0,w1,w2)) == None:
                    log_p_neg += log((1)/(cnt3_wrds_neg + len(cnt3_lis)))
                else:
                    log_p_neg += log((1+cnt3_neg_lis[(w0,w1,w2)])/(cnt3_wrds_neg + len(cnt3_lis)))

        log_p_pos += log(pos_prob)
        log_p_neg += log(neg_prob)

        if (log_p_pos>log_p_neg and type == 'pos') or (log_p_neg>log_p_pos and type == 'neg'):
            accu += 1
    return accu


print("Bigram model : ")

#accuracy of test data

accu = 0

accu += calc_accu(test_pos,'pos') + calc_accu(test_neg,'neg')

accu = accu / (len(test_pos)+len(test_neg))
print("Accuracy of test data : "+ str(accu))

print("Trigram model : ")

#accuracy of test data

accu = 0

accu += calc_accu(test_pos,'pos','b') + calc_accu(test_neg,'neg','b')

accu = accu / (len(test_pos)+len(test_neg))
print("Accuracy of test data : "+ str(accu))

