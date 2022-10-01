import random
import sys
import os

#positive data

test_path = str(sys.argv[2])

test_pos = []
test_neg = []

for (dirpath, dirnames, filenames) in os.walk(os.join(test_path,'/pos')):
    for filename in filenames:
        test_pos.append(os.join(test_path,'/pos/' ,filename))
    break

for (dirpath, dirnames, filenames) in os.walk(os.join(test_path, '/neg')):
    for filename in filenames:
        test_neg.append(os.join(test_path,'/neg/' ,filename))
    break

accu = 0

def calc_accu_rand(files,type):
    accu = 0
    for file in files:
        num = random.randint(0,1)
        if (num == 0 and type == 'neg') or (num == 1 and type == 'pos'):
            accu+=1
    return accu

#accuracy of test data

accu = 0

accu += calc_accu_rand(test_pos,'pos') + calc_accu_rand(test_neg,'neg')

accu = accu / (len(test_pos)+len(test_neg))
print("Accuracy of test data (random): "+ str(accu))

accu = len(test_pos)/(len(test_pos)+len(test_neg))
print("Accuracy of test data (all pos): "+ str(accu))