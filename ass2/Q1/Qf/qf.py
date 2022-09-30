import sys
sys.path.append('../Qe')
import qe

pos_accu = qe.calc_accu(qe.test_pos,'pos')
neg_accu = qe.calc_accu(qe.test_neg,'neg')

tp = pos_accu
tn = neg_accu
fp = len(qe.test_neg)-neg_accu
fn = len(qe.test_pos)-pos_accu

recall = tp/(tp+fn)
print("Recall : " + str(recall))

precision = tp/(tp+fp)
print("Precision : " + str(precision))

f1_score = 2/((1/precision) + (1/recall))
print("F1 score : " + str(f1_score))
