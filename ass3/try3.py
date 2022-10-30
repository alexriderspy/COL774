import matplotlib.pyplot as plt

output_path = '.'

train_accuracies_std = [4,6,9,20,25,30,40,60]
train_accuracies_ccp = [12,20,40,60,90,90,300,600]
train_accuracies_rf = [30,50,90,200,300,600,800,1300]
train_accuracies_xgb = [6,10,20,30,35,35,40,60]
train_accuracies_lgb = [2,4,6,9,10,12,16,30]

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_std)
plt.savefig(output_path + '/std_time.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_ccp)
plt.savefig(output_path + '/ccp_time.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_rf)
plt.savefig(output_path + '/rf_time.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_xgb)
plt.savefig(output_path + '/xgb_time.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_lgb)
plt.savefig(output_path + '/lgb_time.png')
