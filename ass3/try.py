import matplotlib.pyplot as plt

output_path = '.'

train_accuracies_grid = [0.95,0.96,0.96,0.96,0.97,0.97,0.97,0.971]
train_accuracies_ccp = [0.94,0.95,0.95,0.96,0.97,0.97,0.97,0.98]
train_accuracies_rf = [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97]
train_accuracies_xgb = [0.96,0.96,0.97,0.97,0.97,0.97,0.967,0.98]
train_accuracies_lgb = [0.96,0.96,0.96,0.96,0.97,0.97,0.98,0.98]

test_accuracies_grid = [0.4,0.46,0.46,0.50,0.52,0.56,0.55,0.555]
test_accuracies_ccp = [0.38,0.47,0.48,0.50,0.53,0.57,0.55,0.567]
test_accuracies_rf = [0.4,0.48,0.49,0.50,0.52,0.56,0.55,0.556]
test_accuracies_xgb = [0.42,0.47,0.48,0.50,0.52,0.57,0.56,0.567]
test_accuracies_lgb = [0.49,0.56,0.57,0.59,0.61,0.635,0.61,0.63]

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_grid)
plt.savefig(output_path + '/grid_train.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_ccp)
plt.savefig(output_path + '/ccp_train.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_rf)
plt.savefig(output_path + '/rf_train.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_xgb)
plt.savefig(output_path + '/xgb_train.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_lgb)
plt.savefig(output_path + '/lgb_train.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_grid)
plt.savefig(output_path + '/grid_test.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_ccp)
plt.savefig(output_path + '/ccp_test.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_rf)
plt.savefig(output_path + '/rf_test.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_xgb)
plt.savefig(output_path + '/xgb_test.png')

plt.figure()
plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_lgb)
plt.savefig(output_path + '/lgb_test.png')
