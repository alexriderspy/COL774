import matplotlib.pyplot as plt

x = [2,3,4,5]
train = [0.94,0.93,0.89,0.80]
test = [0.86,0.85,0.83,0.82]

plt.figure()
plt.plot(x,train)
plt.xlabel('hidden layers')
plt.ylabel('train_accuracies')
plt.savefig('e3_sigmoid.png')

plt.figure()
plt.plot(x,test)
plt.xlabel('hidden layers')
plt.ylabel('test_accuracies')
plt.savefig('e4_sigmoid.png')