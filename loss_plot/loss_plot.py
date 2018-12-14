import numpy as np
import matplotlib.pyplot as plt

def mov_ave(data):
    data2 = data[:]
    for i in range(1, len(data)-3):
        data2[i] = sum(data[i:i+3])/3.0
    return data2

adam_name = "loss_adam.txt"
augm_name = "loss_augment.txt"
orig_name = "loss_orign.txt"

f_orig = open(orig_name,"r")
f_adam = open(adam_name,"r")

line = f_orig.readline().split()
adam = f_adam.readline().split()


line = list(map(float,line))
adam = list(map(float,adam))

line = mov_ave(line)
adam = mov_ave(adam)

print(adam[0:10])
plt.semilogx(line,label='SGD')
plt.semilogx(adam,label='Adam')

legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.ylabel('Loss')
plt.xlabel('Batches')
plt.title('Loss vs Batches from SGD and Adam in the first epoch')
plt.show()