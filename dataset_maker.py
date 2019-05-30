from doubledummy import DDS
import numpy as np
import os

num = 10 ** 5 

path = 'dataset{}'.format(num)
p = path
i = 1
while os.path.exists(p+'.csv'):
    p = path + '_{}'.format(i)
    i += 1
path = p + '.csv'

with open(path, 'a') as f:
    for i in range(num):
        hands, tricks = DDS()
        np.savetxt(f, np.concatenate((hands, tricks), axis=None).reshape(1,-1), fmt='%2.f')
        print(i)