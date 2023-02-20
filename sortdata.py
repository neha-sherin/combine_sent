import os
import re
import numpy as np

def sorttxt(filep):
    with open(filep, 'r') as f:
        data = f.readlines()

    print('data len', len(data))
    print('d0', data[0].split('|')[0])
    data = sorted(data)
#data.sort(key = lambda x : '-'.join( x.split('|')[0].split('-')[:2] ) )
#data.sort(key = lambda x : list( map(int, re.findall(r'\d+', x.split('|')[0])))[0])
#data.sort(key = lambda x : list( map(int, re.findall(r'\d+', x.split('|')[0])))[-1])

    with open(filep, 'w') as f:
        f.writelines(data)

if __name__=="__main__":
    filep = '/ssd_scratch/cvit/neha/preprocessed_data/train.txt'
    sorttxt(filep)
    filep = '/ssd_scratch/cvit/neha/preprocessed_data/val.txt'
    sorttxt(filep)
