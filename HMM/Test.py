#%%
import multiprocessing as mp
import concurrent.futures

from py import process




def add(a):
    #print(a+b)
    return a

inputs = [2,2,3,4]

res = map(add,inputs)

for f in res:
    print(f)
