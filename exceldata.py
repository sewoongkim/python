import numpy as np
import pandas as pd
import os

base_dir = 'G:\SampleGit01\python'
excel_file = 'data.xlsx'
excel_dir = os.path.join(base_dir, excel_file)

df_from_excel = pd.read_excel(excel_dir, # write your directory here
    sheet_name = 'F_뽑기' 
    # header = 1 
    # dtype = {'구분': str}
)

    # , 'sales_representative': np.int64, 'sales_amount': float dictionary type
    #names = ['region', 'sales_representative', 'sales_amount'], 
    # index_col = 'id', 
    # na_values = 'NaN', 
    # thousands = ',', 
    # nrows = 10, 
    # comment = '#')

# print(df_from_excel)

df_from_excel.head


class Message:
    def __init__(self,msg):
        self.msg = msg

    def __repr__(self):
        return 'Message: %s' % self.msg


x = Message('I have a secret.')
x

import numpy as np

arr = np.empty((8,4), dtype=np.int64)

for i in range(8):
    arr[i] = i

arr[[4,3,0,6]]

arr[[-3, -5, -7]]
arr

arr = np.arange(32).reshape((8,4))
arr


arr[[1,5,7,2,]][:,[0,3,1,2]]


arr[np.ix_([1,5,7,2],[0,3,1,2])]

arr.T

arr = np.arange(4).reshape((2,2))

arr.T

np.dot(arr.T, arr)

np.dot(arr, arr)

arr = np.arange(16).reshape((2,2,4))
arr

arr.transpose((1,0,2))


arr = np.arange(10)

np.sqrt(arr)

np.exp(arr)


x = np.random.randn(8)
y = np.random.randn(8)

x
y

np.maximum(x,y)

arr = np.random.randn(7) * 5

arr
np.modf(arr)

points = np.arange(-5, 5, 1)

xs, ys = np.meshgrid(points, points)

ys

xs

points

z = (xs ** 2 + ys ** 2 )
z


xarr  = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr  = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

result = [ (x if  c else y) for x,y,c in zip(xarr, yarr, cond)]
result

arr = np.random.randn(4,4)

arr

np.where (arr > 0, 2, -2)


arr = np.random.randn(5,5)

arr.mean()

np.mean(arr)
arr = np.arange(16).reshape((4,4))
arr

arr.mean(axis=1)
arr.mean(0)
arr.mean(1)

arr = np.random.randn(4,4)
(arr > 0).sum()
arr

arr = np.random.randn(4,4)
arr
arr.sort(0)
arr


np.save('some_array',arr)

arr1 = np.load('some_array.npy')
arr1


import random
position = 0
walk = [position]

steps = 1000
for i in range(steps):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)

walk.min()
walk.max()

nsteps = 1000

draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()

draws

walk.min()
walk.max()

(np.abs(walk) >= 10).argmax()

nwalks = 50
nsteps = 10

draws = np.random.randint(0,2, size=(nwalks,nsteps))

steps = np.where(draws > 0, 1, -1)

walks = steps.cumsum(1)

walks.max()

walks.min()

hits30 = (np.abs(walks) >= 5).any(1)
hits30.sum()


from pandas import Series, DataFrame


import numpy as np

import pandas as pd

obj = pd.Series([4,7,-5,3])

obj

obj.values

obj.index 

obj2 = pd.Series([4,7,-5,3], index = ['d','b','a','c'])

obj2
obj2['a']

obj2['d']

obj2 [['c','a','d']]

obj2[obj2 > 0]
np.exp(obj2)

'b' in obj2

'e' in obj2 

sdata = { 'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)

obj3

states = ['California', 'Ohio', 'Oregon', 'Texas']

obj4 = pd.Series(sdata, index=states)

obj4 

pd.isnull(obj4)

pd.notnull(obj4)

obj4.isnull()

obj3 + obj4 

obj4.name = 'population'

obj4.index.name = 'state'

obj3

obj.index = ['Bob','Steve','Jeff','Ryan']

obj


data = { 'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 
          'year':  [2000,2001,2002,2001,2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]
}

frame = pd.DataFrame(data)

frame 

DataFrame(data, columns =['year','state','pop'])

frame2 = pd.DataFrame(data, columns=['year','state','pop','debt'],
index= ['one','two', 'three','four','five'])

frame2.columns
