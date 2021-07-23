# 데이터 불러오기
import pandas as pd

df_train = pd.read_csv('d:/PythonCode/fashion-mnist_train.csv') #훈련 데이터 파일 열기
df_test = pd.read_csv('d:/PythonCode/fashion-mnist_test.csv')   #테스트 데이터 파일 열기

print(df_train.info(),'\n')
print(df_test.info(),'\n')
print(df_train.shape,'\n')
print(df_test.shape,'\n')

import numpy as np

data_train = np.array(df_train, dtype=np.float32)

x_train = data_train[:, 1:]
y_train = data_train[:, 0]

data_test = np.array(df_test, dtype=np.float32)
x_test = data_test[:, 1:]
y_test = data_test[:, 0]

# print(df_test)
# print(data_test)

