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

import matplotlib.pyplot as plt
label_dictionary =  {0:'T-shirt/top',1:'Trouser', 2:'Pullover',3: 'Dress', \
    4:'Cost',5:'Sandal', 6:'Shirt', \
    7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

i = 109
plt.imshow(x_train[i].reshape(28,28), cmap='gray')
plt.colorbar()
plt.title('Label {} {}'.format(y_train[i], label_dictionary[y_train[i]]))
plt.show()

for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(x_train[i].reshape(28,28))
    plt.colorbar()
    plt.title('Label {}, {}'.format(y_train[i], label_dictionary[y_train[i]]))

plt.tight_layout()
plt.show()

def AND(x1, x2):
    w1,w2,threshold = 0.2, 0.2, 0.3
    temp = w1 * x1 + w2 * x2
    if temp <= threshold:
        return 0
    elif temp > threshold:
        return 1

def NAND(x1, x2):
    w1,w2,threshold = -0.2, -0.2, -0.3
    temp = w1 * x1 + w2 * x2
    if temp <= threshold:
        return 0
    elif temp > threshold:
        return 1

def OR(x1, x2):
    w1,w2,threshold = 0.3, 0.3, 0.2
    temp = w1 * x1 + w2 * x2
    if temp <= threshold:
        return 0
    elif temp > threshold:
        return 1

def XOR(x1,x2) :
    h1 = NAND(x1,x2)
    h2 = OR(x1,x2)
    y = AND(h1,h2)
    return y


print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))

print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))

print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))

print("XOR")
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))

import numpy as np
import matplotlib.pyplot as plt

def Step(x):
    return np.array(x > 0, dtype = np.int)

x = np.arange(-10.0, 10.0, 0.1)
y = Step(x)

plt.plot(x,y)
plt.grid()
plt.show()

# y2 = Sigmoid(x)

def Softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a

import numpy as np
import matplotlib.pyplot as plt

def Step(x):
    return np.array(x > 0, dtype = np.int64)

def Sigmoid(X):
    return 1/(1 + np.exp(-X))

def ReLU(x):
    return np.maximum(0,x) 

x = np.arange(-10.0, 10.0, 0.1)
y = Step(x)

print(y)
X = np.array([0.1,0.2])

W1 = np.array([[1,2,3],[4,5,6]])
B1 = np.array([1,2,3])

A1 = np.dot(X, W1) + B1
Z1 = Sigmoid(A1)

print(A1)
print(Z1)
print(X.shape, W1.shape, B1.shape, A1.shape)

W2 = np.array([[1,2],[3,4],[5,6]])
B2 = np.array([-1,1])

A2 = np.dot(Z1,W2) + B2
Y = Softmax(A2)

print(A2)
print(Y)
print(Z1.shape, W2.shape, B2.shape, A2.shape)

