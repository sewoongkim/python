import numpy as np
import matplotlib.pyplot as plt
def Step(x):
    return np.array(x>0,dtype=np.int)       # 조건에 따라 참/거짓, 1/0의 값을 반환

#  ReLU함수를 이용하여 출력하기

def ReLU(x):
  return np.maximum(0,x)

x=np.arange(-5.0,5.0,0.1)
y=ReLU(x)
plt.plot(x,y)
plt.grid()
plt.show()

