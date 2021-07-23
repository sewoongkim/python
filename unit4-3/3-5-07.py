# NAND 연산 퍼셉트론 분류하기

# NAND 연산 함수
def NAND(x1,x2):
  #파라미터 값(w1,w2,임계값),AND의 역이므로 AND 가중치에 -1을 곱함
  w1,w2,threshold = -0.2,-0.2,-0.3
  temp = w1*x1+w2*x2
  if temp <= threshold:
    return 0
  elif temp > threshold:
    return 1

print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))
