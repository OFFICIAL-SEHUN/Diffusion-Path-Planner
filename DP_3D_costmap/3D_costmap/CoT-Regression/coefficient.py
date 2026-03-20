import numpy as np

# 1. 논문 그래프에서 읽은 실제 데이터 (x: 경사도, y: CoT)
x = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])
y = np.array([1.75, 1.56, 1.04, 0.93, 0.66, 0.54, 0.53, 0.67, 0.8])

# 2. 다항식 회귀 수행 (4차식)
# np.polyfit(입력값, 결과값, 차수)
coefficients = np.polyfit(x, y, 4)

# 3. 결과 출력
a, b, c, d, e = coefficients
print(f"a = {a:.2e}")
print(f"b = {b:.2e}")
print(f"c = {c:.2e}")
print(f"d = {d:.2e}")
print(f"e = {e:.2f}")