import numpy as np

# 1. 논문 그래프에서 읽은 실제 데이터 (x: 경사도, y: CoT)
x = np.array([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20])
y = np.array([0.7400, 0.6803, 0.5805, 0.4921, 0.4201, 0.4324, 0.4848, 0.7792, 0.7246, 1.3935])

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