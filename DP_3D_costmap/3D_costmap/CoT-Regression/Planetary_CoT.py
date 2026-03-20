import numpy as np
import matplotlib.pyplot as plt

def calculate_paper_cot(slope_deg):
    """
    논문 Fig 4의 빨간색 CoT 곡선을 재현하는 4차 다항식 함수
    """
    # 계수 설정 (앞서 추출한 회귀 계수)
    a = -1.53e-06
    b = 2.07e-05
    c = 2.20e-03
    d = -3.24e-02
    e = 0.65
    
    # 4차 다항식 계산
    cot = (a * slope_deg**4) + (b * slope_deg**3) + (c * slope_deg**2) + (d * slope_deg) + e
    return cot

# 1. 그래프를 그릴 데이터 생성 (-20도부터 20도까지 400개의 점)
x_range = np.linspace(-20, 20, 400)
y_cot = calculate_paper_cot(x_range)

# 2. 논문에 표시된 실제 데이터 포인트 (검증용 삼각형 마커)
measured_x = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])
measured_y = np.array([1.75, 1.56, 1.04, 0.93, 0.66, 0.54, 0.53, 0.67, 0.8])

# 3. 그래프 그리기
plt.figure(figsize=(10, 6))

# 회귀 곡선 (빨간색 실선)
plt.plot(x_range, y_cot, color='red', linewidth=2, label='Polynomial Fit (CoT)')

# 실제 데이터 포인트 (빨간색 삼각형)
plt.scatter(measured_x, measured_y, color='red', marker='^', s=80, label='Measured Data (Paper)')

# 그래프 스타일 설정 (논문과 유사하게)
plt.title('Reconstruction of CoT vs Inclination Angle', fontsize=14)
plt.xlabel('Inclination (deg)', fontsize=12)
plt.ylabel('Value (CoT)', fontsize=12)
plt.xticks(np.arange(-20, 21, 5)) # x축 간격을 5도로 설정
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 최저점(에너지 효율이 가장 좋은 지점) 표시
min_idx = np.argmin(y_cot)
plt.annotate(f'Min CoT: {y_cot[min_idx]:.2f} at {x_range[min_idx]:.1f}°', 
             xy=(x_range[min_idx], y_cot[min_idx]), 
             xytext=(x_range[min_idx]+2, y_cot[min_idx]+0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.show()