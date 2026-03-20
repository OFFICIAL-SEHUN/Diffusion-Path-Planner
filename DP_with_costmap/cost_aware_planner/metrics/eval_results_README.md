# eval_results.csv 설명

## 컬럼

| 컬럼 | 의미 |
|------|------|
| **scale** | Cost Guidance Scale. 샘플링 시 costmap gradient로 경로를 벽에서 밀어내는 세기. 0에 가까우면 가이던스 약함, 크면 장애물 회피 강함. |
| **success_rate_pct** | 성공률 (%). 경로가 collision threshold(기본 0.9)를 넘는 셀을 **한 번도 지나지 않으면** 성공. |
| **mean_cost_pct** | **비율**: Diffusion 경로 평균 비용 ÷ A* 경로 평균 비용 × 100. |

---

## mean_cost_pct (비율) 상세

- **기준**: A* 경로 = GT = **100%**.
- **계산**: 같은 costmap 위에서
  - A* 경로가 지나는 격자점들의 costmap 값 평균 → `A*_mean_cost`
  - Diffusion 경로가 지나는 격자점들의 costmap 값 평균 → `Diffusion_mean_cost`  
  → `mean_cost_pct = (Diffusion_mean_cost / A*_mean_cost) × 100`

**해석**

- **100%**: Diffusion 경로가 A*와 비슷한 비용(같은 정도로 저비용 통로)을 탔다.
- **> 100%**: Diffusion 경로가 A*보다 **비싼** 길을 탔다 (벽 근처·고비용 구간을 더 지남). 숫자가 클수록 A* 대비 더 비쌈.
- **< 100%**: Diffusion 경로가 A*보다 더 저비용 구간을 지났다 (이론상 가능하나, A*가 이미 최적에 가깝다면 100% 근처가 보통).

---

## 현재 CSV 요약 (예시)

```
scale   success_rate_pct   mean_cost_pct
0.01    83%                274.70%
0.015   80%                233.79%
0.02    80%                212.54%
0.025   76%                210.99%
```

- **Success rate**: scale이 커질수록(0.01→0.025) 성공률이 83%→76%로 약간 감소.
- **Mean cost %**: 모든 scale에서 **200%대** → Diffusion 경로가 A* 대비 평균 비용이 약 **2배 이상** 높다. 즉, A*(GT)보다 훨씬 비싼 경로를 탄다.
- scale을 키우면(가이던스 강화) mean_cost_pct가 274%→211%로 **감소**하는 경향: 벽을 더 피해서 비용이 A*에 조금 더 가까워짐.

**정리**: `mean_cost_pct`는 **“A*를 100%로 봤을 때, Diffusion 경로의 평균 비용이 몇 %인가”**를 나타내는 지표다. 100%에 가까울수록 A*와 비슷한 저비용 경로, 200%면 A*보다 두 배 비싼 경로라고 보면 된다.
