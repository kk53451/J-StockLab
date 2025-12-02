"""
J-StockLab: EDA 상관관계 히트맵 생성

경제 지표(FRED + yfinance)와 Nikkei 225 상위 20개 종목 간의
상관관계를 시각화하여 Dual Input Stream 아키텍처의 타당성을 보여줍니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'total.csv')
data = pd.read_csv(data_path, parse_dates=['날짜'])

# 컬럼 분류
economic_indicators = [
    # 일본 경제 지표 (8개)
    '일본 실질 GDP', '일본 실업률', '일본 10년 국채 수익률',
    '일본 3개월 은행간 금리', '일본 총산업생산', '일본 무역수지',
    '일본 소비자 신뢰지수', '일본은행 총자산',
    # 미국 경제 지표 (10개)
    '미국 10년 기대 인플레이션율', '미국 장단기 금리차', '미국 기준금리',
    '미국 2년 만기 국채 수익률', '미국 10년 만기 국채 수익률',
    '미시간대 소비자 심리지수', '미국 실업률', '미국 소비자 물가지수',
    '미국 GDP 성장률', '미국 금융스트레스지수',
    # 시장 지표 (9개)
    '닛케이 225', '닛케이 300', 'TOPIX ETF',
    'S&P 500 지수', '나스닥 종합지수',
    'VIX 지수', '금 가격', '달러 인덱스', '엔/달러 환율'
]

stock_columns = [
    'Toyota', 'SoftBank Group', 'Mitsubishi UFJ Financial', 'Sony Group',
    'Hitachi', 'Fast Retailing', 'SMFG', 'Nintendo',
    'Tokyo Electron', 'Advantest', 'Mitsubishi Heavy Ind', 'Mitsubishi Corp',
    'Keyence', 'Chugai Pharma', 'ITOCHU', 'Mizuho Financial',
    'NTT', 'Mitsui & Co', 'Recruit Holdings', 'Tokio Marine'
]

# 존재하는 컬럼만 필터링
economic_indicators = [col for col in economic_indicators if col in data.columns]
stock_columns = [col for col in stock_columns if col in data.columns]

print(f"경제 지표: {len(economic_indicators)}개")
print(f"주식 종목: {len(stock_columns)}개")

# 상관관계 계산 (경제 지표 vs 주식)
correlation_matrix = data[economic_indicators + stock_columns].corr()

# 경제 지표와 주식 간의 상관관계만 추출
cross_correlation = correlation_matrix.loc[economic_indicators, stock_columns]

# 히트맵 생성
fig, ax = plt.subplots(figsize=(16, 12))

# 색상 맵 설정
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# 히트맵 그리기
heatmap = sns.heatmap(
    cross_correlation,
    annot=False,  # 값 표시 안 함 (너무 많아서)
    cmap=cmap,
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
    ax=ax
)

# 제목 및 레이블
ax.set_title('경제 지표와 Nikkei 225 상위 20개 종목 간의 상관관계',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('주식 종목 (Nikkei 225 Top 20)', fontsize=11, labelpad=10)
ax.set_ylabel('경제 지표 (FRED + yfinance)', fontsize=11, labelpad=10)

# x축 레이블 회전
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)

# 구분선 추가 (일본 지표 / 미국 지표 / 시장 지표)
ax.axhline(y=8, color='black', linewidth=2)   # 일본 지표 아래
ax.axhline(y=18, color='black', linewidth=2)  # 미국 지표 아래

# 그룹 레이블 추가 (왼쪽)
ax.text(-2.5, 4, '일본\n경제지표', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(-2.5, 13, '미국\n경제지표', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(-2.5, 22, '시장\n지표', ha='center', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()

# 저장
output_path = os.path.join(script_dir, 'correlation_heatmap.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n히트맵 저장 완료: {output_path}")

# 주요 상관관계 통계 출력
print("\n" + "=" * 60)
print("주요 상관관계 분석")
print("=" * 60)

# 각 주식별 가장 높은 상관관계를 가진 경제 지표
print("\n[각 종목별 가장 높은 상관관계 경제 지표]")
for stock in stock_columns[:5]:  # 대표 5개만
    correlations = cross_correlation[stock].abs().sort_values(ascending=False)
    top_indicator = correlations.index[0]
    top_value = cross_correlation.loc[top_indicator, stock]
    print(f"  {stock}: {top_indicator} (r={top_value:.3f})")

# 평균 상관관계
print(f"\n[전체 평균 절대 상관관계]: {cross_correlation.abs().values.mean():.3f}")
print(f"[최대 상관관계]: {cross_correlation.values.max():.3f}")
print(f"[최소 상관관계]: {cross_correlation.values.min():.3f}")

plt.show()
