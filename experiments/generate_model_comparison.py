"""
J-StockLab: 모델 성능 비교 시각화
3개 모델(LSTM, Transformer, Linear Regression)의 종목별 성능 비교
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
lstm = pd.read_csv('final_stock_analysis_LSTM.csv')
tf = pd.read_csv('final_stock_analysis_TF.csv')
lr = pd.read_csv('final_stock_analysis_LR.csv')

# 종목 순서 통일 (LSTM 기준 Accuracy 내림차순)
lstm_sorted = lstm.sort_values('Accuracy_Day7 (%)', ascending=True)
stock_order = lstm_sorted['Stock'].tolist()

# 다른 모델도 같은 순서로 정렬
tf_sorted = tf.set_index('Stock').loc[stock_order].reset_index()
lr_sorted = lr.set_index('Stock').loc[stock_order].reset_index()

# Figure 생성 (2개 subplot)
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# ===== 그래프 1: 종목별 Accuracy 비교 (수평 막대) =====
ax1 = axes[0]
y_pos = np.arange(len(stock_order))
bar_height = 0.25

# LSTM (초록색)
bars1 = ax1.barh(y_pos - bar_height, lstm_sorted['Accuracy_Day7 (%)'],
                  bar_height, label='LSTM', color='#2ecc71', alpha=0.9)
# Transformer (파란색)
bars2 = ax1.barh(y_pos, tf_sorted['Accuracy_Day7 (%)'],
                  bar_height, label='Transformer', color='#3498db', alpha=0.9)
# Linear Regression (빨간색, 투명도 높여서 과적합 표시)
bars3 = ax1.barh(y_pos + bar_height, lr_sorted['Accuracy_Day7 (%)'],
                  bar_height, label='Linear Regression (Overfitting)', color='#e74c3c', alpha=0.4)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(stock_order, fontsize=10)
ax1.set_xlabel('Accuracy (%)', fontsize=12)
ax1.set_title('종목별 Day7 Accuracy 비교', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim(50, 105)
ax1.axvline(x=94.01, color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axvline(x=91.36, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.text(94.5, 19.5, 'LSTM 평균: 94.01%', fontsize=9, color='#2ecc71')
ax1.text(88, 18.5, 'TF 평균: 91.36%', fontsize=9, color='#3498db')

# ===== 그래프 2: 모델별 종합 성능 요약 =====
ax2 = axes[1]

# 데이터 준비
models = ['LSTM', 'Transformer', 'Linear Regression\n(Overfitting)']
accuracies = [94.01, 91.36, 100.00]
mapes = [5.99, 8.64, 0.00]
stds = [2.85, 8.08, 0.00]
reliabilities = [0.9, 0.6, 0.1]  # 신뢰도 (0~1)

x = np.arange(len(models))
width = 0.35

# Accuracy 막대 (왼쪽 y축) - 개별적으로 그리기
color_acc = ['#2ecc71', '#3498db', '#e74c3c']
alpha_acc = [0.9, 0.9, 0.4]
bars = []
for i, (acc, col, alp) in enumerate(zip(accuracies, color_acc, alpha_acc)):
    bar = ax2.bar(x[i], acc, width, color=col, alpha=alp)
    bars.append(bar[0])

# 신뢰도 표시 (막대 위에 텍스트)
reliability_labels = ['High', 'Medium', 'Low']
for i, (bar, rel_label) in enumerate(zip(bars, reliability_labels)):
    height = bar.get_height()
    ax2.annotate(f'Reliability: {rel_label}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('모델별 종합 성능 비교', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11)
ax2.set_ylim(0, 115)

# MAPE를 오른쪽 y축에 표시
ax2_twin = ax2.twinx()
ax2_twin.plot(x, mapes, 'o-', color='#9b59b6', linewidth=2, markersize=10, label='MAPE (%)')
ax2_twin.set_ylabel('MAPE (%)', fontsize=12, color='#9b59b6')
ax2_twin.tick_params(axis='y', labelcolor='#9b59b6')
ax2_twin.set_ylim(0, 12)

# 범례
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

# 주석 추가
ax2.annotate('Higher is not always better!\n(과적합 주의)',
             xy=(2, 100), xytext=(1.5, 85),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red', ha='center')

plt.tight_layout()

# 저장
output_path = 'experiments/model_performance_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"이미지 저장 완료: {output_path}")

plt.show()
