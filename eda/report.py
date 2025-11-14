"""
J-StockLab: 일본 주식 예측 - 평가 리포트 및 매수/매도 추천

프로젝트: Nikkei 225 상위 20개 종목의 7일 후 주가 예측 평가
- 평가 지표: MAE, MSE, RMSE, MAPE, Accuracy
- 상승/하락 분석 및 매수/매도 추천
- 파일 직접 업로드 방식 (Google Drive 불필요)

작성자: 최정민, 김종수, 김용균
"""

# 파일 직접 업로드 방식 (Google Drive 불필요)
from google.colab import files
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

######################
# (1) Evaluation Function
######################
def evaluate_predictions(data, target_columns, forecast_horizon):
    """
    실제값과 예측값을 비교하여 다양한 평가 지표를 계산합니다.

    - MAE (Mean Absolute Error): 평균 절대 오차
      (낮을수록 좋음, 원본 데이터와 동일한 단위)
    - MSE (Mean Squared Error): 평균 제곱 오차
      (낮을수록 좋음)
    - RMSE (Root Mean Squared Error): MSE의 제곱근
      (낮을수록 좋음, MAE와 함께 자주 사용됨)
    - MAPE (Mean Absolute Percentage Error): 평균 절대 백분율 오차
      (낮을수록 좋음, 백분율로 표현)
    - Accuracy (%): 정확도 (100 - MAPE)
    """

    metrics = []

    for col in target_columns:
        predicted_col = f'{col}_Predicted'
        actual_col = f'{col}_Actual'

        # 컬럼 존재 여부 확인
        if predicted_col not in data.columns or actual_col not in data.columns:
            print(f"Skipping {col}: Columns not found in data")
            continue

        # 예측값 및 실제값 추출
        predicted = data[predicted_col]
        # 7일 후의 실제값과 비교하기 위해 shift 적용
        actual = data[actual_col].shift(-forecast_horizon)

        # 유효한 데이터만 사용 (NaN 제거)
        valid_idx = ~predicted.isna() & ~actual.isna()
        predicted = predicted[valid_idx]
        actual = actual[valid_idx]

        if len(predicted) == 0:
            print(f"Skipping {col}: No valid prediction/actual pairs.")
            continue

        # 평가 지표 계산
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = mse ** 0.5
        mape = (abs((actual - predicted) / actual).mean()) * 100
        accuracy = 100 - mape

        metrics.append({
            'Stock': col,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'Accuracy (%)': accuracy
        })

    return pd.DataFrame(metrics)

###############################
# (2) Future Rise Analysis
###############################
def analyze_rise_predictions(data, target_columns):
    """
    가장 최근 데이터 기준으로 7일 후 상승/하락 예측 분석
    - 마지막 행의 실제 주가와 예측 주가를 비교
    - 상승 확률(%) 계산
    """

    last_row = data.iloc[-1]
    results = []

    for col in target_columns:
        last_actual_price = last_row.get(f'{col}_Actual', np.nan)
        predicted_future_price = last_row.get(f'{col}_Predicted', np.nan)

        # 상승/하락 및 상승 확률 계산
        if pd.notna(last_actual_price) and pd.notna(predicted_future_price):
            predicted_rise = predicted_future_price > last_actual_price
            rise_probability = ((predicted_future_price - last_actual_price) / last_actual_price) * 100
        else:
            predicted_rise = np.nan
            rise_probability = np.nan

        results.append({
            'Stock': col,
            'Last Actual Price': last_actual_price,
            'Predicted Future Price': predicted_future_price,
            'Predicted Rise': predicted_rise,
            'Rise Probability (%)': rise_probability
        })

    return pd.DataFrame(results)

#######################################
# (3) Buy/Sell Recommendation and Analysis
#######################################
def generate_recommendation(row):
    """
    매수/매도 추천 로직:
    - 상승 예측 & 상승률 > 0% => BUY
    - 상승률 > 2% => STRONG BUY
    - 그 외 => SELL
    """
    rise_prob = row.get('Rise Probability (%)', 0)
    predicted_rise = row.get('Predicted Rise', False)

    if pd.isna(rise_prob) or pd.isna(predicted_rise):
        return "No Data"

    if predicted_rise and rise_prob > 0:
        if rise_prob > 2:
            return "STRONG BUY"
        else:
            return "BUY"
    else:
        return "SELL"

def generate_analysis(row):
    """
    각 종목에 대한 간단한 분석 코멘트 생성
    """
    stock_name = row['Stock']
    rise_prob = row.get('Rise Probability (%)', 0)
    predicted_rise = row.get('Predicted Rise', False)

    if pd.isna(rise_prob) or pd.isna(predicted_rise):
        return f"{stock_name}: Not enough data"

    if predicted_rise:
        return f"{stock_name} is expected to rise by about {rise_prob:.2f}%. Consider buying or holding."
    else:
        return f"{stock_name} is expected to fall by about {-rise_prob:.2f}%. A cautious approach is recommended."

#######################
# (4) Main Code
#######################
print("=" * 80)
print("J-StockLab: Nikkei 225 Stock Prediction Evaluation")
print("=" * 80)

print("\n📁 Please upload 'predicted_stock.csv' file...")
uploaded = files.upload()

print("\nLoading predicted stock data...")
file_path = list(uploaded.keys())[0]  # 업로드된 파일명 자동 인식
data = pd.read_csv(file_path, parse_dates=['날짜'])
print(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")
print(f"Date range: {data['날짜'].min()} ~ {data['날짜'].max()}")

# Nikkei 225 상위 20개 종목 (영문명)
target_columns = [
    'Toyota', 'SoftBank Group', 'Mitsubishi UFJ Financial', 'Sony Group',
    'Hitachi', 'Fast Retailing', 'SMFG', 'Nintendo',
    'Tokyo Electron', 'Advantest', 'Mitsubishi Heavy Ind', 'Mitsubishi Corp',
    'Keyence', 'Chugai Pharma', 'ITOCHU', 'Mizuho Financial',
    'NTT', 'Mitsui & Co', 'Recruit Holdings', 'Tokio Marine'
]

forecast_horizon = 7  # 7일 후 예측

print(f"\nTarget stocks: {len(target_columns)} stocks")
print(f"Forecast horizon: {forecast_horizon} days\n")

# 1) 모델 평가
print("=" * 80)
print("Step 1: Evaluating prediction accuracy...")
print("=" * 80)
evaluation_results = evaluate_predictions(data, target_columns, forecast_horizon)
print("\n============ Evaluation Results ============")
print(evaluation_results.to_string(index=False))

# 2) 미래 상승/하락 분석
print("\n" + "=" * 80)
print("Step 2: Analyzing future price movements...")
print("=" * 80)
rise_results = analyze_rise_predictions(data, target_columns)
print("\n============ Rise Predictions ============")
print(rise_results.to_string(index=False))

# 3) 데이터프레임 병합 (평가 지표 + 상승 분석)
print("\n" + "=" * 80)
print("Step 3: Generating recommendations...")
print("=" * 80)
final_results = pd.merge(evaluation_results, rise_results, on='Stock', how='outer')

# 4) 상승 확률 기준 내림차순 정렬
final_results = final_results.sort_values(by='Rise Probability (%)', ascending=False)

# 5) 매수/매도 추천 및 분석 생성
final_results['Recommendation'] = final_results.apply(generate_recommendation, axis=1)
final_results['Analysis'] = final_results.apply(generate_analysis, axis=1)

# 컬럼 순서 재정렬
column_order = [
    'Stock',
    'MAE', 'MSE', 'RMSE', 'MAPE (%)', 'Accuracy (%)',
    'Last Actual Price', 'Predicted Future Price', 'Predicted Rise', 'Rise Probability (%)',
    'Recommendation', 'Analysis'
]
final_results = final_results[column_order]

# 6) CSV 파일로 저장
output_file_path = 'final_stock_analysis.csv'
final_results.to_csv(output_file_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Final analysis saved to: {output_file_path}")
print(f"   - Total stocks analyzed: {len(final_results)} stocks")
print(f"   - Columns: {len(final_results.columns)}")

# 7) 결과 파일 다운로드
print("\n📥 Downloading final_stock_analysis.csv...")
files.download(output_file_path)

# 8) 최종 리포트 출력
print("\n" + "=" * 80)
print("=============== Final Report ===============")
print("=" * 80)
print(final_results.to_string(index=False))

# 9) 요약 통계
print("\n" + "=" * 80)
print("=============== Summary Statistics ===============")
print("=" * 80)
print(f"\nTotal stocks analyzed: {len(final_results)}")
print(f"Average Accuracy: {final_results['Accuracy (%)'].mean():.2f}%")
print(f"Average MAPE: {final_results['MAPE (%)'].mean():.2f}%")
print(f"\nRecommendation Distribution:")
print(final_results['Recommendation'].value_counts())
print(f"\nTop 5 stocks by Rise Probability:")
print(final_results[['Stock', 'Rise Probability (%)', 'Recommendation']].head(5).to_string(index=False))

print("\n" + "=" * 80)
print("✅ All done! Check the downloaded 'final_stock_analysis.csv' file.")
print("=" * 80)
