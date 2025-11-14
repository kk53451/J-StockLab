import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# FRED API Key 설정
api_key = os.getenv('FRED_API_KEY')
if not api_key:
    raise ValueError("FRED_API_KEY not found in .env file. Please check your .env file.")

# FRED에서 제공하는 지표 코드와 명칭
fred_indicators = {
    # ===========================
    # 일본 FRED 지표 (8개)
    # ===========================
    'JPNRGDPEXP': '일본 실질 GDP',                    # 분기
    'LRUN64TTJPM156S': '일본 실업률',                 # 월간
    'IRLTLT01JPM156N': '일본 10년 국채 수익률',        # 월간
    'IR3TIB01JPM156N': '일본 3개월 은행간 금리',       # 월간
    'JPNPRINTO01GYSAM': '일본 총산업생산',            # 월간
    'XTNTVA01JPM664S': '일본 무역수지',               # 월간
    'CSCICP02JPM460S': '일본 소비자 신뢰지수',         # 월간
    'JPNASSETS': '일본은행 총자산',                    # 월간

    # ===========================
    # 미국 FRED 지표 (10개)
    # ===========================
    # 금리 관련 (5개)
    'T10YIE': '미국 10년 기대 인플레이션율',            # 일간
    'T10Y2Y': '미국 장단기 금리차',                    # 일간
    'FEDFUNDS': '미국 기준금리',                       # 월간
    'DGS2': '미국 2년 만기 국채 수익률',               # 일간
    'DGS10': '미국 10년 만기 국채 수익률',             # 일간

    # 경제 지표 (4개)
    'UMCSENT': '미시간대 소비자 심리지수',             # 월간
    'UNRATE': '미국 실업률',                          # 월간
    'CPIAUCSL': '미국 소비자 물가지수',               # 월간
    'GDPC1': '미국 GDP 성장률',                       # 분기

    # 금융 시장 (1개)
    'STLFSI4': '미국 금융스트레스지수',               # 주간
}


# Yahoo Finance에서 제공하는 지표와 티커
yfinance_indicators = {
    # === 일본 주요 지수 (3개) ===
    '닛케이 225': '^N225',           # Nikkei 225 지수
    '닛케이 300': '^N300',           # Nikkei 300 지수
    'TOPIX ETF': '1306.T',           # TOPIX 지수 추종 ETF

    # === 미국 주요 지수 (2개) ===
    'S&P 500 지수': '^GSPC',         # S&P 500 지수
    '나스닥 종합지수': '^IXIC',      # 나스닥 종합지수

    # === 시장 심리 & 상품 (4개) ===
    'VIX 지수': '^VIX',              # 변동성 지수 (공포 지수)
    '금 가격': 'GC=F',               # 금 가격 (선물)
    '달러 인덱스': 'DX-Y.NYB',       # 달러 인덱스
    '엔/달러 환율': 'JPY=X',         # 일본 엔 대 미국 달러 환율
}

# Nikkei 225 상위 20개 종목 티커 리스트와 한글 이름
nikkei_top_20 = [
    ("7203.T", "토요타"),                      # 1. Toyota Motor
    ("9984.T", "소프트뱅크그룹"),               # 2. SoftBank Group
    ("8306.T", "미쓰비시UFJ파이낸셜그룹"),      # 3. Mitsubishi UFJ Financial
    ("6758.T", "소니그룹"),                     # 4. Sony Group
    ("6501.T", "히타치제작소"),                 # 5. Hitachi
    ("9983.T", "패스트리테일링"),               # 6. Fast Retailing (Uniqlo)
    ("8316.T", "미쓰이스미토모파이낸셜그룹"),   # 7. Sumitomo Mitsui Financial
    ("7974.T", "닌텐도"),                      # 8. Nintendo
    ("8035.T", "도쿄일렉트론"),                 # 9. Tokyo Electron
    ("6857.T", "어드반테스트"),                 # 10. Advantest
    ("7011.T", "미쓰비시중공업"),               # 11. Mitsubishi Heavy Industries
    ("8058.T", "미쓰비시상사"),                 # 12. Mitsubishi Corporation
    ("6861.T", "키엔스"),                      # 13. Keyence
    ("4519.T", "주가이제약"),                   # 14. Chugai Pharmaceutical
    ("8001.T", "이토추"),                      # 15. ITOCHU
    ("8411.T", "미즈호파이낸셜그룹"),           # 16. Mizuho Financial
    ("9432.T", "일본전신전화"),                 # 17. NTT (Nippon Telegraph)
    ("8031.T", "미쓰이물산"),                   # 18. Mitsui & Co
    ("6098.T", "리크루트홀딩스"),               # 19. Recruit Holdings
    ("8766.T", "도쿄해상홀딩스"),               # 20. Tokio Marine Holdings
]

# 데이터 수집 기간 설정
start_date = '2014-10-16'  # 리크루트홀딩스 상장일 (Nikkei 225 상위 20개 종목 중 가장 늦은 상장일)
end_date = datetime.today().strftime('%Y-%m-%d')

# FRED API를 통한 데이터 수집
fred_data_frames = []
for code, name in fred_indicators.items():
    # 지표별 제공 주기에 따른 요청 주기 설정
    if code in ['FEDFUNDS', 'UMCSENT', 'UNRATE', 'CPIAUCSL',
                'LRUN64TTJPM156S', 'IRLTLT01JPM156N', 'IR3TIB01JPM156N',
                'JPNPRINTO01GYSAM', 'XTNTVA01JPM664S', 'CSCICP02JPM460S', 'JPNASSETS']:
        frequency = 'm'  # 월간
    elif code in ['STLFSI4']:
        frequency = 'w'  # 주간
    elif code in ['JPNRGDPEXP', 'GDPC1']:
        frequency = 'q'  # 분기
    else:
        frequency = 'd'  # 일간

    url = f'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': code,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
        'frequency': frequency  # 동적으로 설정된 주기 반영
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json().get('observations', [])
        if data:
            df = pd.DataFrame(data)[['date', 'value']]
            df.columns = ['date', name]  # 컬럼명을 한국어로 직접 설정
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # tz-naive로 설정
            fred_data_frames.append(df.set_index('date'))
        else:
            print(f"No data found for indicator {name} ({code}).")
    else:
        print(f"Failed to fetch data for indicator {name} ({code}): {response.status_code}")

# 데이터 빈도에 따른 리샘플링 처리
for i, df in enumerate(fred_data_frames):
    if df.empty:
        print(f"DataFrame {i} is empty, skipping resampling.")
        continue
    try:
        inferred_freq = df.index.inferred_freq
        # 빈도에 따라 일간 데이터로 변환
        if inferred_freq in ['M', 'MS']:  # 월간 데이터
            fred_data_frames[i] = df.resample('D').ffill()
        elif inferred_freq in ['W', 'W-FRI']:  # 주간 데이터
            fred_data_frames[i] = df.resample('D').ffill()
        elif inferred_freq in ['Q', 'QS-OCT']:  # 분기 데이터
            fred_data_frames[i] = df.resample('D').ffill()
        elif inferred_freq in ['B']:  # 영업일 데이터
            fred_data_frames[i] = df.resample('D').ffill()
        # else:
        #     print(f"Unknown frequency for DataFrame {i}: {inferred_freq}")
        else:
            fred_data_frames[i] = df.resample('D').ffill()
    except Exception as e:
        print(f"Error processing DataFrame {i}: {e}")

# yfinance를 통한 데이터 수집
yfinance_data_frames = []
for name, ticker in yfinance_indicators.items():
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if not df.empty:
        # df = df[['Close']].rename(columns={'Close': name})
        # name만 사용하여 컬럼 이름 지정
        df = df[['Close']]
        df.columns = [name]  # rename 대신 직접 columns 할당
        df.index = df.index.tz_localize(None)
        yfinance_data_frames.append(df)
    else:
        print(f"No data found for indicator {name} ({ticker}).")

# --------------------------
# Nikkei 225 상위 20개 종목 데이터 수집
# --------------------------
nikkei_data_frames = []
for ticker, name in nikkei_top_20:
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if not df.empty:
            #df = df[['Close']].rename(columns={'Close': f"{name}"})
            df = df[['Close']]
            df.columns = [name]  # rename 대신 직접 columns 할당
            df.index = df.index.tz_localize(None)
            nikkei_data_frames.append(df)
    except Exception as e:
        print(f"Error downloading data for {ticker} ({name}): {e}")


# 모든 데이터를 날짜 기준으로 외부 결합하여 하나의 데이터프레임으로 결합
all_data_frames = fred_data_frames + yfinance_data_frames + nikkei_data_frames
if all_data_frames:
    # 결합
    result_df = pd.concat(all_data_frames, axis=1, join='outer')  # 외부 결합으로 누락된 날짜 보완

    # 결측치 및 비정상적인 값 처리
    result_df.replace('.', pd.NA, inplace=True)  # '.'을 NaN으로 변환

    # 결측치를 이전 값으로 채움
    result_df.sort_index(inplace=True)
    result_df.ffill(inplace=True)
    result_df.bfill(inplace=True)  # 첫 행 결측치 처리 (이후 값으로 채움)

    # 일본 핵심 지표 NaN 제거 (일본 주식 예측에 필수적인 지표)
    # forward fill 후에도 NaN이 남아있는 행만 제거
    result_df = result_df.dropna(subset=['일본 10년 국채 수익률', '일본 3개월 은행간 금리', '미국 장단기 금리차'], how='any')

    # 리크루트홀딩스 상장일 이후 데이터만 사용 (2014-10-16)
    result_df = result_df[result_df.index >= '2014-10-16']

    # 특정 열에서 결측치가 있는 행 제거
    # result_df = result_df.dropna(subset=['닛케이 225'])

    # CSV 파일로 저장
    try:
        csv_path = f'total.csv'
        result_df.to_csv(csv_path, index_label="날짜", encoding='utf-8-sig')
        print(f"Data saved to {csv_path}")
    except PermissionError:
        # 현재 시간을 파일명에 추가하여 새로운 파일 생성
        # from datetime import datetime
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # csv_path = f'total_{timestamp}.csv'
        csv_path = f'total.csv'
        result_df.to_csv(csv_path, index_label="날짜", encoding='utf-8-sig')
        print(f"Permission denied for original file. Data saved to {csv_path}")
else:
    print("No data collected for any indicators.")
