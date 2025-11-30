#!/bin/bash

# 데이터베이스 접속 정보
LOCAL_HOST="localhost"
LOCAL_DB="banbu"
LOCAL_USER="bb"
LOCAL_PASS="banbu"

REMOTE_HOST="ls-88dbfbb172c94916c91af0af5d104f2a00717368.cn6a0cymwbl3.ap-northeast-2.rds.amazonaws.com"
REMOTE_DB="banbu"
REMOTE_USER="bb"
REMOTE_PASS="banbu"

# 테이블 목록
TABLES=(
    "apt_deal_sales"
    "apt_lotto_plan"
    "apt_sales"
    "region_code"
    "region_dong"
)

echo "Starting database migration..."

# 스키마 추출
echo "Extracting schema..."
mysqldump -h $LOCAL_HOST -u $LOCAL_USER -p$LOCAL_PASS --no-tablespaces --no-data $LOCAL_DB ${TABLES[@]} > schema.sql

# 데이터 추출
echo "Extracting data..."
mysqldump -h $LOCAL_HOST -u $LOCAL_USER -p$LOCAL_PASS --no-tablespaces --no-create-info $LOCAL_DB ${TABLES[@]} > data.sql

# 원격 DB에 스키마 적용
echo "Applying schema to remote database..."
mysql -h $REMOTE_HOST -u $REMOTE_USER -p$REMOTE_PASS $REMOTE_DB < schema.sql

# 원격 DB에 데이터 적용
echo "Applying data to remote database..."
mysql -h $REMOTE_HOST -u $REMOTE_USER -p$REMOTE_PASS $REMOTE_DB < data.sql

# 임시 파일 삭제
rm schema.sql data.sql

echo "Migration completed!"

# 데이터 검증
echo "Validating data migration..."
for TABLE in "${TABLES[@]}"
do
    echo "Checking table: $TABLE"
    
    # 로컬 레코드 수 확인
    LOCAL_COUNT=$(mysql -h $LOCAL_HOST -u $LOCAL_USER -p$LOCAL_PASS -N -e "SELECT COUNT(*) FROM $LOCAL_DB.$TABLE")
    
    # 원격 레코드 수 확인
    REMOTE_COUNT=$(mysql -h $REMOTE_HOST -u $REMOTE_USER -p$REMOTE_PASS -N -e "SELECT COUNT(*) FROM $REMOTE_DB.$TABLE")
    
    echo "Local records: $LOCAL_COUNT"
    echo "Remote records: $REMOTE_COUNT"
    
    if [ "$LOCAL_COUNT" = "$REMOTE_COUNT" ]; then
        echo "✅ Validation successful for $TABLE"
    else
        echo "❌ Validation failed for $TABLE"
    fi
    echo "-------------------"
done