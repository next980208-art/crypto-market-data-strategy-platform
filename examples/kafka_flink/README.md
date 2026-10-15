# Kafka·Flink 이벤트 처리 예제

시장 이벤트의 정규화, 식별자 생성, Kafka 전달과 Flink 입력 변환을 보여주는 작은 예제입니다.

## 구성

- `kafka_market_event_producer.py`: 저장 데이터 또는 공개 Binance 데이터를 공통 이벤트 계약으로 변환해 Kafka로 전송
- `kafka_market_event_consumer.py`: 실행 ID를 기준으로 이벤트를 소비하고 건수·중복을 검증
- `prepare_flink_input.py`: 소비한 JSON Lines를 Flink 입력 형식으로 변환
- `write_flink_report.py`: 성공 마커와 Parquet 산출물을 검증해 실행 보고서 생성

## 예시

```powershell
python examples/kafka_flink/kafka_market_event_producer.py `
  --count 1000 `
  --run-id portfolio-demo-v1

python examples/kafka_flink/kafka_market_event_consumer.py `
  --expected-count 1000 `
  --run-id portfolio-demo-v1
```

생산자와 소비자는 동일한 시장 메타데이터 계약을 사용하며 테스트에서 이벤트 ID의 결정성, 실행 간 충돌 방지, 종목·시장 정규화를 검증합니다.
