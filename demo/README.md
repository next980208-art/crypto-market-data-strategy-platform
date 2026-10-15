# 재현 가능한 E2E 데모

저장된 Binance BTCUSDT 공개 샘플 1,000건을 Kafka로 재생하고 PyFlink로 처리한 뒤 Parquet Feature Store와 읽기 전용 조회 API까지 연결하는 데모입니다.

```text
saved JSONL → Kafka → consumer → PyFlink → Parquet → serving API
```

## 포함된 증거

| 경로 | 내용 |
| --- | --- |
| `input/binance_usdm_btcusdt_1000.jsonl` | 외부 호출 없이 재생할 공개 입력 1,000건 |
| `output/latest_features.parquet` | 실행에서 생성한 Feature Store 결과 |
| `results/latest_end_to_end_run.json` | 단계별 실행 결과와 건수 |
| `results/latest_serving_response.json` | 저장 결과 재조회 응답 |
| `evidence/` | Flink 완료 화면과 Feature Store 화면 |

대표 실행에서는 입력, Kafka producer, consumer, PyFlink 입력, Parquet 저장, API 조회가 모두 1,000건으로 일치했고 중복과 필수값 결측은 0건이었습니다.

## 실행

프로젝트 루트에서 실행합니다.

```powershell
pip install -r requirements.txt
python demo/run_end_to_end_demo.py
```

Docker 서비스를 이미 실행했다면 다음 옵션을 사용할 수 있습니다.

```powershell
python demo/run_end_to_end_demo.py --skip-docker-start
```

조회 화면만 실행하려면 다음 명령을 사용합니다.

```powershell
python -m demo.serving.feature_store_api
```

브라우저 주소는 `http://127.0.0.1:8077`입니다.

이 데모는 저장된 공개 입력을 사용하며 거래소 주문이나 API 자격 증명을 사용하지 않습니다.
