# 프로젝트 검증 근거

이 문서는 README의 주요 주장을 코드, 테스트, 실행 산출물과 연결합니다. 대용량 원천 데이터나 개인 실행 환경 없이도 저장소에 포함된 공개 샘플로 핵심 흐름을 확인할 수 있습니다.

## 빠른 검증

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/audit_public_repository.py
python -m unittest discover -s tests
python demo/run_end_to_end_demo.py --verify-snapshot-only
```

Linux와 macOS에서는 가상환경 활성화 명령만 `source .venv/bin/activate`로 바꾸면 됩니다.

## 주장과 근거

| 주장 | 코드·테스트 근거 | 실행·결과 근거 |
| --- | --- | --- |
| 이벤트 중복 처리 방지 | [`realtime/idempotency.py`](../realtime/idempotency.py), [`tests/test_realtime_idempotency.py`](../tests/test_realtime_idempotency.py) | E2E 결과의 중복 수 `0` |
| 배치·스트리밍 특성 동등성 | [`tests/test_batch_streaming_feature_parity.py`](../tests/test_batch_streaming_feature_parity.py) | [`demo/results/latest_end_to_end_run.json`](../demo/results/latest_end_to_end_run.json) |
| 호가창 스냅샷·증분 정합성 | [`realtime/depth_book.py`](../realtime/depth_book.py), [`tests/test_depth_book_reconstruction.py`](../tests/test_depth_book_reconstruction.py) | 순서 불일치 시 재동기화 경로를 테스트로 검증 |
| 이벤트 시간과 지연 데이터 처리 | [`flink_jobs/realtime_kafka_feature_job.py`](../flink_jobs/realtime_kafka_feature_job.py), [`tests/test_realtime_event_time_features.py`](../tests/test_realtime_event_time_features.py) | 지연 이벤트 측정과 연속성 테스트 포함 |
| 시계열 누수 방지 | [`14_walk_forward_direction_model.py`](../14_walk_forward_direction_model.py), [`tests/test_walk_forward_direction_model.py`](../tests/test_walk_forward_direction_model.py) | 학습·검증·테스트를 시간 순서로 분리 |
| 비용 인식형 전략 평가 | [`5_r_multiple_backtest.py`](../5_r_multiple_backtest.py), [`scripts/run_cost_aware_gate.py`](../scripts/run_cost_aware_gate.py) | 기본·스트레스·가혹 비용 결과를 README와 스냅샷에 공개 |
| 기준 미달 전략 차단 | [`realtime/risk_paper_service.py`](../realtime/risk_paper_service.py), [`tests/test_cost_aware_gate.py`](../tests/test_cost_aware_gate.py) | 현재 상태 `collecting / no_trade`, 실제 주문 `0` |
| 미래 관찰 결과 보존 | [`scripts/evaluate_fixed_observation.py`](../scripts/evaluate_fixed_observation.py) | [`examples/reports/forward_observation_snapshot.json`](../examples/reports/forward_observation_snapshot.json) |
| 데이터 품질 게이트 | 검증 스크립트와 54개 테스트 모듈 | [`examples/reports/weekly_evaluation_snapshot.json`](../examples/reports/weekly_evaluation_snapshot.json)의 13/13 검사 |
| 공개 저장소 위생 | [`scripts/audit_public_repository.py`](../scripts/audit_public_repository.py) | CI에서 필수 파일, 금지 파일, 대용량 파일, README 링크 검사 |

## E2E 데모의 범위

포함된 1,000개 공개 시장 이벤트를 입력으로 사용해 다음 경로를 검증합니다.

```text
JSONL sample
    -> event validation
    -> feature calculation
    -> partitioned Parquet
    -> serving response
    -> row-count / duplicate / missing-value checks
```

`--verify-snapshot-only`는 외부 Kafka 클러스터 없이 데이터 계약과 커밋된 성공 결과를 빠르게 확인합니다. 전체 경로를 다시 실행하려면 Docker Desktop을 실행한 뒤 옵션 없이 데모를 시작합니다. Kafka·Flink·Airflow 서비스 간 연결은 `docker-compose.yml`과 통합 스크립트로 구성되어 있으며, 소형 데모가 전체 운영 환경의 부하·장애 특성을 대신하지는 않습니다.

## 결과 해석 원칙

- 백테스트 수익과 미래 관찰 수익을 구분합니다.
- 거래비용, 슬리피지, 펀딩비가 포함되지 않은 수익을 승인 기준으로 사용하지 않습니다.
- 실패한 모델도 제거하지 않고 탈락 이유와 함께 기록합니다.
- 미래 관찰 중에는 규칙을 다시 최적화하지 않습니다.
- 모든 게이트를 통과하기 전에는 실제 주문을 허용하지 않습니다.

## 공개 저장소에서 제외한 항목

- 거래소 자격 증명과 개인 환경 설정
- 전체 원천 시장 데이터와 대용량 중간 산출물
- 학습 모델 바이너리, SQLite 런타임 DB와 로그
- 로컬 스케줄러 상태와 임시 파일

제외된 항목은 데모 실행이나 단위 테스트에 필요하지 않습니다.
