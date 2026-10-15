# GitHub 업로드 안내

## 1. GitHub에서 빈 저장소 만들기

1. <https://github.com/new>에 접속합니다.
2. Repository name에 `crypto-market-data-strategy-platform`을 입력합니다.
3. Description에는 아래 문장을 사용합니다.

   ```text
   Airflow, Kafka, Flink, Parquet 기반 암호화폐 시장 데이터 파이프라인과 비용 인식형 전략 검증 플랫폼
   ```

4. 취업 포트폴리오로 공개할 경우 `Public`을 선택합니다.
5. `Add a README file`, `.gitignore`, `Choose a license`는 선택하지 않습니다. 이 저장소에 이미 필요한 파일이 들어 있습니다.
6. `Create repository`를 누릅니다.

## 2. 로컬 폴더를 GitHub로 전송하기

PowerShell에서 이 프로젝트 폴더로 이동한 다음 아래 명령을 실행합니다.

```powershell
git config user.name "본인의 GitHub 표시 이름"
git config user.email "본인의 GitHub 이메일"
git add .
git commit -m "Add crypto market data and strategy validation platform"
git remote add origin https://github.com/본인의아이디/crypto-market-data-strategy-platform.git
git push -u origin main
```

처음 `git push`를 실행하면 브라우저 로그인 또는 Git Credential Manager 인증 화면이 나타날 수 있습니다. GitHub 비밀번호를 터미널에 직접 입력하는 방식은 사용하지 않습니다.

## 3. 업로드 후 저장소 설정

GitHub 저장소의 오른쪽 `About` 영역에서 다음 Topics를 추가하면 검색과 기술 파악에 도움이 됩니다.

```text
data-engineering
apache-kafka
apache-flink
apache-airflow
machine-learning
algorithmic-trading
parquet
python
mlops
```

가능하면 대표 이미지로 `docs/images/forward_observation_dashboard.png`를 활용합니다.

## 4. 확인할 사항

- 저장소 첫 화면에 `README.md`가 정상 표시되는지 확인합니다.
- README의 대시보드 이미지 2개가 보이는지 확인합니다.
- `Actions` 탭에서 unit-tests가 실행되는지 확인합니다.
- 저장소에 `.env`, API 키, SQLite DB, 대용량 시장 데이터가 없는지 확인합니다.
- 저장소 URL을 이력서와 포트폴리오 문서에 연결합니다.

## 오류 대응

원격 저장소 주소를 잘못 등록했다면 다음 명령으로 수정합니다.

```powershell
git remote set-url origin https://github.com/본인의아이디/crypto-market-data-strategy-platform.git
```

Git 사용자 정보가 아직 없다면 반드시 현재 저장소에서만 `user.name`과 `user.email`을 설정합니다. `--global`은 사용하지 않아도 됩니다.
