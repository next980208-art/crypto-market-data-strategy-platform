"""Serve a finalized Parquet feature file through a small read-only HTTP API."""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = SUBMISSION_ROOT / "output" / "latest_features.parquet"
DEFAULT_RUN_REPORT = SUBMISSION_ROOT / "results" / "latest_end_to_end_run.json"
REQUIRED_COLUMNS = {
    "timestamp",
    "datetime_utc",
    "symbol",
    "market",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_5",
    "return_1m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve or inspect the latest Parquet feature output")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--run-report", type=Path, default=DEFAULT_RUN_REPORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--once", action="store_true", help="Print one read result and exit.")
    parser.add_argument("--output", type=Path, help="Optional JSON path used with --once.")
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def load_feature_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet output does not exist: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Parquet output is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Parquet output is empty")
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(SUBMISSION_ROOT.parent)).replace("\\", "/")
    except ValueError:
        return path.name


def frame_records(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    selected = frame.tail(max(1, min(limit, 100)))
    return [
        {column: json_value(value) for column, value in row.items()}
        for row in selected.to_dict(orient="records")
    ]


def build_dataset_view(path: Path) -> dict[str, Any]:
    frame = load_feature_frame(path)
    required_nulls = int(frame[list(REQUIRED_COLUMNS)].isna().sum().sum())
    duplicate_timestamps = int(frame["timestamp"].duplicated().sum())
    return {
        "status": "healthy" if required_nulls == 0 and duplicate_timestamps == 0 else "warning",
        "storage_format": "parquet",
        "file": display_path(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "symbol": str(frame["symbol"].iloc[-1]),
        "market": str(frame["market"].iloc[-1]),
        "timeframe": str(frame["timeframe"].iloc[-1]),
        "event_time_start": json_value(frame["datetime_utc"].iloc[0]),
        "event_time_end": json_value(frame["datetime_utc"].iloc[-1]),
        "latest_close": float(frame["close"].iloc[-1]),
        "minimum_close": float(frame["close"].min()),
        "maximum_close": float(frame["close"].max()),
        "average_volume": float(frame["volume"].mean()),
        "duplicate_timestamps": duplicate_timestamps,
        "missing_required_values": required_nulls,
        "feature_columns": ["ma_5", "return_1m"],
        "columns": frame.columns.tolist(),
        "latest_rows": frame_records(frame, 10),
    }


def read_run_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_available", "file": display_path(path)}
    return json.loads(path.read_text(encoding="utf-8"))


DASHBOARD_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTCUSDT Feature Store</title>
  <style>
    :root { --ink:#17202a; --muted:#667085; --line:#d9dee7; --green:#18794e; --blue:#175cd3; --gold:#a15c00; --bg:#f6f7f9; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Arial,"Malgun Gothic",sans-serif; letter-spacing:0; }
    header { background:#fff; border-bottom:1px solid var(--line); padding:20px 28px; }
    header h1 { margin:0 0 6px; font-size:24px; }
    header p { margin:0; color:var(--muted); font-size:14px; }
    main { width:min(1180px,calc(100% - 32px)); margin:24px auto 40px; }
    .status { display:flex; align-items:center; gap:10px; margin-bottom:18px; font-weight:700; }
    .dot { width:10px; height:10px; border-radius:50%; background:var(--green); }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:6px; padding:16px; min-width:0; }
    .metric span { color:var(--muted); font-size:12px; display:block; margin-bottom:8px; }
    .metric strong { font-size:22px; overflow-wrap:anywhere; }
    section { margin-top:22px; }
    h2 { font-size:17px; margin:0 0 10px; }
    .flow { background:#fff; border:1px solid var(--line); border-radius:6px; padding:14px 16px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .step { color:var(--blue); font-weight:700; font-size:14px; }
    .arrow { color:var(--muted); }
    .table-wrap { overflow:auto; background:#fff; border:1px solid var(--line); border-radius:6px; }
    table { width:100%; border-collapse:collapse; font-size:13px; white-space:nowrap; }
    th,td { padding:10px 12px; text-align:right; border-bottom:1px solid #edf0f4; }
    th { position:sticky; top:0; background:#f9fafb; color:#475467; }
    th:first-child,td:first-child { text-align:left; }
    .note { color:var(--muted); font-size:13px; line-height:1.6; }
    code { color:var(--gold); overflow-wrap:anywhere; }
    @media (max-width:760px) { .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); } header { padding:18px 16px; } }
  </style>
</head>
<body>
  <header><h1>BTCUSDT Feature Store 조회</h1><p>PyFlink가 저장한 최종 Parquet 결과를 읽는 서빙 레이어</p></header>
  <main>
    <div class="status"><span class="dot"></span><span id="status">데이터 확인 중</span></div>
    <div class="metrics">
      <div class="metric"><span>최종 저장 행</span><strong id="rows">-</strong></div>
      <div class="metric"><span>최종 컬럼</span><strong id="columns">-</strong></div>
      <div class="metric"><span>최근 종가</span><strong id="close">-</strong></div>
      <div class="metric"><span>중복 / 결측</span><strong id="quality">-</strong></div>
    </div>
    <section><h2>한 번의 실행 흐름</h2><div class="flow"><span class="step">저장된 Binance 입력</span><span class="arrow">-&gt;</span><span class="step">Kafka</span><span class="arrow">-&gt;</span><span class="step">PyFlink</span><span class="arrow">-&gt;</span><span class="step">Parquet</span><span class="arrow">-&gt;</span><span class="step">이 화면</span></div></section>
    <section><h2>최근 피처 데이터</h2><div class="table-wrap"><table><thead><tr><th>UTC 시각</th><th>종가</th><th>거래량</th><th>MA 5</th><th>1분 수익률</th></tr></thead><tbody id="table"></tbody></table></div></section>
    <section><p class="note">데이터: <code id="range">-</code><br>파일: <code id="file">-</code></p></section>
  </main>
  <script>
    const number = (value, digits=2) => Number(value).toLocaleString('ko-KR',{maximumFractionDigits:digits});
    fetch('/api/summary').then(r => r.json()).then(d => {
      document.getElementById('status').textContent = d.status === 'healthy' ? 'Parquet 조회 정상' : '데이터 확인 필요';
      document.getElementById('rows').textContent = number(d.row_count,0);
      document.getElementById('columns').textContent = number(d.column_count,0);
      document.getElementById('close').textContent = number(d.latest_close,2);
      document.getElementById('quality').textContent = `${d.duplicate_timestamps} / ${d.missing_required_values}`;
      document.getElementById('range').textContent = `${d.symbol} ${d.timeframe} | ${d.event_time_start} ~ ${d.event_time_end}`;
      document.getElementById('file').textContent = d.file;
      document.getElementById('table').innerHTML = d.latest_rows.slice().reverse().map(row => `<tr><td>${row.datetime_utc}</td><td>${number(row.close)}</td><td>${number(row.volume,4)}</td><td>${number(row.ma_5)}</td><td>${number(row.return_1m,8)}</td></tr>`).join('');
    }).catch(error => { document.getElementById('status').textContent = `조회 실패: ${error}`; });
  </script>
</body>
</html>"""


def make_handler(parquet_path: Path, run_report_path: Path) -> type[BaseHTTPRequestHandler]:
    class FeatureHandler(BaseHTTPRequestHandler):
        def send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/health":
                    view = build_dataset_view(parquet_path)
                    self.send_json({"status": view["status"], "row_count": view["row_count"]})
                    return
                if parsed.path == "/api/summary":
                    self.send_json(build_dataset_view(parquet_path))
                    return
                if parsed.path == "/api/rows":
                    limit_text = parse_qs(parsed.query).get("limit", ["10"])[0]
                    limit = max(1, min(int(limit_text), 100))
                    frame = load_feature_frame(parquet_path)
                    self.send_json({"count": limit, "rows": frame_records(frame, limit)})
                    return
                if parsed.path == "/api/schema":
                    frame = load_feature_frame(parquet_path)
                    self.send_json({"columns": {name: str(dtype) for name, dtype in frame.dtypes.items()}})
                    return
                if parsed.path == "/api/run":
                    self.send_json(read_run_report(run_report_path))
                    return
                self.send_json({"error": "not_found"}, status=404)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"status": "error", "error": str(exc)}, status=500)

        def log_message(self, format_text: str, *args: Any) -> None:
            print(f"[feature-api] {self.address_string()} {format_text % args}")

    return FeatureHandler


def serve(parquet_path: Path, run_report_path: Path, host: str, port: int) -> None:
    build_dataset_view(parquet_path)
    server = ThreadingHTTPServer((host, port), make_handler(parquet_path, run_report_path))
    print(f"Feature Store API: http://{host}:{port}")
    print(f"Parquet: {parquet_path.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    parquet_path = args.parquet.resolve()
    if args.once:
        payload = build_dataset_view(parquet_path)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    serve(parquet_path, args.run_report.resolve(), args.host, args.port)


if __name__ == "__main__":
    main()
