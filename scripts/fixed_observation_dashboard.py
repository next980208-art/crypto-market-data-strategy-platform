"""Read-only dashboard for the frozen breakout forward observation."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATION = PROJECT_ROOT / "runtime_reports" / "fixed_breakout_forward_20260922"
COST_CASES = ("normal", "stress", "severe")
TABLE_KINDS = ("trades", "daily")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--once", action="store_true", help="Print the current dashboard summary and exit")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required observation file is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def records(frame: pd.DataFrame, limit: int = 100) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    selected = frame.tail(max(1, min(limit, 500)))
    return [
        {column: json_value(value) for column, value in row.items()}
        for row in selected.to_dict(orient="records")
    ]


def load_table(root: Path, cost: str, kind: str, limit: int = 100) -> list[dict[str, Any]]:
    if cost not in COST_CASES or kind not in TABLE_KINDS:
        raise ValueError("Unsupported observation table")
    path = root / f"current_{cost}_{kind}.parquet"
    if not path.exists():
        return []
    return records(pd.read_parquet(path), limit)


def latest_market(root: Path) -> dict[str, Any]:
    path = root / "bars_15m.parquet"
    if not path.exists():
        return {"latest_close": None, "data_start_utc": None, "data_end_utc": None}
    frame = pd.read_parquet(path, columns=["timestamp", "close"])
    if frame.empty:
        return {"latest_close": None, "data_start_utc": None, "data_end_utc": None}
    frame = frame.sort_values("timestamp")
    return {
        "latest_close": float(frame.close.iloc[-1]),
        "data_start_utc": pd.Timestamp(int(frame.timestamp.iloc[0]), unit="ms", tz="UTC").isoformat(),
        "data_end_utc": pd.Timestamp(int(frame.timestamp.iloc[-1]), unit="ms", tz="UTC").isoformat(),
    }


def load_weekly_evaluation(root: Path) -> dict[str, Any]:
    path = root / "weekly_evaluation.json"
    if not path.exists():
        return {
            "generated_utc": None,
            "quality": {"status": "not_available", "warning_count": 0, "critical_count": 0, "checks": []},
            "weekly": {"complete_weeks": 0, "positive_weeks": 0, "failed_weeks": 0,
                       "current_week": None, "weeks": [], "four_week_blocks": []},
            "monthly": {"periods": []},
            "review": {"status": "not_available", "decision": "no_trade"},
        }
    return read_json(path)


def build_summary(root: Path) -> dict[str, Any]:
    root = root.resolve()
    protocol = read_json(root / "protocol.json")
    state = read_json(root / "state.json")
    report = read_json(root / "latest_report.json")
    observed = float(report.get("observed_days", 0.0))
    required = float(protocol.get("minimum_observation_days", 90))
    result_rows = {}
    for cost in COST_CASES:
        metrics = report.get("results", {}).get(cost, {}).get("metrics", {})
        result_rows[cost] = {
            "return_pct": float(metrics.get("total_return_pct", 0.0)),
            "trades": int(metrics.get("trades", 0)),
            "win_rate_pct": float(metrics.get("win_rate_pct", 0.0)),
            "avg_return_r": float(metrics.get("avg_return_r", 0.0)),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
            "signals": int(metrics.get("signals", 0)),
            "long_signals": int(metrics.get("long_signals", 0)),
            "short_signals": int(metrics.get("short_signals", 0)),
        }
    storage_limit = int(protocol.get("storage_limit_bytes", 0))
    database_bytes = int(report.get("database_bytes", 0))
    rule = str(protocol.get("rule", ""))
    if rule == "range_position_48bar >=0.9 long; <=0.1 short":
        rule = "직전 48봉 범위 위치가 0.9 이상이면 롱, 0.1 이하면 숏"
    weekly_gate = protocol.get("weekly_gate")
    if weekly_gate == "At least 3 positive complete weeks out of each 4-week review block":
        weekly_gate = "각 4주 검토 구간에서 완전한 주 3주 이상 수익"
    monitoring = load_weekly_evaluation(root)
    quality = monitoring.get("quality", {})
    weekly = monitoring.get("weekly", {})
    return {
        "status": str(report.get("status", state.get("status", "unknown"))),
        "decision": str(report.get("decision", "no_trade")),
        "orders_sent": int(report.get("orders_sent", 0)),
        "credentials_used": bool(report.get("credentials_used", False)),
        "symbol": str(protocol.get("symbol", "BTCUSDT")),
        "timeframe": str(protocol.get("timeframe", "15m")),
        "rule_id": str(protocol.get("rule_id", "unknown")),
        "rule": rule,
        "observation_start_utc": protocol.get("observation_start_utc"),
        "review_after_utc": protocol.get("review_after_utc"),
        "last_run_utc": state.get("last_run_utc"),
        "observed_days": observed,
        "required_days": required,
        "remaining_days": max(0.0, required - observed),
        "progress_pct": min(100.0, 100.0 * observed / required) if required else 0.0,
        "minimum_monthly_trades": int(protocol.get("minimum_monthly_trades", 0)),
        "weekly_gate": weekly_gate,
        "contract_bars": int(report.get("contract_bars", 0)),
        "mark_bars": int(report.get("mark_bars", 0)),
        "funding_events": int(report.get("funding_events", 0)),
        "mature_signal_rows": int(report.get("mature_signal_rows", 0)),
        "run_count": len(state.get("runs", [])),
        "database_bytes": database_bytes,
        "storage_limit_bytes": storage_limit,
        "storage_pct": 100.0 * database_bytes / storage_limit if storage_limit else 0.0,
        "risk_per_trade_pct": 100.0 * float(protocol.get("risk_per_trade_pct", 0.0)),
        "stop_distance_pct": 100.0 * float(protocol.get("stop_distance_pct", 0.0)),
        "take_profit_r": float(protocol.get("take_profit_r", 0.0)),
        "max_holding_minutes": int(protocol.get("max_holding_minutes", 0)),
        "costs": result_rows,
        "market": latest_market(root),
        "monitoring": {
            "generated_utc": monitoring.get("generated_utc"),
            "quality_status": quality.get("status", "not_available"),
            "warning_count": int(quality.get("warning_count", 0)),
            "critical_count": int(quality.get("critical_count", 0)),
            "complete_weeks": int(weekly.get("complete_weeks", 0)),
            "positive_weeks": int(weekly.get("positive_weeks", 0)),
            "failed_weeks": int(weekly.get("failed_weeks", 0)),
            "current_week": weekly.get("current_week"),
            "review_status": monitoring.get("review", {}).get("status", "not_available"),
        },
    }


def load_runs(root: Path) -> list[dict[str, Any]]:
    state = read_json(root / "state.json")
    return list(reversed(state.get("runs", [])))


DASHBOARD_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTCUSDT 미래 관찰</title>
  <style>
    :root {
      --ink:#182230; --muted:#667085; --line:#d0d5dd; --surface:#ffffff; --bg:#f3f5f7;
      --blue:#175cd3; --green:#067647; --red:#b42318; --amber:#b54708; --cyan:#0e7090;
    }
    * { box-sizing:border-box; }
    html,body { width:100vw; max-width:100vw; }
    body { margin:0; min-width:0; overflow-x:hidden; color:var(--ink); background:var(--bg); font-family:Arial,"Malgun Gothic",sans-serif; letter-spacing:0; }
    header { width:100vw; max-width:100vw; overflow:hidden; background:var(--surface); border-bottom:1px solid var(--line); }
    .header-inner { width:min(1280px,calc(100% - 32px)); margin:auto; min-height:78px; display:flex; align-items:center; justify-content:space-between; gap:20px; }
    .header-inner > div { min-width:0; max-width:100%; }
    h1 { margin:0 0 5px; font-size:22px; font-weight:700; }
    .subtitle { color:var(--muted); font-size:13px; }
    .status-line { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:700; white-space:nowrap; }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--amber); }
    button { border:1px solid var(--line); background:#fff; color:var(--ink); padding:8px 12px; border-radius:5px; font-weight:700; cursor:pointer; }
    button:hover { border-color:#98a2b3; background:#f9fafb; }
    main { width:min(1280px,calc(100% - 32px)); margin:20px auto 40px; }
    .notice { border-left:4px solid var(--amber); background:#fffaeb; padding:11px 14px; font-size:13px; line-height:1.5; margin-bottom:16px; }
    .progress-head { display:flex; justify-content:space-between; gap:12px; margin-bottom:7px; font-size:13px; }
    .progress-track { height:10px; border:1px solid #b2ddff; background:#eff8ff; overflow:hidden; }
    .progress-fill { width:0; height:100%; background:var(--blue); transition:width .25s ease; }
    .metrics { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin-top:16px; }
    .metric { min-width:0; min-height:92px; border:1px solid var(--line); border-radius:6px; background:var(--surface); padding:13px; }
    .metric span { display:block; color:var(--muted); font-size:12px; margin-bottom:9px; }
    .metric strong { display:block; font-size:20px; overflow-wrap:anywhere; }
    .metric small { display:block; color:var(--muted); font-size:11px; margin-top:6px; }
    .positive { color:var(--green); } .negative { color:var(--red); } .neutral { color:var(--ink); }
    .warning { color:var(--amber); } .critical { color:var(--red); }
    .badge { display:inline-block; padding:3px 7px; border-radius:4px; font-size:11px; font-weight:700; background:#f2f4f7; color:#475467; }
    .badge.pass,.badge.healthy { background:#ecfdf3; color:var(--green); }
    .badge.warning,.badge.collecting { background:#fffaeb; color:var(--amber); }
    .badge.critical,.badge.fail { background:#fef3f2; color:var(--red); }
    .grid { display:grid; width:100%; max-width:100%; grid-template-columns:minmax(0,1.35fr) minmax(320px,.65fr); gap:16px; margin-top:18px; }
    section { min-width:0; }
    .panel { width:100%; max-width:100%; min-width:0; background:var(--surface); border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .panel-head { padding:13px 15px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:12px; }
    h2 { margin:0; font-size:15px; }
    .panel-body { padding:14px 15px; }
    canvas { display:block; width:100%; max-width:100%; height:230px; }
    .legend { display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
    .legend i { display:inline-block; width:18px; height:3px; margin-right:5px; vertical-align:middle; }
    .definition { display:grid; grid-template-columns:145px 1fr; gap:9px 12px; margin:0; font-size:13px; }
    .definition dt { color:var(--muted); } .definition dd { min-width:0; margin:0; overflow-wrap:anywhere; }
    .table-wrap { width:100%; max-width:100%; overflow:auto; max-height:330px; }
    table { width:100%; border-collapse:collapse; font-size:12px; white-space:nowrap; }
    th,td { padding:9px 11px; text-align:right; border-bottom:1px solid #eaecf0; }
    th { position:sticky; top:0; z-index:1; color:#475467; background:#f9fafb; }
    th:first-child,td:first-child { text-align:left; }
    .empty { color:var(--muted); text-align:center; padding:30px 12px; }
    .footer-note { color:var(--muted); font-size:12px; line-height:1.6; margin-top:14px; }
    @media (max-width:1050px) { .metrics { grid-template-columns:repeat(3,minmax(0,1fr)); } .grid { grid-template-columns:minmax(0,1fr); } }
    @media (max-width:620px) {
      .header-inner { width:calc(100vw - 20px); max-width:calc(100vw - 20px); align-items:flex-start; padding:15px 0; flex-direction:column; }
      .header-actions { width:calc(100vw - 20px); max-width:calc(100vw - 20px); display:grid; grid-template-columns:minmax(0,1fr); align-items:start; gap:10px; }
      .header-actions button { justify-self:start; }
      .status-line { min-width:0; white-space:normal; }
      main { width:calc(100vw - 20px); max-width:calc(100vw - 20px); margin-top:12px; }
      .notice { overflow-wrap:anywhere; }
      .progress-head { align-items:flex-start; flex-direction:column; gap:4px; }
      .grid,.panel { width:calc(100vw - 20px); max-width:calc(100vw - 20px); }
      .panel-body canvas { width:calc(100vw - 52px); max-width:calc(100vw - 52px); }
      .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .metric { min-height:88px; } .metric strong { font-size:18px; }
      .definition { grid-template-columns:110px 1fr; }
      canvas { height:190px; }
    }
    @media (max-width:520px) {
      .metrics { grid-template-columns:1fr; }
      .metric { min-height:80px; }
      .definition { grid-template-columns:95px minmax(0,1fr); }
      h1 { font-size:20px; }
    }
  </style>
</head>
<body>
  <header><div class="header-inner">
    <div><h1>BTCUSDT 고정 규칙 미래 관찰</h1><div class="subtitle">과거 결과에 맞추지 않고 고정된 돌파 규칙을 새 데이터에서 관찰합니다.</div></div>
    <div class="header-actions"><div class="status-line"><span class="dot" id="statusDot"></span><span id="status">불러오는 중</span></div><button id="refresh" type="button">새로고침</button></div>
  </div></header>
  <main>
    <div class="notice"><strong>연구 전용:</strong> 이 화면은 공개 데이터를 이용한 모의 관찰 결과입니다. 실제 주문은 차단되어 있으며 수익을 보장하지 않습니다.</div>
    <section aria-label="관찰 진행률">
      <div class="progress-head"><strong id="progressText">관찰 진행률</strong><span id="reviewDate">-</span></div>
      <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
    </section>
    <div class="metrics">
      <div class="metric"><span>관찰 경과</span><strong id="days">-</strong><small id="remaining">-</small></div>
      <div class="metric"><span>확정 거래</span><strong id="trades">-</strong><small id="signals">-</small></div>
      <div class="metric"><span>기본 비용 수익률</span><strong id="normalReturn">-</strong><small id="normalWin">-</small></div>
      <div class="metric"><span>스트레스 수익률</span><strong id="stressReturn">-</strong><small id="stressDrawdown">-</small></div>
      <div class="metric"><span>가혹 비용 수익률</span><strong id="severeReturn">-</strong><small id="severeDrawdown">-</small></div>
      <div class="metric"><span>실제 주문</span><strong id="orders">-</strong><small id="decision">-</small></div>
      <div class="metric"><span>최근 BTC 종가</span><strong id="close">-</strong><small>저장된 15분봉</small></div>
      <div class="metric"><span>신호 방향</span><strong id="direction">-</strong><small id="directionShare">-</small></div>
      <div class="metric"><span>펀딩 이벤트</span><strong id="funding">-</strong><small>실제 시점 기록</small></div>
      <div class="metric"><span>완료 15분봉</span><strong id="bars">-</strong><small id="matureRows">-</small></div>
      <div class="metric"><span>수집 DB</span><strong id="storage">-</strong><small id="storageLimit">-</small></div>
      <div class="metric"><span>최근 갱신</span><strong id="updated">-</strong><small id="runCount">-</small></div>
      <div class="metric"><span>데이터 품질</span><strong id="quality">-</strong><small id="qualityCounts">-</small></div>
      <div class="metric"><span>주간 평가</span><strong id="weeklyStatus">-</strong><small id="weeklyCounts">-</small></div>
    </div>

    <div class="grid">
      <section class="panel"><div class="panel-head"><h2>기본 비용 거래별 자산 흐름</h2><div class="legend"><span><i style="background:#175cd3"></i>자산</span><span><i style="background:#98a2b3"></i>시작값</span></div></div><div class="panel-body"><canvas id="equityChart" aria-label="거래별 자산 변화 차트"></canvas></div></section>
      <section class="panel"><div class="panel-head"><h2>고정 프로토콜</h2></div><div class="panel-body"><dl class="definition">
        <dt>규칙</dt><dd id="rule">-</dd><dt>손절</dt><dd id="stop">-</dd><dt>목표</dt><dd id="target">-</dd><dt>최대 보유</dt><dd id="holding">-</dd><dt>거래당 위험</dt><dd id="risk">-</dd><dt>월 최소 거래</dt><dd id="minimumTrades">-</dd><dt>주별 기준</dt><dd id="weeklyGate">-</dd>
      </dl></div></section>
    </div>

    <div class="grid">
      <section class="panel"><div class="panel-head"><h2>최근 확정 거래</h2><span class="subtitle">기본 비용</span></div><div class="table-wrap"><table><thead><tr><th>진입 시각</th><th>방향</th><th>진입가</th><th>보유</th><th>R</th><th>수익률</th><th>자산</th></tr></thead><tbody id="tradeRows"></tbody></table><div class="empty" id="tradeEmpty">확정 거래가 없습니다.</div></div></section>
      <section class="panel"><div class="panel-head"><h2>수집 실행 이력</h2><span class="subtitle">최근 실행 우선</span></div><div class="table-wrap"><table><thead><tr><th>거래소 시각</th><th>관찰일</th><th>상태</th></tr></thead><tbody id="runRows"></tbody></table></div></section>
    </div>
    <div class="grid">
      <section class="panel"><div class="panel-head"><h2>주별 성과 기록</h2><span class="subtitle">완료되지 않은 주는 판정 제외</span></div><div class="table-wrap"><table><thead><tr><th>주</th><th>종료 시각</th><th>상태</th><th>거래</th><th>기본</th><th>스트레스</th><th>가혹</th></tr></thead><tbody id="weekRows"></tbody></table><div class="empty" id="weekEmpty">주간 평가 자료가 없습니다.</div></div></section>
      <section class="panel"><div class="panel-head"><h2>데이터·안전 검사</h2><span class="subtitle" id="monitorUpdated">-</span></div><div class="table-wrap"><table><thead><tr><th>검사</th><th>상태</th><th>설명</th></tr></thead><tbody id="checkRows"></tbody></table><div class="empty" id="checkEmpty">검사 자료가 없습니다.</div></div></section>
    </div>
    <p class="footer-note">화면은 60초마다 갱신됩니다. 수치는 <code>latest_report.json</code>과 Parquet 원장을 직접 읽습니다. 전략 조건은 대시보드에서 변경할 수 없습니다.</p>
  </main>
  <script>
    const $ = id => document.getElementById(id);
    const fmt = (v,d=2) => Number(v).toLocaleString('ko-KR',{minimumFractionDigits:d,maximumFractionDigits:d});
    const pct = v => `${Number(v)>=0?'+':''}${fmt(v,3)}%`;
    const kst = v => v ? new Intl.DateTimeFormat('ko-KR',{timeZone:'Asia/Seoul',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(v)) : '-';
    const colorize = (node,value) => { node.className = Number(value)>0?'positive':Number(value)<0?'negative':'neutral'; };
    function drawEquity(rows) {
      const canvas=$('equityChart'), rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
      canvas.width=Math.max(1,Math.floor(rect.width*dpr)); canvas.height=Math.max(1,Math.floor(rect.height*dpr));
      const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr); const w=rect.width,h=rect.height,p={l:48,r:18,t:18,b:32};
      const values=[1,...rows.map(r=>Number(r.equity))]; const min=Math.min(...values),max=Math.max(...values),span=Math.max(.005,max-min);
      const low=min-span*.2,high=max+span*.2; const x=i=>p.l+(w-p.l-p.r)*(i/Math.max(1,values.length-1)); const y=v=>p.t+(h-p.t-p.b)*(high-v)/(high-low);
      ctx.clearRect(0,0,w,h); ctx.strokeStyle='#eaecf0'; ctx.lineWidth=1; ctx.fillStyle='#667085'; ctx.font='11px Arial';
      for(let i=0;i<4;i++){const v=low+(high-low)*i/3,yy=y(v);ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(w-p.r,yy);ctx.stroke();ctx.fillText(fmt(v,3),3,yy+4);}
      ctx.strokeStyle='#98a2b3';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(p.l,y(1));ctx.lineTo(w-p.r,y(1));ctx.stroke();ctx.setLineDash([]);
      ctx.strokeStyle='#175cd3';ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>{if(i===0)ctx.moveTo(x(i),y(v));else ctx.lineTo(x(i),y(v));});ctx.stroke();
      values.forEach((v,i)=>{ctx.fillStyle='#175cd3';ctx.beginPath();ctx.arc(x(i),y(v),3,0,Math.PI*2);ctx.fill();});
      ctx.fillStyle='#667085';ctx.fillText('시작',p.l-10,h-9);ctx.fillText(`${rows.length}거래`,Math.max(p.l,w-p.r-34),h-9);
    }
    async function load() {
      const [summary,trades,runs,weekly]=await Promise.all([
        fetch('/api/summary',{cache:'no-store'}).then(r=>r.json()),
        fetch('/api/table?cost=normal&kind=trades&limit=100',{cache:'no-store'}).then(r=>r.json()),
        fetch('/api/runs',{cache:'no-store'}).then(r=>r.json()),
        fetch('/api/weekly',{cache:'no-store'}).then(r=>r.json())
      ]);
      $('status').textContent=summary.status==='ready_for_review'?'수동 검토 필요':'관찰 중'; $('statusDot').style.background=summary.status==='ready_for_review'?'#067647':'#b54708';
      $('progressText').textContent=`${fmt(summary.observed_days,2)} / ${fmt(summary.required_days,0)}일`;
      $('reviewDate').textContent=`최초 검토 ${kst(summary.review_after_utc)}`; $('progressFill').style.width=`${summary.progress_pct}%`;
      $('days').textContent=`${fmt(summary.observed_days,2)}일`; $('remaining').textContent=`약 ${fmt(summary.remaining_days,2)}일 남음`;
      const n=summary.costs.normal,s=summary.costs.stress,v=summary.costs.severe;
      $('trades').textContent=`${n.trades}건`; $('signals').textContent=`신호 ${n.signals}개`;
      [['normalReturn',n.return_pct],['stressReturn',s.return_pct],['severeReturn',v.return_pct]].forEach(([id,value])=>{$(id).textContent=pct(value);colorize($(id),value);});
      $('normalWin').textContent=`승률 ${fmt(n.win_rate_pct,1)}%`; $('stressDrawdown').textContent=`MDD ${pct(s.max_drawdown_pct)}`; $('severeDrawdown').textContent=`MDD ${pct(v.max_drawdown_pct)}`;
      $('orders').textContent=`${summary.orders_sent}건`; $('decision').textContent=summary.decision; $('close').textContent=summary.market.latest_close==null?'-':fmt(summary.market.latest_close,1);
      $('direction').textContent=`롱 ${n.long_signals} / 숏 ${n.short_signals}`; $('directionShare').textContent=n.signals?`롱 비중 ${fmt(100*n.long_signals/n.signals,1)}%`:'신호 없음';
      $('funding').textContent=`${summary.funding_events}건`; $('bars').textContent=`${summary.contract_bars.toLocaleString('ko-KR')}개`; $('matureRows').textContent=`성숙 행 ${summary.mature_signal_rows.toLocaleString('ko-KR')}개`;
      $('storage').textContent=`${fmt(summary.database_bytes/1024,1)} KiB`; $('storageLimit').textContent=`한도 ${fmt(summary.storage_limit_bytes/1024/1024,0)} MiB · ${fmt(summary.storage_pct,2)}%`;
      $('updated').textContent=kst(summary.last_run_utc); $('runCount').textContent=`실행 ${summary.run_count}회`;
      const monitor=summary.monitoring, current=monitor.current_week;
      $('quality').textContent=monitor.quality_status; $('quality').className=monitor.quality_status;
      $('qualityCounts').textContent=`경고 ${monitor.warning_count} · 심각 ${monitor.critical_count}`;
      $('weeklyStatus').textContent=current?`${current.week}주차 ${current.status}`:'평가 대기';
      $('weeklyStatus').className=current?current.status:'neutral';
      $('weeklyCounts').textContent=`완료 ${monitor.complete_weeks} · 양수 ${monitor.positive_weeks} · 실패 ${monitor.failed_weeks}`;
      $('rule').textContent=summary.rule; $('stop').textContent=`${fmt(summary.stop_distance_pct,1)}%`; $('target').textContent=`${fmt(summary.take_profit_r,1)}R`; $('holding').textContent=`${fmt(summary.max_holding_minutes/60,0)}시간`; $('risk').textContent=`${fmt(summary.risk_per_trade_pct,1)}%`; $('minimumTrades').textContent=`${summary.minimum_monthly_trades}건`; $('weeklyGate').textContent=summary.weekly_gate;
      $('tradeEmpty').style.display=trades.rows.length?'none':'block';
      $('tradeRows').innerHTML=trades.rows.slice().reverse().map(r=>`<tr><td>${kst(r.entry_time_utc)}</td><td>${r.action==='long'?'롱':'숏'}</td><td>${fmt(r.entry_price,1)}</td><td>${fmt(r.holding_minutes,0)}분</td><td class="${r.return_r>=0?'positive':'negative'}">${fmt(r.return_r,3)}</td><td class="${r.return_pct>=0?'positive':'negative'}">${pct(100*r.return_pct)}</td><td>${fmt(r.equity,4)}</td></tr>`).join('');
      $('runRows').innerHTML=runs.runs.map(r=>`<tr><td>${kst(new Date(Number(r.exchange_server_ms)))}</td><td>${fmt(r.observed_days,2)}</td><td>${r.status}</td></tr>`).join('');
      const weeks=weekly.weekly?.weeks||[]; $('weekEmpty').style.display=weeks.length?'none':'block';
      $('weekRows').innerHTML=weeks.slice().reverse().map(w=>`<tr><td>${w.week}주</td><td>${kst(w.end_utc)}</td><td><span class="badge ${w.status}">${w.status}</span></td><td>${w.costs.normal.trades}</td><td class="${w.costs.normal.return_pct>=0?'positive':'negative'}">${pct(w.costs.normal.return_pct)}</td><td class="${w.costs.stress.return_pct>=0?'positive':'negative'}">${pct(w.costs.stress.return_pct)}</td><td class="${w.costs.severe.return_pct>=0?'positive':'negative'}">${pct(w.costs.severe.return_pct)}</td></tr>`).join('');
      const checks=weekly.quality?.checks||[]; $('checkEmpty').style.display=checks.length?'none':'block';
      $('checkRows').innerHTML=checks.map(c=>`<tr><td>${c.id}</td><td><span class="badge ${c.status}">${c.status}</span></td><td>${c.message}</td></tr>`).join('');
      $('monitorUpdated').textContent=`평가 ${kst(weekly.generated_utc)}`;
      drawEquity(trades.rows);
    }
    $('refresh').addEventListener('click',()=>load().catch(showError)); window.addEventListener('resize',()=>load().catch(()=>{}));
    function showError(error){$('status').textContent=`조회 실패: ${error.message}`;$('statusDot').style.background='#b42318';}
    load().catch(showError); setInterval(()=>load().catch(showError),60000);
  </script>
</body>
</html>"""


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
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
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/health":
                    summary = build_summary(root)
                    self.send_json({"status": summary["status"], "decision": summary["decision"],
                                    "orders_sent": summary["orders_sent"],
                                    "quality": summary["monitoring"]["quality_status"]})
                    return
                if parsed.path == "/api/summary":
                    self.send_json(build_summary(root))
                    return
                if parsed.path == "/api/runs":
                    self.send_json({"runs": load_runs(root)})
                    return
                if parsed.path == "/api/weekly":
                    self.send_json(load_weekly_evaluation(root))
                    return
                if parsed.path == "/api/table":
                    query = parse_qs(parsed.query)
                    cost = query.get("cost", ["normal"])[0]
                    kind = query.get("kind", ["trades"])[0]
                    limit = max(1, min(int(query.get("limit", ["100"])[0]), 500))
                    self.send_json({"cost": cost, "kind": kind, "rows": load_table(root, cost, kind, limit)})
                    return
                self.send_json({"error": "not_found"}, status=404)
            except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
                self.send_json({"status": "error", "error": str(exc)}, status=500)

        def log_message(self, format_text: str, *args: Any) -> None:
            print(f"[observation-dashboard] {self.address_string()} {format_text % args}")

    return DashboardHandler


def validate_root(path: Path) -> Path:
    root = path.resolve()
    if root == PROJECT_ROOT or not root.is_relative_to(PROJECT_ROOT):
        raise ValueError("Observation must be a project subdirectory")
    build_summary(root)
    return root


def serve(root: Path, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(root))
    print(f"Forward observation dashboard: http://{host}:{port}")
    print(f"Observation: {root.relative_to(PROJECT_ROOT)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    root = validate_root(args.observation)
    if args.once:
        print(json.dumps(build_summary(root), ensure_ascii=False, indent=2))
        return
    serve(root, args.host, args.port)


if __name__ == "__main__":
    main()
