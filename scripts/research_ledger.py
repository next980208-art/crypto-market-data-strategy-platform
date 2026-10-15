"""Generate a portable Markdown ledger from completed research JSON, without omitting failures."""

import argparse
from pathlib import Path

from scripts import run_bounded_profitability_search as search
from scripts import search_trade_patterns as prior
from scripts import automatic_research as runner


def render(paths):
    lines = ["# 전체 연구 실행 기록표", "", "실제 JSON에서 생성한 기록이다. 실패·부분 수익·후보를 모두 포함한다.",
             "정책 실행 건수는 고유 모델 수가 아니다. 같은 평가 기간을 반복 사용했으므로 독립 성과가 아니다.",
             "수익률은 구간 1/2/3 순서이며 서로 더하지 않는다. 후보라도 실거래 승인과는 다르다.", ""]
    for path in paths:
        data = prior.read(path)
        lines += [f"## {path.parent.name}", "", f"원본: `{path.relative_to(search.ROOT)}`", "",
                  f"원본 SHA256: `{runner.checksum(path)}`", "",
                  "| 정책 | 판정 | 기본 비용 수익률 % | 거래 수 | 스트레스 수익률 % | 탈락·오류 사유 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in data["trials"]:
            name = row["config"]["id"]
            if row.get("status", "completed") != "completed":
                error = str(row.get("error", row.get("reason", "not completed"))).replace("|", "/").replace("\n", " ")
                lines.append(f"| {name} | {row['status']} | - | - | - | {error} |")
                continue
            verdict = "탐색 후보" if row["candidate"] else "일부 구간 양수·탈락" if row["normal"]["positive_folds"] else "탈락"
            returns = lambda key: " / ".join(f"{v:+.3f}" for v in row[key]["fold_returns_pct"])
            trades = " / ".join(str(f["normal"]["trades"]) for f in row["folds"])
            reasons = "; ".join(row["rejection_reasons"]).replace("|", "/") or "탐색 기준 통과, 독립 검증 전"
            lines.append(f"| {name} | {verdict} | {returns('normal')} | {trades} | {returns('stress')} | {reasons} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(search.ROOT) or output == search.ROOT:
        raise ValueError("Output must be inside project")
    output.write_text(render([p.resolve() for p in args.results]), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
