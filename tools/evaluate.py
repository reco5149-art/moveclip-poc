"""
evaluate.py — 개선 효과 검증 채점기.

정답 라벨(sample_workout_truth.json)과 파이프라인 결과(result.json)를 대조해
  - 세트 구간 재현율/정밀도 (IoU 0.5 기준)
  - 반복 수 정확도 (±1회 허용)
  - 처리 시간
을 계산하고 docs/validation_result.md 로 저장한다.

사용법:
    python tools/evaluate.py data/sample_workout_truth.json outputs/sample/result.json
"""

import json
import sys

IOU_THRESHOLD = 0.5
REP_TOLERANCE = 1

# 수작업 기준선 (임시값 — 본인 영상으로 스톱워치 측정해서 바꿔 쓰세요)
# 영상 1분당 수작업 소요 시간(초). 짧은 영상일수록 수작업도 금방 끝나므로
# 이 선형 가정은 60초짜리 테스트 영상에서 단축률을 과소평가한다.
MANUAL_SEC_PER_MIN = 150.0


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def main(truth_path, result_path, out_md="docs/validation_result.md"):
    truth = json.load(open(truth_path))
    res = json.load(open(result_path))

    gt = truth["sets"]
    pred = res["sets_detected"]

    matched, used = [], set()
    for i, g in enumerate(gt):
        best_j, best_iou = None, 0.0
        for j, p in enumerate(pred):
            if j in used:
                continue
            v = iou((g["start"], g["end"]), (p["start"], p["end"]))
            if v > best_iou:
                best_j, best_iou = j, v
        if best_j is not None and best_iou >= IOU_THRESHOLD:
            used.add(best_j)
            matched.append((i, best_j, best_iou))

    recall = len(matched) / len(gt) if gt else 0.0
    precision = len(matched) / len(pred) if pred else 0.0
    rep_ok = sum(1 for i, j, _ in matched
                 if abs(gt[i]["reps"] - pred[j]["reps"]) <= REP_TOLERANCE)
    rep_exact = sum(1 for i, j, _ in matched if gt[i]["reps"] == pred[j]["reps"])
    rep_acc = rep_ok / len(gt) if gt else 0.0

    dur_min = res["video"]["duration_sec"] / 60
    manual = MANUAL_SEC_PER_MIN * dur_min
    auto = res["timing_sec"]["total"]

    rows = []
    for i, g in enumerate(gt):
        m = next((x for x in matched if x[0] == i), None)
        if m:
            p = pred[m[1]]
            rows.append(f"| {i + 1} | {g['start']}–{g['end']}s | {g['reps']}회 | "
                        f"{p['start']}–{p['end']}s | {p['reps']}회 | {m[2]:.2f} | "
                        f"{'O' if abs(g['reps'] - p['reps']) <= REP_TOLERANCE else 'X'} |")
        else:
            rows.append(f"| {i + 1} | {g['start']}–{g['end']}s | {g['reps']}회 | "
                        f"미검출 | — | 0.00 | X |")

    md = f"""# 검증 결과 (자동 생성)

- 입력: `{res['source']}` / {res['video']['duration_sec']}초 / {res['video']['resolution']}
- 인물 검출률: {res['video']['person_detect_ratio'] * 100:.0f}%
- 분석: {res['video']['frames_analyzed']}프레임 @ {res['video']['analyzed_fps']}fps

## 1. 세트 구간 검출

| # | 정답 구간 | 정답 반복 | 검출 구간 | 검출 반복 | IoU | 반복 ±1 |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

- **재현율(recall)**: {recall:.2f} ({len(matched)}/{len(gt)})
- **정밀도(precision)**: {precision:.2f} ({len(matched)}/{len(pred)})
- **반복 수 정확도(±{REP_TOLERANCE}회)**: {rep_acc:.2f} ({rep_ok}/{len(gt)})
- **반복 수 완전일치**: {rep_exact}/{len(gt)}

## 2. 처리 시간

| 구분 | 시간 |
|---|---|
| 수작업 기준선 (추정 {MANUAL_SEC_PER_MIN:.0f}초/영상1분) | 약 {manual / 60:.1f}분 |
| PoC 자동 처리 | {auto:.1f}초 ({auto / 60:.1f}분) |
| **단축률** | **{(1 - auto / manual) * 100:.0f}%** |

단계별: {" / ".join(f"{k} {v}s" for k, v in res['timing_sec'].items() if k != 'total')}

## 3. 산출물

영상 1개 → 숏츠 {len(res['clips'])}개 + 카드뉴스 {sum(len(c['cards']) for c in res['clips'])}장 + 캡션 {len(res['clips'])}건
"""
    with open(out_md, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    a = sys.argv[1:] or ["data/sample_workout_truth.json", "outputs/sample/result.json"]
    main(*a[:3])
