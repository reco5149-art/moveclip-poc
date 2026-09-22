"""
main.py — CLI 실행기.

사용법:
    python main.py 영상.mp4                    # 결과는 outputs/영상이름/ 에 저장
    python main.py 영상.mp4 -o out -k 3        # 숏츠 3개
    python main.py 영상.mp4 --no-stt           # 음성인식 건너뛰기 (빠름)
"""

import argparse
import os
import sys
import time

from pipeline import process


def main():
    ap = argparse.ArgumentParser(description="운동 영상 → 숏츠·카드뉴스 자동 생성")
    ap.add_argument("video", help="입력 영상 경로")
    ap.add_argument("-o", "--out", default=None, help="출력 폴더 (기본: outputs/<영상이름>)")
    ap.add_argument("-k", "--top-k", type=int, default=3, help="뽑을 숏츠 개수 (기본 3)")
    ap.add_argument("--fps", type=float, default=5.0, help="분석 샘플링 fps (기본 5)")
    ap.add_argument("--lang", default="ko", help="음성인식 언어 (기본 ko)")
    ap.add_argument("--handle", default="@my_movement", help="카드뉴스에 넣을 계정명")
    ap.add_argument("--stt-model", choices=["base", "small"], default=None,
                    help="음성인식 모델 (기본: small이 있으면 small, 없으면 base)")
    ap.add_argument("--no-stt", action="store_true", help="음성인식 생략")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"영상을 찾을 수 없습니다: {args.video}")

    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out or os.path.join("outputs", name)

    last = [""]

    def progress(stage, pct):
        bar = "█" * int(pct * 28) + "░" * (28 - int(pct * 28))
        line = f"\r  [{bar}] {pct * 100:3.0f}%  {stage:<18}"
        sys.stdout.write(line)
        sys.stdout.flush()
        last[0] = stage

    print(f"\n▶ 입력: {args.video}")
    t = time.time()
    r = process(
        args.video, out_dir,
        top_k=args.top_k, sample_fps=args.fps, language=args.lang,
        handle=args.handle, do_stt=not args.no_stt, stt_model=args.stt_model,
        progress=progress,
    )
    print("\n")

    v = r["video"]
    print(f"  영상 길이   {v['duration_sec']}초 ({v['resolution']}, {v['src_fps']}fps)")
    print(f"  분석 프레임 {v['frames_analyzed']}장 @ {v['analyzed_fps']}fps"
          f" / 인물 검출률 {v['person_detect_ratio'] * 100:.0f}%")
    print(f"  세트 검출   {len(r['sets_detected'])}개 / 발화 {r['utterances']}구간")
    print()
    for c in r["clips"]:
        print(f"  #{c['rank']}  {c['start']:>6.1f}s ~ {c['end']:>6.1f}s "
              f"({c['duration']:>4.1f}초)  {c['exercise']}  반복 {c['reps']}회  "
              f"점수 {c['score']:.2f}  — {c['reason']}")
    print(f"\n  ⏱  총 {r['timing_sec']['total']}초 "
          f"(포즈 {r['timing_sec']['pose']}s / 반복검출 {r['timing_sec']['reps']}s / "
          f"음성인식 {r['timing_sec']['stt']}s / 렌더 {r['timing_sec']['render']}s)")
    print(f"  📁 결과: {os.path.abspath(out_dir)}\n")


if __name__ == "__main__":
    main()
