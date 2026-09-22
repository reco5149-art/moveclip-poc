#!/usr/bin/env bash
# 모델 내려받기 (총 ~220MB). 한 번만 실행하면 됩니다.
set -e
cd "$(dirname "$0")"
mkdir -p models && cd models

Y=https://github.com/ultralytics/assets/releases/download/v8.3.0
K=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models

echo "[1/4] YOLO11n-pose (자세 추출, 6MB)"
[ -f yolo11n-pose.pt ] || curl -fsSL -o yolo11n-pose.pt "$Y/yolo11n-pose.pt"

echo "[2/4] YOLO11n-seg (합성 샘플 생성용, 6MB)"
[ -f yolo11n-seg.pt ] || curl -fsSL -o yolo11n-seg.pt "$Y/yolo11n-seg.pt"

echo "[3/4] Whisper base ONNX (음성인식, 200MB)"
[ -d sherpa-onnx-whisper-base ] || {
  curl -fsSL -o w.tar.bz2 "$K/sherpa-onnx-whisper-base.tar.bz2"
  tar xjf w.tar.bz2 && rm w.tar.bz2
}

echo "[4/4] Silero VAD (음성 구간 검출, 1.7MB)"
[ -f silero_vad.onnx ] || curl -fsSL -o silero_vad.onnx "$K/silero_vad.onnx"

# 한국어 자막 품질을 올리려면:  bash setup_models.sh --small   (1.3GB 추가)
# 실측 문자정확도  base 61%  →  small 71%  (docs/03_validation.md)
if [ "$1" = "--small" ]; then
  echo "[+] Whisper small ONNX (한국어 권장, 1.3GB)"
  [ -d sherpa-onnx-whisper-small ] || {
    curl -fsSL -o s.tar.bz2 "$K/sherpa-onnx-whisper-small.tar.bz2"
    tar xjf s.tar.bz2 && rm s.tar.bz2
  }
fi

echo "완료. $(du -sh . | cut -f1) 사용 중"
