"""
app.py — 데모 웹앱 (FastAPI).

영상 업로드 → 진행률 확인 → 숏츠·카드뉴스 결과 확인까지 한 화면에서.

실행:  python app.py       →  http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pipeline import process

OUT_ROOT = os.path.abspath("outputs")
UP_ROOT = os.path.abspath("uploads")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(UP_ROOT, exist_ok=True)

app = FastAPI(title="MoveClip")
app.mount("/files", StaticFiles(directory=OUT_ROOT), name="files")

JOBS: dict[str, dict] = {}


def run_job(job_id: str, video_path: str, top_k: int, do_stt: bool):
    job = JOBS[job_id]

    def progress(stage, pct):
        job["stage"] = stage
        job["pct"] = round(pct * 100)

    try:
        job["result"] = process(
            video_path, os.path.join(OUT_ROOT, job_id),
            top_k=top_k, do_stt=do_stt, progress=progress,
        )
        job["status"] = "done"
    except Exception as e:  # 실패해도 화면에 이유를 보여준다
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"


PAGE = """<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MoveClip — 운동 영상 자동 편집</title><style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#111113;color:#eee;font:16px/1.6 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:48px 20px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.5px}
.sub{color:#8b8b93;margin:0 0 36px}
.card{background:#19191c;border:1px solid #26262b;border-radius:16px;padding:26px;margin-bottom:20px}
label{display:block;font-size:14px;color:#a0a0a8;margin-bottom:8px}
input[type=file]{width:100%;padding:14px;background:#111;border:1px dashed #3a3a42;border-radius:10px;color:#ccc}
.row{display:flex;gap:14px;flex-wrap:wrap;margin-top:18px;align-items:end}
select,button{padding:12px 16px;border-radius:10px;border:1px solid #33333a;background:#1f1f24;color:#eee;font-size:15px}
button{background:#dcff5e;color:#15151a;border:0;font-weight:700;cursor:pointer;padding:13px 26px}
button:disabled{opacity:.5;cursor:default}
.bar{height:8px;background:#232329;border-radius:99px;overflow:hidden;margin:14px 0 8px}
.bar>i{display:block;height:100%;width:0;background:#dcff5e;transition:width .4s}
.meta{display:flex;gap:26px;flex-wrap:wrap;color:#8b8b93;font-size:14px;margin-top:6px}
.clip{display:flex;gap:20px;border-top:1px solid #26262b;padding:22px 0}
.clip video{width:190px;border-radius:12px;background:#000}
.tag{display:inline-block;background:#232329;border-radius:99px;padding:3px 11px;font-size:12.5px;color:#b9b9c2;margin:0 6px 6px 0}
.cards{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.cards img{width:86px;border-radius:8px;border:1px solid #2a2a31}
.cap{white-space:pre-wrap;background:#141417;border-radius:10px;padding:12px;font-size:13px;color:#9a9aa3;margin-top:10px;max-height:120px;overflow:auto}
a{color:#dcff5e}
small{color:#70707a}
</style></head><body><div class=wrap>
<h1>MoveClip</h1>
<p class=sub>운동 코칭 영상에서 숏츠 후보와 카드뉴스를 자동으로 뽑아냅니다.</p>
__BODY__
</div></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    body = """
<form class=card method=post action="/upload" enctype="multipart/form-data" onsubmit="this.querySelector('button').disabled=true;this.querySelector('button').textContent='업로드 중…'">
  <label>운동 영상 파일 (mp4 / mov)</label>
  <input type=file name=file accept="video/*" required>
  <div class=row>
    <div><label>뽑을 숏츠 개수</label>
      <select name=top_k><option>2</option><option selected>3</option><option>4</option></select></div>
    <div><label>음성인식(자막·카드뉴스)</label>
      <select name=stt><option value=1 selected>사용</option><option value=0>끄기 (빠름)</option></select></div>
    <button type=submit>분석 시작</button>
  </div>
</form>
<div class=card>
  <b>처리 흐름</b>
  <div class=meta style="margin-top:10px">
    <span>1. YOLO11-pose 자세 추출</span><span>2. 반복·세트 검출</span>
    <span>3. Whisper 음성인식</span><span>4. 하이라이트 선별</span><span>5. 9:16 렌더 + 카드뉴스</span>
  </div>
</div>"""
    return PAGE.replace("__BODY__", body)


@app.post("/upload")
async def upload(file: UploadFile = File(...),
                 top_k: int = Form(3), stt: int = Form(1)):
    job_id = uuid.uuid4().hex[:10]
    path = os.path.join(UP_ROOT, f"{job_id}_{file.filename}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"status": "running", "stage": "준비 중", "pct": 0, "name": file.filename}
    threading.Thread(target=run_job, args=(job_id, path, int(top_k), bool(int(stt))),
                     daemon=True).start()
    return RedirectResponse(f"/job/{job_id}", status_code=303)


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    return JSONResponse(JOBS.get(job_id, {"status": "unknown"}))


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return PAGE.replace("__BODY__", "<div class=card>없는 작업입니다. <a href='/'>처음으로</a></div>")

    if job["status"] == "running":
        body = f"""
<div class=card>
  <b>{job['name']}</b> 분석 중
  <div class=bar><i id=b></i></div>
  <div class=meta><span id=s>{job['stage']}</span><span id=p>{job['pct']}%</span></div>
</div>
<script>
setInterval(async()=>{{
  const r=await (await fetch('/api/job/{job_id}')).json();
  if(r.status!=='running'){{location.reload();return}}
  b.style.width=r.pct+'%'; s.textContent=r.stage; p.textContent=r.pct+'%';
}},1200);
</script>"""
        return PAGE.replace("__BODY__", body)

    if job["status"] == "error":
        return PAGE.replace("__BODY__",
                            f"<div class=card><b>처리 실패</b><div class=cap>{job['error']}</div>"
                            f"<div class=row><a href='/'>처음으로</a></div></div>")

    r = job["result"]
    v, t = r["video"], r["timing_sec"]
    parts = []
    parts.append(f"""
<div class=card>
  <b>{job['name']}</b>
  <div class=meta>
    <span>길이 {v['duration_sec']}초</span><span>{v['resolution']}</span>
    <span>분석 {v['frames_analyzed']}프레임 @{v['analyzed_fps']}fps</span>
    <span>인물 검출 {v['person_detect_ratio'] * 100:.0f}%</span>
    <span>세트 {len(r['sets_detected'])}개</span><span>발화 {r['utterances']}구간</span>
    <span><b style="color:#dcff5e">총 {t['total']}초</b> 처리</span>
  </div>
</div>""")

    clips = []
    for c in r["clips"]:
        cards = "".join(
            f"<a href='/files/{job_id}/{p}' target=_blank><img src='/files/{job_id}/{p}'></a>"
            for p in c["cards"])
        clips.append(f"""
<div class=clip>
  <video src="/files/{job_id}/{c['video']}" controls muted playsinline></video>
  <div style="flex:1;min-width:0">
    <b>#{c['rank']} · {c['start']}s ~ {c['end']}s ({c['duration']}초)</b>
    <div style="margin:8px 0">
      <span class=tag>{c['exercise']}</span><span class=tag>반복 {c['reps']}회</span>
      <span class=tag>점수 {c['score']}</span><span class=tag>기준신호 {c['signal']}</span>
    </div>
    <small>선정 이유 — {c['reason']}</small>
    <div class=cards>{cards}</div>
    <div class=cap>{c['caption']}</div>
  </div>
</div>""")
    parts.append(f"<div class=card><b>숏츠 후보 {len(r['clips'])}개</b>{''.join(clips)}</div>")
    parts.append(f"<div class=card><a href='/files/{job_id}/result.json' target=_blank>"
                 f"result.json 원본 보기</a> · <a href='/'>새 영상 분석</a></div>")
    return PAGE.replace("__BODY__", "".join(parts))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
