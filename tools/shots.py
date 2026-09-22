"""데모 화면 스크린샷 생성 (README 시연 자료용). python tools/shots.py"""

import os
import time

from playwright.sync_api import sync_playwright

OUT = "docs/images"
os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1000, "height": 900}, device_scale_factor=2)

    pg.goto("http://127.0.0.1:8000/")
    pg.wait_for_timeout(600)
    pg.screenshot(path=f"{OUT}/01_upload.png")

    pg.set_input_files("input[type=file]", "data/sample_workout.mp4")
    pg.click("button[type=submit]")
    pg.wait_for_url("**/job/**")
    pg.wait_for_timeout(9000)
    pg.screenshot(path=f"{OUT}/02_progress.png")

    for _ in range(90):
        if "분석 중" not in pg.content():
            break
        pg.wait_for_timeout(3000)
    pg.wait_for_timeout(3000)
    pg.screenshot(path=f"{OUT}/03_result.png", full_page=True)

    b.close()
print("saved:", os.listdir(OUT))
