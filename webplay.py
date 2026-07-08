"""Browser playground for the Gemini API (local dev tool).

Run:
    GEMINI_API_KEY=... .venv/bin/python webplay.py
Then open http://localhost:5001 in your browser.

Lets you pick model / temperature / prompt, feed a census page image, and
optionally crop to a row band (using the corpus layout geometry) before sending
— so you can eyeball how Gemini reads the whole page vs a tight strip.
"""
import html
import io
import json
import os
from pathlib import Path

from flask import Flask, redirect, request
from google import genai
from google.genai import types
from PIL import Image

app = Flask(__name__)
CORPUS = Path("corpora/us_census_1850")
DEFAULT_IMAGE = str(
    CORPUS / "data/reels/populationschedu0604unix/populationschedu0604unix_0023.png"
)

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Gemini Playground</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222}}
 h1{{font-size:20px}} label{{display:block;font-weight:600;margin:12px 0 4px}}
 input[type=text],textarea,select{{width:100%;padding:8px;font-size:14px;box-sizing:border-box}}
 textarea{{font-family:ui-monospace,monospace}} .row{{display:flex;gap:16px}} .row>div{{flex:1}}
 button{{margin-top:16px;padding:10px 20px;font-size:15px;font-weight:600;cursor:pointer}}
 pre{{background:#f4f1ea;padding:16px;white-space:pre-wrap;word-wrap:break-word;border-radius:6px}}
 .meta{{color:#666;font-size:13px}}
 .spinner{{display:none;width:15px;height:15px;border:2px solid #ccc;border-top-color:#333;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-left:10px}}
 @keyframes spin{{to{{transform:rotate(360deg)}}}}
</style></head><body>
<h1>Gemini Playground &mdash; 1850 Census</h1>
<form method=post>
 <div class=row>
  <div><label>Model</label><select name=model>{model_opts}</select></div>
  <div><label>Temperature</label><input type=text name=temperature value="{temperature}"></div>
 </div>
 <label>System instruction</label><textarea name=system rows=3>{system}</textarea>
 <label>Prompt</label><textarea name=prompt rows=3>{prompt}</textarea>
 <label>Image path <span class=meta>(blank = no image)</span></label>
 <input type=text name=image value="{image}">
 <div class=row>
  <div><label>Crop rows from <span class=meta>(optional)</span></label><input type=text name=crop_from value="{crop_from}"></div>
  <div><label>to</label><input type=text name=crop_to value="{crop_to}"></div>
 </div>
 <label><input type=checkbox name=json_mode {json_checked}> Force JSON output</label>
 <button type=submit id=runbtn>Run</button>
 <span id=spin class=spinner></span>
 <span id=spinmsg class=meta style="display:none">Thinking&hellip; (Pro can take ~a minute)</span>
</form>
{result}
<script>
document.querySelector('form').addEventListener('submit',function(){{
 var b=document.getElementById('runbtn');
 b.disabled=true; b.textContent='Running…';
 document.getElementById('spin').style.display='inline-block';
 document.getElementById('spinmsg').style.display='inline';
}});
</script>
</body></html>"""


def model_options(selected):
    opts = ""
    for m in ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"):
        sel = " selected" if m == selected else ""
        opts += f'<option value="{m}"{sel}>{m}</option>'
    return opts


def crop_band(path, r0, r1):
    layout = json.loads((CORPUS / "config" / "layout.json").read_text())
    top, pitch, margin = layout["row1_top"], layout["row_pitch"], layout.get("crop_margin", 0)
    img = Image.open(path)
    W, H = img.size
    y0 = max(0, top + (r0 - 1) * pitch - margin)
    y1 = min(H, top + r1 * pitch + margin)
    buf = io.BytesIO()
    img.crop((0, y0, W, y1)).save(buf, format="PNG")
    return buf.getvalue()


def run_gemini(f):
    client = genai.Client()
    contents = []
    img_path = (f.get("image") or "").strip()
    if img_path:
        cf, ct = f.get("crop_from", "").strip(), f.get("crop_to", "").strip()
        if cf and ct:
            data = crop_band(img_path, int(cf), int(ct))
            contents.append(types.Part.from_bytes(data=data, mime_type="image/png"))
        else:
            size = os.path.getsize(img_path)
            if size > 15_000_000:
                contents.append(client.files.upload(file=img_path))
            else:
                mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
                contents.append(types.Part.from_bytes(data=open(img_path, "rb").read(), mime_type=mime))
    contents.append(f.get("prompt", ""))
    cfg = types.GenerateContentConfig(
        system_instruction=f.get("system") or None,
        temperature=float(f.get("temperature") or 0),
    )
    if f.get("json_mode"):
        cfg.response_mime_type = "application/json"
    resp = client.models.generate_content(model=f.get("model"), contents=contents, config=cfg)
    usage = getattr(resp, "usage_metadata", None)
    return resp.text or "(empty response)", usage


LAST = {}  # last run's inputs + rendered result (single-user local tool)


@app.route("/", methods=["GET", "POST"])
def index():
    global LAST
    if request.method == "POST":
        f = request.form.to_dict()
        try:
            text, usage = run_gemini(f)
            meta = f"<div class=meta>tokens: {usage}</div>" if usage else ""
            result = f"<h3>Response</h3>{meta}<pre>{html.escape(text)}</pre>"
        except Exception as e:
            result = f"<h3>Error</h3><pre>{html.escape(f'{type(e).__name__}: {e}')}</pre>"
        LAST = {"inputs": f, "result": result}
        return redirect("/")  # Post/Redirect/Get: reload re-shows, doesn't re-submit
    f = LAST.get("inputs", {})
    result = LAST.get("result", "")
    return PAGE.format(
        model_opts=model_options(f.get("model", "gemini-2.5-flash")),
        temperature=f.get("temperature", "0.0"),
        system=f.get("system", "You are an expert transcriber of 1850 US Census handwriting."),
        prompt=f.get("prompt", "Read the names on lines 38 and 39."),
        image=f.get("image", DEFAULT_IMAGE),
        crop_from=f.get("crop_from", ""),
        crop_to=f.get("crop_to", ""),
        json_checked="checked" if f.get("json_mode") else "",
        result=result,
    )


if __name__ == "__main__":
    host = os.environ.get("WEBPLAY_HOST", "127.0.0.1")
    print(f"Gemini playground -> http://{host}:5001")
    app.run(host=host, port=5001, debug=False)
