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

from flask import Flask, redirect, request, send_file
from google import genai
from google.genai import types
from PIL import Image

app = Flask(__name__)
CORPUS = Path("corpora/us_census_1850")
REEL = "populationschedu0604unix"
OUT = CORPUS / "output" / "rows" / REEL
DEFAULT_IMAGE = str(CORPUS / f"data/reels/{REEL}/{REEL}_0023.png")

# (field, short column header) for the output table
VIEW_COLS = [
    ("dwelling_number", "Dw"), ("family_number", "Fam"),
    ("interpreted_first_name", "First"), ("interpreted_last_name", "Last"),
    ("age", "Age"), ("sex", "Sex"), ("color", "Clr"), ("occupation", "Occupation"),
    ("real_estate_value", "R.Est"), ("place_of_birth", "Birthplace"),
    ("married_within_year", "Mar"), ("attended_school", "Sch"),
    ("cannot_read_write", "R/W"), ("infirmities", "Infirm"),
]

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
<p><a href="/view">View transcribed output</a> &middot; <a href="/pipeline_v2">pipeline_v2</a> &middot; Strategies: <a href="/strategies/s1">s1</a> &middot; <a href="/strategies/s2">s2</a> &middot; <a href="/strategies/s3">s3</a> &middot; <a href="/strategies/s4">s4</a> &middot; <a href="/strategies/s5">s5</a> &middot; <a href="/strategies/s6">s6</a> &middot; <a href="/strategies/s7">s7</a> &middot; <a href="/strategies/s8">s8</a> &middot; <a href="/strategies/s9">s9</a> &middot; <a href="/strategies/s10">s10</a> &middot; <a href="/strategies/s11">s11</a> &middot; <a href="/strategies/s12">s12</a> &middot; <a href="/strategies/s13">s13</a> &middot; <a href="/strategies/s14">s14</a> &middot; <a href="/strategies/s15">s15</a> &middot; <a href="/strategies/s16">s16</a> &middot; <a href="/strategies/s17">s17</a> &middot; <a href="/strategies/s18">s18</a> &middot; <a href="/strategies/s19">s19</a> &middot; <a href="/strategies/s20">s20</a> &middot; <a href="/strategies/s21">s21</a> &middot; <a href="/strategies/s22">s22</a> &middot; <a href="/strategies/s23">s23</a></p>
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
    for m in ("gemini-3.5-flash", "gemini-flash-latest", "gemini-pro-latest"):
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


VIEW_CSS = """<style>
 body{font-family:system-ui,sans-serif;margin:20px;color:#222}
 .wrap{max-width:1500px;margin:0 auto}
 h1{font-size:20px} .meta{color:#666;font-size:13px}
 .nav a{margin-right:10px;font-weight:600} .nav a.on{text-decoration:none;color:#000;background:#eee;padding:1px 6px;border-radius:4px}
 .tablewrap{overflow-x:auto;border:1px solid #eee;border-radius:6px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{border:1px solid #e2e2e2;padding:4px 7px;text-align:left;vertical-align:top;white-space:nowrap}
 th{background:#f0ece3;position:sticky;top:0}
 td.ln{font-weight:600} td.conflict{background:#fff3cd} tr.review{background:#fcfbf6}
 .conflict b{color:#9a5b00;font-weight:600} .miss{color:#b00;font-weight:normal;font-size:11px}
 .legend span{margin-right:14px}
</style>"""


def list_frames():
    if not OUT.exists():
        return []
    return sorted({p.name.split("_")[-1].split(".")[0]
                   for p in OUT.glob(f"{REEL}_*.consensus.json")})


def _cell(row, field):
    conflicts = row.get("conflicts", {})
    if field in conflicts:
        c = conflicts[field]
        inner = " / ".join(
            f'<b>{a[0].upper()}:</b>&nbsp;{html.escape(str(c.get(a)) if c.get(a) not in (None, "") else "—")}'
            for a in c)
        return f'<td class=conflict>{inner}</td>'
    v = row.get(field)
    return f'<td>{html.escape(str(v)) if v not in (None, "") else ""}</td>'


def render_view(frame):
    p = OUT / f"{REEL}_{int(frame):04d}.consensus.json"
    if not p.exists():
        return f"<p>No consensus file for frame {frame} yet.</p>"
    d = json.loads(p.read_text())
    s, m = d.get("summary", {}), d.get("metadata", {})
    head = (f"<h2>Frame {int(frame):04d} &mdash; {html.escape(str(m.get('location_town') or '?'))}, "
            f"{html.escape(str(m.get('location_county') or '?'))} County</h2>"
            f"<p class=meta>Consensus of {', '.join(d.get('agents', []))} &middot; "
            f"cell agreement {s.get('cells_agree')}/{s.get('cells_total')} "
            f"(<b>{s.get('cell_agreement_pct')}%</b>) &middot; "
            f"{s.get('rows_with_conflict')} rows need review</p>"
            "<p class='meta legend'><span>✓ = all cells agree</span>"
            "<span>⚑ = has a flagged cell</span>"
            "<span style='background:#fff3cd;padding:0 4px'>yellow = conflict (C: Claude / G: Gemini)</span></p>")
    ths = "".join(f"<th>{lbl}</th>" for _, lbl in VIEW_COLS)
    trs = ""
    for row in d["rows"]:
        conf = row.get("confidence", "")
        rc = row.get("conflicts", {})
        note = ""
        if "_row" in rc:
            present = [a for a, v in rc["_row"].items() if v == "present"]
            note = f" <span class=miss>(only {', '.join(present)})</span>"
        badge = "✓" if conf == "HIGH" else "⚑"
        tds = "".join(_cell(row, f) for f, _ in VIEW_COLS)
        cls = "" if conf == "HIGH" else " class=review"
        trs += f'<tr{cls}><td class=ln>{row["line_number"]} {badge}{note}</td>{tds}</tr>'
    return (head + f'<div class=tablewrap><table><thead><tr><th>Ln</th>{ths}</tr>'
            f'</thead><tbody>{trs}</tbody></table></div>')


@app.route("/view")
@app.route("/view/<int:frame>")
def view(frame=None):
    frames = list_frames()
    if not frames:
        return VIEW_CSS + "<div class=wrap><h1>Scriptorium Output</h1><p>No consensus files found. Run the pipeline + reconcile first.</p><p><a href='/'>&larr; API playground</a></p></div>"
    if frame is None:
        frame = int(frames[0])
    nav = " ".join(
        f'<a class="{"on" if int(fr) == int(frame) else ""}" href="/view/{int(fr)}">{fr}</a>'
        for fr in frames)
    body = render_view(frame)
    return (VIEW_CSS + "<div class=wrap><h1>Scriptorium Output</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; "
            "<a href='/pipeline_v2'>pipeline_v2</a> &middot; "
            "Strategies: <a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; <a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; <a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; <a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; <a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a> &middot; <a href='/strategies/s15'>s15</a></p>"
            f"<p class=nav>Pages: {nav}</p>{body}</div>")


S1_DIR = Path("src/strategies/_s1")
S2_DIR = Path("src/strategies/_s2")
S3_DIR = Path("src/strategies/_s3")
S4_DIR = Path("src/strategies/_s4")
S5_DIR = Path("src/strategies/_s5")
S6_DIR = Path("src/strategies/_s6")
S7_DIR = Path("src/strategies/_s7")
S8_DIR = Path("src/strategies/_s8")
S9_DIR = Path("src/strategies/_s9")
S10_DIR = Path("src/strategies/_s10")
S11_DIR = Path("src/strategies/_s11")
S12_DIR = Path("src/strategies/_s12")
S13_DIR = Path("src/strategies/_s13")
S14_DIR = Path("src/strategies/_s14")
S15_DIR = Path("src/strategies/_s15")
S16_DIR = Path("src/strategies/_s16")
S17_DIR = Path("src/strategies/_s17")
S18_DIR = Path("src/strategies/_s18")
S19_DIR = Path("src/strategies/_s19")
S20_DIR = Path("src/strategies/_s20")
S21_DIR = Path("src/strategies/_s21")
S22_DIR = Path("src/strategies/_s22")
S23_DIR = Path("src/strategies/_s23")
GROUND_TRUTH = Path("src/strategies/ground_truth.json")


def _fresh_truth() -> dict:
    """Load ground_truth.json fresh (not the copy baked into results.json).

    Returns {(line, field): correct_str_or_None}. Viewer uses this so edits to
    ground_truth.json show up on refresh without re-running the strategy.
    """
    d = json.loads(GROUND_TRUTH.read_text())
    return {(c["line"], c["field"]): c["correct"] for c in d["cases"]}


def _truth_first(t: dict, line: int) -> str:
    return t.get((line, "interpreted_first_name")) or ""


def _truth_last(t: dict, line: int) -> str:
    return t.get((line, "interpreted_last_name")) or ""


def _truth_combined(t: dict, line: int) -> str:
    parts = [p for p in (_truth_first(t, line), _truth_last(t, line)) if p]
    return " ".join(parts) if parts else ""


def _match(a: str, b: str) -> bool:
    """Loose match: casefold, strip ALL whitespace ('E.P.' == 'E. P.'), drop [DITTO]."""
    def norm(v):
        s = (v or "").casefold().replace("[ditto]", "")
        return "".join(s.split())
    return norm(a) == norm(b) and norm(a) != ""


@app.route("/strategies/s1/crop/<name>")
def s1_crop(name):
    p = S1_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s1")
def s1_view():
    p = S1_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s1</h1><p>No results.json yet — run <code>src/strategies/s1.py</code>.</p></div>"
    d = json.loads(p.read_text())
    truth_map = _fresh_truth()
    rows = ""
    ctally = gtally = total = 0
    for r in d["results"]:
        truth = _truth_combined(truth_map, r["line"]) or r.get("truth", "")
        c_name, g_name = r["claude"]["name"], r["gemini"]["name"]
        c_conf, g_conf = r["claude"]["confidence"], r["gemini"]["confidence"]
        c_ok = _match(c_name, truth); g_ok = _match(g_name, truth)
        agree = _match(c_name, g_name)
        total += 1
        ctally += c_ok; gtally += g_ok
        c_cls = "pass" if c_ok else "fail"
        g_cls = "pass" if g_ok else "fail"
        agree_badge = "<span class=agree>agree</span>" if agree else "<span class=disagree>disagree</span>"
        rows += (f"<tr><td class=ln>L{r['line']:02d}</td>"
                 f"<td><img src='/strategies/s1/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                 f"<td class={c_cls}>{html.escape(c_name)}<div class=meta>{c_conf}</div></td>"
                 f"<td class={g_cls}>{html.escape(g_name)}<div class=meta>{g_conf}</div></td>"
                 f"<td class=truth>{html.escape(truth)}</td>"
                 f"<td>{agree_badge}</td></tr>")
    head = (f"<h2>s1 — combined-name read (no adjudication)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · Claude {ctally}/{total} · "
            f"Gemini {gtally}/{total} · <a href='/view/{d['frame']}'>consensus view →</a></p>"
            f"<details><summary class=meta>prompt</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px'>{html.escape(d['prompt'])}</pre></details>")
    css = VIEW_CSS + """<style>
     .s1 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s1 td.pass{background:#e6f5d9}
     .s1 td.fail{background:#fbe6e6}
     .s1 td.truth{background:#eef0f5;font-weight:600}
     .s1 td .meta{font-size:11px;color:#666}
     .agree{color:#3a7a1a;font-weight:600}
     .disagree{color:#a03000;font-weight:600}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1' style='font-weight:600'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; <a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; <a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; <a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; <a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s1'><table><thead><tr>"
            "<th>Ln</th><th>Crop</th><th>Claude</th><th>Gemini</th><th>Ground truth</th><th>Agree?</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s2/crop/<name>")
def s2_crop(name):
    p = S2_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s2")
def s2_view():
    p = S2_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s2</h1><p>No results.json yet — run <code>src/strategies/s2.py</code>.</p></div>"
    d = json.loads(p.read_text())
    truth_map = _fresh_truth()
    rows = ""
    cf_pass = cl_pass = gf_pass = gl_pass = tot_first = tot_last = 0
    for r in d["results"]:
        tf = _truth_first(truth_map, r["line"])
        tl = _truth_last(truth_map, r["line"])
        cf, cl = r["claude"]["first"], r["claude"]["last"]
        gf, gl = r["gemini"]["first"], r["gemini"]["last"]

        def cell(read, expected, count_key=None):
            if not expected:
                return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
            ok = _match(read["name"], expected)
            cls = "pass" if ok else "fail"
            return (f"<td class={cls}>{html.escape(read['name'])}"
                    f"<div class=meta>{read['confidence']}</div></td>"), ok

        c1, ok = cell(cf, tf); cf_pass += ok; tot_first += (1 if tf else 0)
        c2, ok = cell(cl, tl); cl_pass += ok; tot_last += (1 if tl else 0)
        g1, ok = cell(gf, tf); gf_pass += ok
        g2, ok = cell(gl, tl); gl_pass += ok
        rows += (f"<tr><td class=ln>L{r['line']:02d}</td>"
                 f"<td><img src='/strategies/s2/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                 f"{c1}{c2}{g1}{g2}"
                 f"<td class=truth>{html.escape(tf) or '—'}</td>"
                 f"<td class=truth>{html.escape(tl) or '—'}</td></tr>")
    head = (f"<h2>s2 — split-name read (no adjudication)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"Claude first {cf_pass}/{tot_first} · last {cl_pass}/{tot_last} · "
            f"Gemini first {gf_pass}/{tot_first} · last {gl_pass}/{tot_last} · "
            f"<a href='/strategies/s1'>s1 →</a></p>"
            f"<details><summary class=meta>prompts</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px'>FIRST:\n{html.escape(d['first_prompt'])}\n\nLAST:\n{html.escape(d['last_prompt'])}</pre></details>")
    css = VIEW_CSS + """<style>
     .s2 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s2 td.pass{background:#e6f5d9}
     .s2 td.fail{background:#fbe6e6}
     .s2 td.nogt{background:#f6f6f6;color:#666}
     .s2 td.truth{background:#eef0f5;font-weight:600}
     .s2 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2' style='font-weight:600'>s2</a> &middot; <a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; <a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; <a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; <a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s2'><table><thead><tr>"
            "<th>Ln</th><th>Crop</th>"
            "<th>Claude first</th><th>Claude last</th>"
            "<th>Gemini first</th><th>Gemini last</th>"
            "<th>Truth first</th><th>Truth last</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s3/crop/<name>")
def s3_crop(name):
    p = S3_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s3")
def s3_view():
    p = S3_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s3</h1><p>No results.json yet — run <code>src/strategies/s3.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}
    move_class = {"flipped": "flipped", "kept": "kept", "drifted": "drifted"}
    rows = ""
    c_pass = g_pass = tot = 0
    for r in d["results"]:
        truth = tm.get((r["line"], field_key[r["field"]])) or ""
        c_orig, c_saw = r["claude"]["orig"], r["claude"]["saw"]
        c_new = r["claude"]["new"]["name"]; c_conf = r["claude"]["new"]["confidence"]
        c_move = r["claude"]["move"]
        g_orig, g_saw = r["gemini"]["orig"], r["gemini"]["saw"]
        g_new = r["gemini"]["new"]["name"]; g_conf = r["gemini"]["new"]["confidence"]
        g_move = r["gemini"]["move"]
        if truth:
            tot += 1
            c_ok = _match(c_new, truth); g_ok = _match(g_new, truth)
            c_pass += c_ok; g_pass += g_ok
        else:
            c_ok = g_ok = False
        def cell(orig, saw, new, conf, move, ok, have_truth):
            cls = ("pass" if ok else "fail") if have_truth else "nogt"
            return (f"<td class={cls}>"
                    f"<div class=orig>{html.escape(orig)}</div>"
                    f"<div class=saw>saw &ldquo;{html.escape(saw)}&rdquo;</div>"
                    f"<div class=new>&rarr; {html.escape(new)}</div>"
                    f"<div class=meta>{conf} &middot; <span class={move_class[move]}>{move}</span></div>"
                    f"</td>")
        rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{r['field']}</div></td>"
                 f"<td><img src='/strategies/s3/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                 f"{cell(c_orig, c_saw, c_new, c_conf, c_move, c_ok, bool(truth))}"
                 f"{cell(g_orig, g_saw, g_new, g_conf, g_move, g_ok, bool(truth))}"
                 f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")
    head = (f"<h2>s3 — neutral cross-model nudge (observational, no winner picked)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} &middot; "
            f"disagreements-only from s2 &middot; Claude re-read {c_pass}/{tot} &middot; "
            f"Gemini re-read {g_pass}/{tot}</p>"
            f"<details><summary class=meta>nudge templates</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px'>FIRST:\n{html.escape(d['first_nudge_template'])}\n\nLAST:\n{html.escape(d['last_nudge_template'])}</pre></details>")
    css = VIEW_CSS + """<style>
     .s3 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s3 td.pass{background:#e6f5d9} .s3 td.fail{background:#fbe6e6}
     .s3 td.nogt{background:#f6f6f6;color:#333}
     .s3 td.truth{background:#eef0f5;font-weight:600}
     .s3 .orig{font-size:12px;color:#666}
     .s3 .saw{font-size:11px;color:#8a5a00;font-style:italic;margin:2px 0}
     .s3 .new{font-size:14px;font-weight:600;margin-top:2px}
     .s3 .meta{font-size:11px;color:#666;margin-top:2px}
     .flipped{color:#a03000;font-weight:600}
     .kept{color:#3a7a1a;font-weight:600}
     .drifted{color:#7a3a7a;font-weight:600}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3' style='font-weight:600'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; <a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; <a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; <a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s3'><table><thead><tr>"
            "<th>Ln / field</th><th>Crop</th>"
            "<th>Claude (orig &rarr; saw other &rarr; re-read)</th>"
            "<th>Gemini (orig &rarr; saw other &rarr; re-read)</th>"
            "<th>Truth</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s4/crop/<name>")
def s4_crop(name):
    p = S4_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s4")
def s4_view():
    p = S4_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s4</h1><p>No results.json yet — run <code>src/strategies/s4.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    tallies = {"claude": {"first": 0, "last": 0}, "gemini_pro": {"first": 0, "last": 0}}
    tot_first = tot_last = 0
    rows = ""
    for r in d["results"]:
        tf = _truth_first(tm, r["line"]); tl = _truth_last(tm, r["line"])
        if tf: tot_first += 1
        if tl: tot_last += 1

        def cell(read, expected):
            if not expected:
                return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
            ok = _match(read["name"], expected)
            cls = "pass" if ok else "fail"
            return (f"<td class={cls}>{html.escape(read['name'])}"
                    f"<div class=meta>{read['confidence']}</div></td>"), ok
        cells_html = ""
        for m in ("claude", "gemini_pro"):
            c1, ok1 = cell(r[m]["first"], tf); tallies[m]["first"] += ok1
            c2, ok2 = cell(r[m]["last"],  tl); tallies[m]["last"]  += ok2
            cells_html += c1 + c2
        rows += (f"<tr><td class=ln>L{r['line']:02d}</td>"
                 f"<td><img src='/strategies/s4/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                 f"{cells_html}"
                 f"<td class=truth>{html.escape(tf) or '—'}</td>"
                 f"<td class=truth>{html.escape(tl) or '—'}</td></tr>")
    mm = d.get("models", {})
    head = (f"<h2>s4 — split-name read (Claude + Gemini Pro), no adjudication</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"Claude first {tallies['claude']['first']}/{tot_first} · last {tallies['claude']['last']}/{tot_last} · "
            f"Gemini-Pro first {tallies['gemini_pro']['first']}/{tot_first} · last {tallies['gemini_pro']['last']}/{tot_last}</p>"
            f"<p class=meta>models: claude={html.escape(str(mm.get('claude','')))} · "
            f"gemini-pro={html.escape(str(mm.get('gemini_pro','')))}</p>")
    css = VIEW_CSS + """<style>
     .s4 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s4 td.pass{background:#e6f5d9} .s4 td.fail{background:#fbe6e6}
     .s4 td.nogt{background:#f6f6f6;color:#666}
     .s4 td.truth{background:#eef0f5;font-weight:600}
     .s4 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4' style='font-weight:600'>s4</a> &middot; <a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; <a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; <a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s4'><table><thead><tr>"
            "<th>Ln</th><th>Crop</th>"
            "<th>Claude first</th><th>Claude last</th>"
            "<th>Gemini-Pro first</th><th>Gemini-Pro last</th>"
            "<th>Truth first</th><th>Truth last</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s5/crop/<name>")
def s5_crop(name):
    p = S5_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s5")
def s5_view():
    p = S5_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s5</h1><p>No results.json yet — run <code>src/strategies/s5.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    models = ("claude", "gemini_flash", "grok")
    labels = {"claude": "Claude", "gemini_flash": "Gemini Flash", "grok": "Grok"}
    tallies = {m: {"first": 0, "last": 0} for m in models}
    tot_first = tot_last = 0
    rows = ""
    for r in d["results"]:
        tf = _truth_first(tm, r["line"]); tl = _truth_last(tm, r["line"])
        if tf: tot_first += 1
        if tl: tot_last += 1

        def cell(read, expected):
            if not expected:
                return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
            ok = _match(read["name"], expected)
            cls = "pass" if ok else "fail"
            return (f"<td class={cls}>{html.escape(read['name'])}"
                    f"<div class=meta>{read['confidence']}</div></td>"), ok
        cells_html = ""
        for m in models:
            c1, ok1 = cell(r[m]["first"], tf); tallies[m]["first"] += ok1
            c2, ok2 = cell(r[m]["last"],  tl); tallies[m]["last"]  += ok2
            cells_html += c1 + c2
        rows += (f"<tr><td class=ln>L{r['line']:02d}</td>"
                 f"<td><img src='/strategies/s5/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                 f"{cells_html}"
                 f"<td class=truth>{html.escape(tf) or '—'}</td>"
                 f"<td class=truth>{html.escape(tl) or '—'}</td></tr>")
    mm = d.get("models", {})
    tally_line = " · ".join(
        f"{labels[m]} {tallies[m]['first']}/{tot_first} + {tallies[m]['last']}/{tot_last}"
        for m in models)
    head = (f"<h2>s5 — split-name read, three models (Claude + Gemini Flash + Grok)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · {tally_line}</p>"
            f"<p class=meta>models: "
            f"claude={html.escape(str(mm.get('claude','')))} · "
            f"flash={html.escape(str(mm.get('gemini_flash','')))} · "
            f"grok={html.escape(str(mm.get('grok','')))}</p>")
    css = VIEW_CSS + """<style>
     .s5 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s5 td.pass{background:#e6f5d9} .s5 td.fail{background:#fbe6e6}
     .s5 td.nogt{background:#f6f6f6;color:#666}
     .s5 td.truth{background:#eef0f5;font-weight:600}
     .s5 td .meta{font-size:11px;color:#666}
     .s5 th.grp{background:#e6ded0}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5' style='font-weight:600'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; <a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; <a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s5'><table><thead>"
            "<tr><th rowspan=2>Ln</th><th rowspan=2>Crop</th>"
            "<th class=grp colspan=2>Claude</th>"
            "<th class=grp colspan=2>Gemini Flash</th>"
            "<th class=grp colspan=2>Grok</th>"
            "<th rowspan=2>Truth first</th><th rowspan=2>Truth last</th></tr>"
            "<tr><th>first</th><th>last</th><th>first</th><th>last</th><th>first</th><th>last</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s6/crop/<name>")
def s6_crop(name):
    p = S6_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s6")
def s6_view():
    p = S6_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s6</h1><p>No results.json yet — run <code>src/strategies/s6.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    models = ("claude", "gemini_flash", "openai")
    labels = {"claude": "Claude", "gemini_flash": "Gemini Flash", "openai": "OpenAI"}
    tallies = {m: {"first": 0, "last": 0} for m in models}
    tot_first = tot_last = 0
    rows = ""
    for r in d["results"]:
        tf = _truth_first(tm, r["line"]); tl = _truth_last(tm, r["line"])
        if tf: tot_first += 1
        if tl: tot_last += 1

        def cell(read, expected):
            if not expected:
                return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
            ok = _match(read["name"], expected)
            cls = "pass" if ok else "fail"
            return (f"<td class={cls}>{html.escape(read['name'])}"
                    f"<div class=meta>{read['confidence']}</div></td>"), ok
        cells_html = ""
        for m in models:
            c1, ok1 = cell(r[m]["first"], tf); tallies[m]["first"] += ok1
            c2, ok2 = cell(r[m]["last"],  tl); tallies[m]["last"]  += ok2
            cells_html += c1 + c2
        rows += (f"<tr><td class=ln>L{r['line']:02d}</td>"
                 f"<td><img src='/strategies/s6/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                 f"{cells_html}"
                 f"<td class=truth>{html.escape(tf) or '—'}</td>"
                 f"<td class=truth>{html.escape(tl) or '—'}</td></tr>")
    mm = d.get("models", {})
    tally_line = " · ".join(
        f"{labels[m]} {tallies[m]['first']}/{tot_first} + {tallies[m]['last']}/{tot_last}"
        for m in models)
    head = (f"<h2>s6 — split-name read, three models (Claude + Gemini Flash + OpenAI)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · {tally_line}</p>"
            f"<p class=meta>models: "
            f"claude={html.escape(str(mm.get('claude','')))} · "
            f"flash={html.escape(str(mm.get('gemini_flash','')))} · "
            f"openai={html.escape(str(mm.get('openai','')))}</p>")
    css = VIEW_CSS + """<style>
     .s6 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s6 td.pass{background:#e6f5d9} .s6 td.fail{background:#fbe6e6}
     .s6 td.nogt{background:#f6f6f6;color:#666}
     .s6 td.truth{background:#eef0f5;font-weight:600}
     .s6 td .meta{font-size:11px;color:#666}
     .s6 th.grp{background:#e6ded0}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6' style='font-weight:600'>s6</a> &middot; <a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; <a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s6'><table><thead>"
            "<tr><th rowspan=2>Ln</th><th rowspan=2>Crop</th>"
            "<th class=grp colspan=2>Claude</th>"
            "<th class=grp colspan=2>Gemini Flash</th>"
            "<th class=grp colspan=2>OpenAI</th>"
            "<th rowspan=2>Truth first</th><th rowspan=2>Truth last</th></tr>"
            "<tr><th>first</th><th>last</th><th>first</th><th>last</th><th>first</th><th>last</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s7/crop/<name>")
def s7_crop(name):
    p = S7_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s7")
def s7_view():
    p = S7_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s7</h1><p>No results.json yet — run <code>src/strategies/s7.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    tallies = {"claude": {"forward": 0, "backward": 0},
               "gemini_flash": {"forward": 0, "backward": 0}}
    tot = 0
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}
    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""

            def cell(read):
                letters = read.get("letters") or []
                letter_details = ""
                if letters:
                    items = "".join(
                        f"<li><b>{html.escape(str(L.get('letter','?')))}</b> "
                        f"— {html.escape(str(L.get('structure','')))}</li>"
                        for L in letters
                    )
                    letter_details = (f"<details><summary>{len(letters)} letters</summary>"
                                      f"<ol style='padding-left:18px;margin:4px 0'>{items}</ol></details>")
                if truth:
                    ok = _match(read["name"], truth)
                    cls = "pass" if ok else "fail"
                    ok_val = ok
                else:
                    cls = "nogt"; ok_val = False
                return (f"<td class={cls}>{html.escape(read['name'])}"
                        f"<div class=meta>{read['confidence']}</div>"
                        f"{letter_details}</td>"), ok_val

            cf_html, cf_ok = cell(fd["claude"]["forward"]);        tallies["claude"]["forward"]        += cf_ok
            cb_html, cb_ok = cell(fd["claude"]["backward"]);       tallies["claude"]["backward"]       += cb_ok
            gf_html, gf_ok = cell(fd["gemini_flash"]["forward"]);  tallies["gemini_flash"]["forward"]  += gf_ok
            gb_html, gb_ok = cell(fd["gemini_flash"]["backward"]); tallies["gemini_flash"]["backward"] += gb_ok
            if truth:
                tot += 1
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s7/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{cf_html}{cb_html}{gf_html}{gb_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")
    head = (f"<h2>s7 — split-name read, grapheme decomposition (forward + backward)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"Claude fwd {tallies['claude']['forward']}/{tot} · "
            f"Claude bwd {tallies['claude']['backward']}/{tot} · "
            f"Gemini fwd {tallies['gemini_flash']['forward']}/{tot} · "
            f"Gemini bwd {tallies['gemini_flash']['backward']}/{tot}</p>")
    css = VIEW_CSS + """<style>
     .s7 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s7 td.pass{background:#e6f5d9} .s7 td.fail{background:#fbe6e6}
     .s7 td.nogt{background:#f6f6f6;color:#333}
     .s7 td.truth{background:#eef0f5;font-weight:600}
     .s7 td .meta{font-size:11px;color:#666}
     .s7 details{font-size:11px;color:#444;margin-top:4px}
     .s7 details summary{cursor:pointer;color:#666}
     .s7 th.grp{background:#e6ded0}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7' style='font-weight:600'>s7</a> &middot; "
            "<a href='/strategies/s8'>s8</a> &middot; <a href='/strategies/s9'>s9</a> &middot; "
            "<a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; "
            "<a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s7'><table><thead>"
            "<tr><th rowspan=2>Ln / field</th><th rowspan=2>Crop</th>"
            "<th class=grp colspan=2>Claude</th>"
            "<th class=grp colspan=2>Gemini Flash</th>"
            "<th rowspan=2>Truth</th></tr>"
            "<tr><th>forward</th><th>backward</th><th>forward</th><th>backward</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s8/crop/<name>")
def s8_crop(name):
    p = S8_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s8")
def s8_view():
    p = S8_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s8</h1><p>No results.json yet — run <code>src/strategies/s8.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    tallies = {"claude": {"original": 0, "decomposition": 0},
               "gemini_flash": {"original": 0, "decomposition": 0}}
    changed = {"claude": 0, "gemini_flash": 0}  # cells where decomp differed from orig
    tot = 0
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}
    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""

            def cell(read):
                letters = read.get("letters") or []
                details = ""
                if letters:
                    items = "".join(
                        f"<li><b>{html.escape(str(L.get('letter','?')))}</b> "
                        f"— {html.escape(str(L.get('structure','')))}</li>"
                        for L in letters
                    )
                    details = (f"<details><summary>{len(letters)} letters</summary>"
                               f"<ol style='padding-left:18px;margin:4px 0'>{items}</ol></details>")
                if truth:
                    ok = _match(read["name"], truth)
                    cls = "pass" if ok else "fail"
                    ok_val = ok
                else:
                    cls = "nogt"; ok_val = False
                return (f"<td class={cls}>{html.escape(read['name'])}"
                        f"<div class=meta>{read['confidence']}</div>"
                        f"{details}</td>"), ok_val

            co_html, co_ok = cell(fd["claude"]["original"]);      tallies["claude"]["original"]      += co_ok
            cd_html, cd_ok = cell(fd["claude"]["decomposition"]); tallies["claude"]["decomposition"] += cd_ok
            go_html, go_ok = cell(fd["gemini_flash"]["original"]);      tallies["gemini_flash"]["original"]      += go_ok
            gd_html, gd_ok = cell(fd["gemini_flash"]["decomposition"]); tallies["gemini_flash"]["decomposition"] += gd_ok
            if _match(fd["claude"]["original"]["name"], fd["claude"]["decomposition"]["name"]) is False:
                changed["claude"] += 1
            if _match(fd["gemini_flash"]["original"]["name"], fd["gemini_flash"]["decomposition"]["name"]) is False:
                changed["gemini_flash"] += 1
            if truth:
                tot += 1
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s8/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{co_html}{cd_html}{go_html}{gd_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")
    head = (f"<h2>s8 — original prompt + decomposition prompt (per model)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"Claude orig {tallies['claude']['original']}/{tot} · "
            f"Claude decomp {tallies['claude']['decomposition']}/{tot} · "
            f"Gemini orig {tallies['gemini_flash']['original']}/{tot} · "
            f"Gemini decomp {tallies['gemini_flash']['decomposition']}/{tot}</p>"
            f"<p class=meta>cells where decomp differs from original: "
            f"Claude {changed['claude']}/16 · Gemini {changed['gemini_flash']}/16</p>")
    css = VIEW_CSS + """<style>
     .s8 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s8 td.pass{background:#e6f5d9} .s8 td.fail{background:#fbe6e6}
     .s8 td.nogt{background:#f6f6f6;color:#333}
     .s8 td.truth{background:#eef0f5;font-weight:600}
     .s8 td .meta{font-size:11px;color:#666}
     .s8 details{font-size:11px;color:#444;margin-top:4px}
     .s8 details summary{cursor:pointer;color:#666}
     .s8 th.grp{background:#e6ded0}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8' style='font-weight:600'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s8'><table><thead>"
            "<tr><th rowspan=2>Ln / field</th><th rowspan=2>Crop</th>"
            "<th class=grp colspan=2>Claude</th>"
            "<th class=grp colspan=2>Gemini Flash</th>"
            "<th rowspan=2>Truth</th></tr>"
            "<tr><th>original</th><th>decomp</th><th>original</th><th>decomp</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s9/crop/<name>")
def s9_crop(name):
    p = S9_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s9")
def s9_view():
    p = S9_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s9</h1><p>No results.json yet — run <code>src/strategies/s9.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    def truth_counts(name: str):
        if not name or name.strip().upper() == "[DITTO]":
            return 0, 0
        letters = [c for c in name if c.isalpha()]
        return len(letters), len({c.lower() for c in letters})

    rows = ""
    both_agree = only_tot_agree = neither = 0
    scoreable = 0
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            t_tot, t_uniq = truth_counts(truth)
            c, g = fd["claude"], fd["gemini_flash"]

            def counts_cell(rec, t_t, t_u, have_truth):
                tot, uniq = rec.get("total_letters"), rec.get("unique_letters")
                if not have_truth:
                    return (f"<td class=nogt>{tot} / {uniq}"
                            f"<div class=meta>{rec.get('confidence','?')}</div></td>")
                cls = ("pass" if tot == t_t and uniq == t_u else
                       "partial" if tot == t_t or uniq == t_u else "fail")
                return (f"<td class={cls}>{tot} / {uniq}"
                        f"<div class=meta>{rec.get('confidence','?')}</div></td>")

            have_truth = bool(truth)
            c_cell = counts_cell(c, t_tot, t_uniq, have_truth)
            g_cell = counts_cell(g, t_tot, t_uniq, have_truth)

            if have_truth:
                scoreable += 1
                c_ok = c.get("total_letters") == t_tot and c.get("unique_letters") == t_uniq
                g_ok = g.get("total_letters") == t_tot and g.get("unique_letters") == t_uniq
                if c.get("total_letters") == g.get("total_letters") and c.get("unique_letters") == g.get("unique_letters"):
                    both_agree += 1
                elif c.get("total_letters") == g.get("total_letters"):
                    only_tot_agree += 1
                else:
                    neither += 1

            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s9/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_cell}{g_cell}"
                     f"<td class=truth>{html.escape(truth) or '—'}<div class=meta>{t_tot} / {t_uniq}</div></td></tr>")

    head = (f"<h2>s9 — letter-count read (no name transcription)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · counts shown as "
            f"<b>total / unique</b> · truth column shows both the ground-truth name "
            f"and its computed letter counts</p>"
            f"<p class=meta>agreement between the two models on scoreable cells: "
            f"both counts match {both_agree}/{scoreable} · only total matches "
            f"{only_tot_agree}/{scoreable} · both diverge {neither}/{scoreable}</p>")
    css = VIEW_CSS + """<style>
     .s9 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s9 td.pass{background:#e6f5d9}
     .s9 td.partial{background:#faf1cd}
     .s9 td.fail{background:#fbe6e6}
     .s9 td.nogt{background:#f6f6f6;color:#333}
     .s9 td.truth{background:#eef0f5;font-weight:600}
     .s9 td .meta{font-size:11px;color:#666;font-weight:normal}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9' style='font-weight:600'>s9</a> &middot; "
            "<a href='/strategies/s10'>s10</a> &middot; <a href='/strategies/s11'>s11</a> &middot; "
            "<a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s9'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude tot / uniq</th><th>Gemini tot / uniq</th>"
            "<th>Truth (name + counts)</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s10/crop/<name>")
def s10_crop(name):
    p = S10_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s10")
def s10_view():
    p = S10_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s10</h1><p>No results.json yet — run <code>src/strategies/s10.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    def truth_counts(name: str):
        if not name or name.strip().upper() == "[DITTO]":
            return 0, 0
        letters = [c for c in name if c.isalpha()]
        return len(letters), len({c.lower() for c in letters})

    tallies = {"claude": 0, "gemini_flash": 0}
    self_ok = {"claude": 0, "gemini_flash": 0}
    tot = 0; count_agree = 0
    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            t_tot, t_uniq = truth_counts(truth)
            c, g = fd["claude"], fd["gemini_flash"]
            # counts agreement (independent of truth)
            counts_match = (c.get("total_letters") == g.get("total_letters")
                            and c.get("unique_letters") == g.get("unique_letters"))
            if truth:
                tot += 1
                if _match(c["name"], truth): tallies["claude"] += 1
                if _match(g["name"], truth): tallies["gemini_flash"] += 1
            if c.get("self_consistent"): self_ok["claude"] += 1
            if g.get("self_consistent"): self_ok["gemini_flash"] += 1
            if counts_match: count_agree += 1

            def cell(rec, t_t, t_u, have_truth):
                nm = rec["name"]
                if have_truth:
                    ok = _match(nm, truth)
                    name_cls = "pass" if ok else "fail"
                else:
                    name_cls = "nogt"
                tot_v = rec.get("total_letters"); uniq_v = rec.get("unique_letters")
                count_note = f"{tot_v}/{uniq_v}"
                if have_truth:
                    count_match_truth = (tot_v == t_t and uniq_v == t_u)
                    count_note += f" · truth {t_t}/{t_u}{' ✓' if count_match_truth else ' ✗'}"
                self_flag = "self-consistent" if rec.get("self_consistent") else "SELF-INCONSISTENT"
                self_cls = "sok" if rec.get("self_consistent") else "sbad"
                return (f"<td class={name_cls}>{html.escape(nm)}"
                        f"<div class=meta>{count_note}</div>"
                        f"<div class='meta {self_cls}'>{self_flag} · {rec.get('confidence','?')}</div></td>")

            have_truth = bool(truth)
            c_html = cell(c, t_tot, t_uniq, have_truth)
            g_html = cell(g, t_tot, t_uniq, have_truth)
            agree_badge = ("<span class=agree>counts agree</span>"
                           if counts_match
                           else "<span class=disagree>counts differ</span>")

            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div>"
                     f"<div class=meta>{agree_badge}</div></td>"
                     f"<td><img src='/strategies/s10/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_html}{g_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}<div class=meta>{t_tot}/{t_uniq}</div></td></tr>")

    head = (f"<h2>s10 — name + counts, one call per model per cell</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"Claude {tallies['claude']}/{tot} · Gemini {tallies['gemini_flash']}/{tot} · "
            f"cross-model count agreement: {count_agree}/16 · "
            f"self-consistent (name len == its own count): "
            f"Claude {self_ok['claude']}/16 · Gemini {self_ok['gemini_flash']}/16</p>")
    css = VIEW_CSS + """<style>
     .s10 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s10 td.pass{background:#e6f5d9}
     .s10 td.fail{background:#fbe6e6}
     .s10 td.nogt{background:#f6f6f6;color:#333}
     .s10 td.truth{background:#eef0f5;font-weight:600}
     .s10 td .meta{font-size:11px;color:#666;font-weight:normal;margin-top:2px}
     .s10 .sok{color:#3a7a1a}
     .s10 .sbad{color:#a03000;font-weight:600}
     .s10 .agree{color:#3a7a1a;font-weight:600}
     .s10 .disagree{color:#a03000;font-weight:600}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; "
            "<a href='/strategies/s10' style='font-weight:600'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; <a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; <a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; <a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; <a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; <a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s10'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini Flash</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s11/crop/<name>")
def s11_crop(name):
    p = S11_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s11")
def s11_view():
    p = S11_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s11</h1><p>No results.json yet — run <code>src/strategies/s11.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    FEATURES = ("tall_strokes", "descenders", "dots", "closed_bowls_at_baseline")

    def format_features(rec, expected=None):
        parts = []
        for f in FEATURES:
            v = rec.get(f)
            e = expected.get(f) if expected else None
            label = {"tall_strokes": "T", "descenders": "D",
                     "dots": "d", "closed_bowls_at_baseline": "B"}[f]
            match_class = ""
            if e is not None and v is not None:
                match_class = " match" if v == e else " miss"
            display = "-" if v is None else str(v)
            parts.append(f"<span class='feat{match_class}'>{label}<b>{display}</b></span>")
        return " ".join(parts)

    def truth_features(name: str):
        if not name or name.strip().upper() == "[DITTO]":
            return {"tall_strokes": 0, "descenders": 0, "dots": 0, "closed_bowls_at_baseline": 0}
        tall = desc = dots = bowls = 0
        _asc = set("bdfhklt"); _desc = set("gjpqy"); _dot = set("ij"); _bowl = set("bdpq")
        for ch in name:
            if not ch.isalpha(): continue
            low = ch.lower()
            if ch.isupper() or low in _asc: tall += 1
            if low in _desc: desc += 1
            if low in _dot: dots += 1
            if low in _bowl: bowls += 1
        return {"tall_strokes": tall, "descenders": desc,
                "dots": dots, "closed_bowls_at_baseline": bowls}

    rows = ""
    agree_all = agree_none = 0
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            expected = truth_features(truth) if truth else None
            c, g = fd["claude"], fd["gemini_flash"]

            # cross-model feature agreement summary
            same_all = all(c.get(f) == g.get(f) for f in FEATURES)
            if same_all: agree_all += 1
            else: agree_none += 1

            claude_html = format_features(c, expected)
            gemini_html = format_features(g, expected)
            expected_html = format_features(expected, expected) if expected else "—"
            agree_badge = ("<span class=agree>features agree</span>"
                           if same_all
                           else "<span class=disagree>features differ</span>")

            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div>"
                     f"<div class=meta>{agree_badge}</div></td>"
                     f"<td><img src='/strategies/s11/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"<td>{claude_html}<div class=meta>{c.get('confidence','?')}</div></td>"
                     f"<td>{gemini_html}<div class=meta>{g.get('confidence','?')}</div></td>"
                     f"<td class=truth>{html.escape(truth) or '—'}"
                     f"<div>{expected_html}</div></td></tr>")

    head = (f"<h2>s11 — shape-only visual-feature probes (no name asked)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · counts: "
            f"<b>T</b>=tall_strokes · <b>D</b>=descenders · <b>d</b>=dots · "
            f"<b>B</b>=closed_bowls_at_baseline · green = matches expected · "
            f"red = differs</p>"
            f"<p class=meta>cross-model full agreement on all four features: "
            f"{agree_all}/16 · at least one differs: {agree_none}/16</p>")
    css = VIEW_CSS + """<style>
     .s11 img{max-width:520px;max-height:120px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s11 td.truth{background:#eef0f5;font-weight:600}
     .s11 td .meta{font-size:11px;color:#666;font-weight:normal;margin-top:2px}
     .s11 .feat{display:inline-block;margin-right:8px;font-family:ui-monospace,monospace;font-size:13px;padding:1px 4px;border-radius:3px}
     .s11 .feat b{margin-left:2px}
     .s11 .feat.match{background:#e6f5d9;color:#2c5c11}
     .s11 .feat.miss{background:#fbe6e6;color:#a03000}
     .s11 .agree{color:#3a7a1a;font-weight:600}
     .s11 .disagree{color:#a03000;font-weight:600}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11' style='font-weight:600'>s11</a> &middot; "
            "<a href='/strategies/s12'>s12</a></p>"
            f"{head}"
            "<div class='tablewrap s11'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini</th><th>Truth (name + expected)</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s12/crop/<name>")
def s12_crop(name):
    p = S12_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s12")
def s12_view():
    p = S12_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s12</h1><p>No results.json yet — run <code>src/strategies/s12.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}
    c_pass = g_pass = tot_first = tot_last = c_last_pass = g_last_pass = 0
    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            c, g = fd["claude"], fd["gemini_flash"]

            def cell(read, expected):
                if not expected:
                    return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
                ok = _match(read["name"], expected)
                cls = "pass" if ok else "fail"
                return (f"<td class={cls}>{html.escape(read['name'])}"
                        f"<div class=meta>{read['confidence']}</div></td>"), ok

            c_html, c_ok = cell(c, truth)
            g_html, g_ok = cell(g, truth)
            if truth:
                if field == "first":
                    tot_first += 1; c_pass += c_ok; g_pass += g_ok
                else:
                    tot_last += 1; c_last_pass += c_ok; g_last_pass += g_ok
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s12/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_html}{g_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    head = (f"<h2>s12 — wider crop (5 rows, target boxed) for scribe-consistency context</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"Claude first {c_pass}/{tot_first} · last {c_last_pass}/{tot_last} · "
            f"Gemini first {g_pass}/{tot_first} · last {g_last_pass}/{tot_last}</p>"
            f"<p class=meta>each crop shows target row in a red box, with 2 rows above and 2 below "
            f"as scribe-style reference</p>")
    css = VIEW_CSS + """<style>
     .s12 img{max-width:520px;max-height:280px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s12 td.pass{background:#e6f5d9} .s12 td.fail{background:#fbe6e6}
     .s12 td.nogt{background:#f6f6f6;color:#333}
     .s12 td.truth{background:#eef0f5;font-weight:600}
     .s12 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; "
            "<a href='/strategies/s12' style='font-weight:600'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a></p>"
            f"{head}"
            "<div class='tablewrap s12'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s13/crop/<name>")
def s13_crop(name):
    p = S13_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s13")
def s13_view():
    p = S13_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s13</h1><p>No results.json yet — run <code>src/strategies/s13.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    rows = ""
    c_pass = g_pass = tot = 0
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            c, g = fd["claude"], fd["gemini_flash"]

            def cell(read, expected):
                if not expected:
                    return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
                ok = _match(read["name"], expected)
                cls = "pass" if ok else "fail"
                return (f"<td class={cls}>{html.escape(read['name'])}"
                        f"<div class=meta>{read['confidence']}</div></td>"), ok

            c_html, c_ok = cell(c, truth)
            g_html, g_ok = cell(g, truth)
            if truth:
                tot += 1
                c_pass += c_ok; g_pass += g_ok
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s13/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_html}{g_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    head = (f"<h2>s13 — archaic abbreviation hint in prompt (L35 only)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"target lines: {d.get('target_lines')} · "
            f"Claude {c_pass}/{tot} · Gemini {g_pass}/{tot}</p>"
            f"<details><summary class=meta>prompt (first name)</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px'>{html.escape(d['first_prompt'])}</pre></details>")
    css = VIEW_CSS + """<style>
     .s13 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s13 td.pass{background:#e6f5d9} .s13 td.fail{background:#fbe6e6}
     .s13 td.nogt{background:#f6f6f6;color:#333}
     .s13 td.truth{background:#eef0f5;font-weight:600}
     .s13 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13' style='font-weight:600'>s13</a> &middot; "
            "<a href='/strategies/s14'>s14</a> &middot; <a href='/strategies/s15'>s15</a></p>"
            f"{head}"
            "<div class='tablewrap s13'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s14/crop/<name>")
def s14_crop(name):
    p = S14_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s14")
def s14_view():
    p = S14_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s14</h1><p>No results.json yet — run <code>src/strategies/s14.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    rows = ""
    c_pass = g_pass = tot = 0
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            c, g = fd["claude"], fd["gemini_flash"]

            def cell(read, expected):
                if not expected:
                    return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
                ok = _match(read["name"], expected)
                cls = "pass" if ok else "fail"
                return (f"<td class={cls}>{html.escape(read['name'])}"
                        f"<div class=meta>{read['confidence']}</div></td>"), ok

            c_html, c_ok = cell(c, truth)
            g_html, g_ok = cell(g, truth)
            if truth:
                tot += 1
                c_pass += c_ok; g_pass += g_ok
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s14/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_html}{g_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    head = (f"<h2>s14 — candidate-list prompt (L35 only)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"target lines: {d.get('target_lines')} · "
            f"Claude {c_pass}/{tot} · Gemini {g_pass}/{tot}</p>"
            f"<details><summary class=meta>prompt (first name)</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px'>{html.escape(d['first_prompt'])}</pre></details>")
    css = VIEW_CSS + """<style>
     .s14 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s14 td.pass{background:#e6f5d9} .s14 td.fail{background:#fbe6e6}
     .s14 td.nogt{background:#f6f6f6;color:#333}
     .s14 td.truth{background:#eef0f5;font-weight:600}
     .s14 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; "
            "<a href='/strategies/s14' style='font-weight:600'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a></p>"
            f"{head}"
            "<div class='tablewrap s14'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s15/crop/<name>")
def s15_crop(name):
    p = S15_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s15")
def s15_view():
    p = S15_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s15</h1><p>No results.json yet — run <code>src/strategies/s15.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    rows = ""
    c_pass = g_pass = tot = 0
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            c, g = fd["claude"], fd["gemini_flash"]

            def cell(read, expected):
                if not expected:
                    return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
                ok = _match(read["name"], expected)
                cls = "pass" if ok else "fail"
                return (f"<td class={cls}>{html.escape(read['name'])}"
                        f"<div class=meta>{read['confidence']}</div></td>"), ok

            c_html, c_ok = cell(c, truth)
            g_html, g_ok = cell(g, truth)
            if truth:
                tot += 1
                c_pass += c_ok; g_pass += g_ok
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s15/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_html}{g_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    head = (f"<h2>s15 — full-row crop with demographic-context inference (L35 only)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"target lines: {d.get('target_lines')} · "
            f"Claude {c_pass}/{tot} · Gemini {g_pass}/{tot}</p>"
            f"<details><summary class=meta>prompt (first name)</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px'>{html.escape(d['first_prompt'])}</pre></details>")
    css = VIEW_CSS + """<style>
     .s15 img{max-width:900px;max-height:220px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s15 td.pass{background:#e6f5d9} .s15 td.fail{background:#fbe6e6}
     .s15 td.nogt{background:#f6f6f6;color:#333}
     .s15 td.truth{background:#eef0f5;font-weight:600}
     .s15 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15' style='font-weight:600'>s15</a> &middot; "
            "<a href='/strategies/s16'>s16</a></p>"
            f"{head}"
            "<div class='tablewrap s15'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s16/crop/<name>")
def s16_crop(name):
    p = S16_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s16")
def s16_view():
    p = S16_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s16</h1><p>No results.json yet — run <code>src/strategies/s16.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    rows = ""
    c_pass = g_pass = tot = 0
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            c, g = fd["claude"], fd["gemini_flash"]

            def cell(read, expected):
                if not expected:
                    return f"<td class=nogt>{html.escape(read['name'])}<div class=meta>{read['confidence']}</div></td>", False
                ok = _match(read["name"], expected)
                cls = "pass" if ok else "fail"
                return (f"<td class={cls}>{html.escape(read['name'])}"
                        f"<div class=meta>{read['confidence']}</div></td>"), ok

            c_html, c_ok = cell(c, truth)
            g_html, g_ok = cell(g, truth)
            if truth:
                tot += 1
                c_pass += c_ok; g_pass += g_ok
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s16/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_html}{g_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    head = (f"<h2>s16 — s15 full-row context + s13 abbreviation hint (L35 only)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"target lines: {d.get('target_lines')} · "
            f"Claude {c_pass}/{tot} · Gemini {g_pass}/{tot}</p>"
            f"<details><summary class=meta>prompt (first name)</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px'>{html.escape(d['first_prompt'])}</pre></details>")
    css = VIEW_CSS + """<style>
     .s16 img{max-width:900px;max-height:220px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s16 td.pass{background:#e6f5d9} .s16 td.fail{background:#fbe6e6}
     .s16 td.nogt{background:#f6f6f6;color:#333}
     .s16 td.truth{background:#eef0f5;font-weight:600}
     .s16 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a> &middot; "
            "<a href='/strategies/s16' style='font-weight:600'>s16</a> &middot; "
            "<a href='/strategies/s17'>s17</a></p>"
            f"{head}"
            "<div class='tablewrap s16'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s17/crop/<name>")
def s17_crop(name):
    p = S17_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s17")
def s17_view():
    p = S17_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s17</h1><p>No results.json yet — run <code>src/strategies/s17.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""

            def cell(rec, expected):
                if not expected:
                    return f"<td class=nogt>{html.escape(rec['name'])}<div class=meta>{rec['confidence']}<br>{rec['model']}</div></td>"
                ok = _match(rec["name"], expected)
                cls = "pass" if ok else "fail"
                return (f"<td class={cls}>{html.escape(rec['name'])}"
                        f"<div class=meta>{rec['confidence']}<br>{rec['model']}</div></td>")

            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s17/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{cell(fd['opus'], truth)}{cell(fd['sonnet'], truth)}{cell(fd['haiku'], truth)}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    head = (f"<h2>s17 — Claude Opus vs Sonnet vs Haiku on L35 only</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"target lines: {d.get('target_lines')} · "
            f"same tight cell crop + simple exact-transcription prompt (s2 baseline)</p>")
    css = VIEW_CSS + """<style>
     .s17 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s17 td.pass{background:#e6f5d9} .s17 td.fail{background:#fbe6e6}
     .s17 td.nogt{background:#f6f6f6;color:#333}
     .s17 td.truth{background:#eef0f5;font-weight:600}
     .s17 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; "
            "<a href='/strategies/s17' style='font-weight:600'>s17</a> &middot; "
            "<a href='/strategies/s18'>s18</a></p>"
            f"{head}"
            "<div class='tablewrap s17'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Opus</th><th>Sonnet</th><th>Haiku</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s18/crop/<name>")
def s18_crop(name):
    p = S18_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s18")
def s18_view():
    p = S18_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s18</h1><p>No results.json yet — run <code>src/strategies/s18.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    c_pass = g_pass = tot = 0
    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            c, g = fd["claude"], fd["gemini_flash"]

            def cell(rec, expected):
                cands = rec.get("candidates") or []
                cand_html = ""
                if cands:
                    items = "".join(
                        f"<li><b>{html.escape(str(k.get('reading','?')))}</b>"
                        f" — <span class=why>{html.escape(str(k.get('why','')))}</span></li>"
                        for k in cands)
                    cand_html = (f"<details><summary>{len(cands)} candidates</summary>"
                                 f"<ol style='padding-left:18px;margin:4px 0'>{items}</ol></details>")
                if not expected:
                    return f"<td class=nogt>{html.escape(rec['final'])}<div class=meta>{rec['confidence']}</div>{cand_html}</td>", False
                ok = _match(rec["final"], expected)
                cls = "pass" if ok else "fail"
                return (f"<td class={cls}>{html.escape(rec['final'])}"
                        f"<div class=meta>{rec['confidence']}</div>{cand_html}</td>"), ok

            c_html, c_ok = cell(c, truth)
            g_html, g_ok = cell(g, truth)
            if truth:
                tot += 1
                c_pass += c_ok; g_pass += g_ok
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s18/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_html}{g_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    head = (f"<h2>s18 — enumerate 3 candidates before committing (generic)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"Claude {c_pass}/{tot} · Gemini {g_pass}/{tot} · "
            f"click 'N candidates' to see each model's ranked alternatives</p>")
    css = VIEW_CSS + """<style>
     .s18 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s18 td.pass{background:#e6f5d9} .s18 td.fail{background:#fbe6e6}
     .s18 td.nogt{background:#f6f6f6;color:#333}
     .s18 td.truth{background:#eef0f5;font-weight:600}
     .s18 td .meta{font-size:11px;color:#666}
     .s18 details{font-size:11px;color:#444;margin-top:4px}
     .s18 details summary{cursor:pointer;color:#666}
     .s18 .why{color:#555;font-style:italic}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; "
            "<a href='/strategies/s17'>s17</a> &middot; "
            "<a href='/strategies/s18' style='font-weight:600'>s18</a> &middot; "
            "<a href='/strategies/s19'>s19</a></p>"
            f"{head}"
            "<div class='tablewrap s18'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s19/crop/<name>")
def s19_crop(name):
    p = S19_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s19")
def s19_view():
    p = S19_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s19</h1><p>No results.json yet — run <code>src/strategies/s19.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}
    models = ("opus", "sonnet", "haiku")
    labels = {"opus": "Opus", "sonnet": "Sonnet", "haiku": "Haiku"}
    tallies = {m: {"first": 0, "last": 0} for m in models}
    tot_first = tot_last = 0

    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""

            def cell(rec, expected):
                if not expected:
                    return f"<td class=nogt>{html.escape(rec['name'])}<div class=meta>{rec['confidence']}</div></td>", False
                ok = _match(rec["name"], expected)
                cls = "pass" if ok else "fail"
                return (f"<td class={cls}>{html.escape(rec['name'])}"
                        f"<div class=meta>{rec['confidence']}</div></td>"), ok

            cells_html = ""
            for m in models:
                cell_html, ok = cell(fd[m], truth)
                cells_html += cell_html
                if truth:
                    tallies[m][field] += ok
            if truth:
                if field == "first": tot_first += 1
                else: tot_last += 1
            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s19/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{cells_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    tally_line = " · ".join(
        f"{labels[m]} first {tallies[m]['first']}/{tot_first} · last {tallies[m]['last']}/{tot_last}"
        for m in models)
    mm = d.get("models", {})
    head = (f"<h2>s19 — Claude Opus vs Sonnet vs Haiku on full fixture (no Gemini)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · {tally_line}</p>"
            f"<p class=meta>models: opus={html.escape(str(mm.get('opus','')))} · "
            f"sonnet={html.escape(str(mm.get('sonnet','')))} · "
            f"haiku={html.escape(str(mm.get('haiku','')))}</p>")
    css = VIEW_CSS + """<style>
     .s19 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s19 td.pass{background:#e6f5d9} .s19 td.fail{background:#fbe6e6}
     .s19 td.nogt{background:#f6f6f6;color:#333}
     .s19 td.truth{background:#eef0f5;font-weight:600}
     .s19 td .meta{font-size:11px;color:#666}
     .s19 th.grp{background:#e6ded0}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; "
            "<a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; "
            "<a href='/strategies/s19' style='font-weight:600'>s19</a> &middot; "
            "<a href='/strategies/s20'>s20</a></p>"
            f"{head}"
            "<div class='tablewrap s19'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Opus</th><th>Sonnet</th><th>Haiku</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s20/crop/<name>")
def s20_crop(name):
    p = S20_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s20")
def s20_view():
    p = S20_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s20</h1><p>No results.json yet — run <code>src/strategies/s20.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    tallies = {"claude": 0, "gemini_flash": 0}
    tot = 0
    calibration = {"claude": {"low_when_wrong": 0, "review_when_wrong": 0,
                              "low_when_right": 0, "review_when_right": 0,
                              "n_wrong": 0, "n_right": 0},
                   "gemini_flash": {"low_when_wrong": 0, "review_when_wrong": 0,
                                    "low_when_right": 0, "review_when_right": 0,
                                    "n_wrong": 0, "n_right": 0}}

    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""

            def cell(rec, expected, model_key):
                nr = rec.get("needs_review", False)
                conf = rec.get("confidence", "?")
                if not expected:
                    return f"<td class=nogt>{html.escape(rec['name'])}<div class=meta>{conf}{' · ⚑' if nr else ''}</div></td>", False, None
                ok = _match(rec["name"], expected)
                cls = "pass" if ok else "fail"
                badge = " · ⚑ review" if nr else ""
                return (f"<td class={cls}>{html.escape(rec['name'])}"
                        f"<div class=meta>{conf}{badge}</div></td>"), ok, {
                    "conf": conf, "nr": nr, "ok": ok}

            c_html, c_ok, c_info = cell(fd["claude"], truth, "claude")
            g_html, g_ok, g_info = cell(fd["gemini_flash"], truth, "gemini_flash")

            if truth:
                tot += 1
                tallies["claude"] += c_ok
                tallies["gemini_flash"] += g_ok
                for m, info in (("claude", c_info), ("gemini_flash", g_info)):
                    if info is None: continue
                    is_low = info["conf"] == "LOW"
                    if info["ok"]:
                        calibration[m]["n_right"] += 1
                        if is_low: calibration[m]["low_when_right"] += 1
                        if info["nr"]: calibration[m]["review_when_right"] += 1
                    else:
                        calibration[m]["n_wrong"] += 1
                        if is_low: calibration[m]["low_when_wrong"] += 1
                        if info["nr"]: calibration[m]["review_when_wrong"] += 1

            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s20/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{c_html}{g_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    def cal_line(m):
        cal = calibration[m]
        return (f"{cal['low_when_wrong']}/{cal['n_wrong']} wrong-flagged-LOW · "
                f"{cal['review_when_wrong']}/{cal['n_wrong']} wrong-flagged-review · "
                f"{cal['low_when_right']}/{cal['n_right']} right-flagged-LOW · "
                f"{cal['review_when_right']}/{cal['n_right']} right-flagged-review")

    head = (f"<h2>s20 — calibration rubric + needs_review flag</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · "
            f"Claude {tallies['claude']}/{tot} · Gemini {tallies['gemini_flash']}/{tot}</p>"
            f"<p class=meta>Claude calibration: {cal_line('claude')}</p>"
            f"<p class=meta>Gemini calibration: {cal_line('gemini_flash')}</p>"
            f"<p class=meta>⚑ = needs_review flag set. Ideal: wrong-flagged-LOW / "
            f"wrong-flagged-review should be HIGH (model knows when it's wrong); "
            f"right-flagged-LOW / right-flagged-review should be LOW (not over-cautious).</p>")
    css = VIEW_CSS + """<style>
     .s20 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s20 td.pass{background:#e6f5d9} .s20 td.fail{background:#fbe6e6}
     .s20 td.nogt{background:#f6f6f6;color:#333}
     .s20 td.truth{background:#eef0f5;font-weight:600}
     .s20 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; "
            "<a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; "
            "<a href='/strategies/s19'>s19</a> &middot; "
            "<a href='/strategies/s20' style='font-weight:600'>s20</a> &middot; "
            "<a href='/strategies/s21'>s21</a></p>"
            f"{head}"
            "<div class='tablewrap s20'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Gemini</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s21/crop/<name>")
def s21_crop(name):
    p = S21_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s21")
def s21_view():
    p = S21_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s21</h1><p>No results.json yet — run <code>src/strategies/s21.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    tally_pass = 0
    tot = 0
    cal = {"low_when_wrong": 0, "review_when_wrong": 0, "n_wrong": 0,
           "low_when_right": 0, "review_when_right": 0, "n_right": 0}

    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            c = fd["claude"]
            nr = c.get("needs_review", False)
            conf = c.get("confidence", "?")
            if not truth:
                cell_html = f"<td class=nogt>{html.escape(c['name'])}<div class=meta>{conf}{' · ⚑' if nr else ''}</div></td>"
            else:
                tot += 1
                ok = _match(c["name"], truth)
                tally_pass += ok
                cls = "pass" if ok else "fail"
                badge = " · ⚑ review" if nr else ""
                cell_html = (f"<td class={cls}>{html.escape(c['name'])}"
                             f"<div class=meta>{conf}{badge}</div></td>")
                if ok:
                    cal["n_right"] += 1
                    if conf == "LOW": cal["low_when_right"] += 1
                    if nr: cal["review_when_right"] += 1
                else:
                    cal["n_wrong"] += 1
                    if conf == "LOW": cal["low_when_wrong"] += 1
                    if nr: cal["review_when_wrong"] += 1

            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s21/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{cell_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    head = (f"<h2>s21 — calibration + familiar-name red flag + stroke verify (Claude only)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · Claude {tally_pass}/{tot}</p>"
            f"<p class=meta>calibration: {cal['low_when_wrong']}/{cal['n_wrong']} wrong-LOW · "
            f"{cal['review_when_wrong']}/{cal['n_wrong']} wrong-review · "
            f"{cal['low_when_right']}/{cal['n_right']} right-LOW · "
            f"{cal['review_when_right']}/{cal['n_right']} right-review</p>")
    css = VIEW_CSS + """<style>
     .s21 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s21 td.pass{background:#e6f5d9} .s21 td.fail{background:#fbe6e6}
     .s21 td.nogt{background:#f6f6f6;color:#333}
     .s21 td.truth{background:#eef0f5;font-weight:600}
     .s21 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; "
            "<a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; "
            "<a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; "
            "<a href='/strategies/s21' style='font-weight:600'>s21</a> &middot; "
            "<a href='/strategies/s22'>s22</a></p>"
            f"{head}"
            "<div class='tablewrap s21'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Claude</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s22/crop/<name>")
def s22_crop(name):
    p = S22_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s22")
def s22_view():
    p = S22_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s22</h1><p>No results.json yet — run <code>src/strategies/s22.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    field_key = {"first": "interpreted_first_name", "last": "interpreted_last_name"}

    tally_pass = 0
    tot = 0
    cal = {"low_when_wrong": 0, "review_when_wrong": 0, "n_wrong": 0,
           "low_when_right": 0, "review_when_right": 0, "n_right": 0}

    rows = ""
    for r in d["results"]:
        for field in ("first", "last"):
            fd = r["fields"][field]
            truth = tm.get((r["line"], field_key[field])) or ""
            c = fd["fable_5"]
            nr = c.get("needs_review", False)
            conf = c.get("confidence", "?")
            if not truth:
                cell_html = f"<td class=nogt>{html.escape(c['name'])}<div class=meta>{conf}{' · ⚑' if nr else ''}</div></td>"
            else:
                tot += 1
                ok = _match(c["name"], truth)
                tally_pass += ok
                cls = "pass" if ok else "fail"
                badge = " · ⚑ review" if nr else ""
                cell_html = (f"<td class={cls}>{html.escape(c['name'])}"
                             f"<div class=meta>{conf}{badge}</div></td>")
                if ok:
                    cal["n_right"] += 1
                    if conf == "LOW": cal["low_when_right"] += 1
                    if nr: cal["review_when_right"] += 1
                else:
                    cal["n_wrong"] += 1
                    if conf == "LOW": cal["low_when_wrong"] += 1
                    if nr: cal["review_when_wrong"] += 1

            rows += (f"<tr><td class=ln>L{r['line']:02d}<div class=meta>{field}</div></td>"
                     f"<td><img src='/strategies/s22/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                     f"{cell_html}"
                     f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")

    mm = d.get("models", {})
    head = (f"<h2>s22 — s21 rubric on {html.escape(str(mm.get('fable_5','')))} (no Opus)</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · Fable-5 {tally_pass}/{tot}</p>"
            f"<p class=meta>calibration: {cal['low_when_wrong']}/{cal['n_wrong']} wrong-LOW · "
            f"{cal['review_when_wrong']}/{cal['n_wrong']} wrong-review · "
            f"{cal['low_when_right']}/{cal['n_right']} right-LOW · "
            f"{cal['review_when_right']}/{cal['n_right']} right-review</p>")
    css = VIEW_CSS + """<style>
     .s22 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s22 td.pass{background:#e6f5d9} .s22 td.fail{background:#fbe6e6}
     .s22 td.nogt{background:#f6f6f6;color:#333}
     .s22 td.truth{background:#eef0f5;font-weight:600}
     .s22 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; "
            "<a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; "
            "<a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; "
            "<a href='/strategies/s21'>s21</a> &middot; "
            "<a href='/strategies/s22' style='font-weight:600'>s22</a> &middot; "
            "<a href='/strategies/s23'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s22'><table><thead>"
            "<tr><th>Ln / field</th><th>Crop</th>"
            "<th>Fable-5</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/strategies/s23/crop/<name>")
def s23_crop(name):
    p = S23_DIR / name
    if not p.exists() or ".." in name or "/" in name:
        return ("", 404)
    return send_file(p.resolve(), mimetype="image/png")


@app.route("/strategies/s23")
def s23_view():
    p = S23_DIR / "results.json"
    if not p.exists():
        return VIEW_CSS + "<div class=wrap><h1>s23</h1><p>No results.json yet — run <code>src/strategies/s23.py</code>.</p></div>"
    d = json.loads(p.read_text())
    tm = _fresh_truth()
    rows = ""
    passes = tot = 0
    for r in d["results"]:
        truth = _truth_combined(tm, r["line"]) or r.get("truth", "")
        rec = r["fable_5"]
        conf = rec.get("confidence", "?")
        if truth:
            tot += 1
            ok = _match(rec["name"], truth)
            passes += ok
            cls = "pass" if ok else "fail"
        else:
            cls = "nogt"
        rows += (f"<tr><td class=ln>L{r['line']:02d}</td>"
                 f"<td><img src='/strategies/s23/crop/{html.escape(r['crop'])}' loading=lazy></td>"
                 f"<td class={cls}>{html.escape(rec['name'])}<div class=meta>{conf}</div></td>"
                 f"<td class=truth>{html.escape(truth) or '—'}</td></tr>")
    mm = d.get("models", {})
    head = (f"<h2>s23 — combined-name read on {html.escape(str(mm.get('fable_5','')))} only</h2>"
            f"<p class=meta>{d['reel']} frame {d['frame']:04d} · Fable-5 {passes}/{tot} · "
            f"one call per row (half the cost of s2-shape split-name)</p>")
    css = VIEW_CSS + """<style>
     .s23 img{max-width:520px;max-height:180px;border:1px solid #ddd;border-radius:3px;background:#fff}
     .s23 td.pass{background:#e6f5d9} .s23 td.fail{background:#fbe6e6}
     .s23 td.nogt{background:#f6f6f6;color:#333}
     .s23 td.truth{background:#eef0f5;font-weight:600}
     .s23 td .meta{font-size:11px;color:#666}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — Strategies</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; <a href='/view'>consensus view</a> &middot; "
            "<a href='/strategies/s1'>s1</a> &middot; <a href='/strategies/s2'>s2</a> &middot; "
            "<a href='/strategies/s3'>s3</a> &middot; <a href='/strategies/s4'>s4</a> &middot; "
            "<a href='/strategies/s5'>s5</a> &middot; <a href='/strategies/s6'>s6</a> &middot; "
            "<a href='/strategies/s7'>s7</a> &middot; <a href='/strategies/s8'>s8</a> &middot; "
            "<a href='/strategies/s9'>s9</a> &middot; <a href='/strategies/s10'>s10</a> &middot; "
            "<a href='/strategies/s11'>s11</a> &middot; <a href='/strategies/s12'>s12</a> &middot; "
            "<a href='/strategies/s13'>s13</a> &middot; <a href='/strategies/s14'>s14</a> &middot; "
            "<a href='/strategies/s15'>s15</a> &middot; <a href='/strategies/s16'>s16</a> &middot; "
            "<a href='/strategies/s17'>s17</a> &middot; <a href='/strategies/s18'>s18</a> &middot; "
            "<a href='/strategies/s19'>s19</a> &middot; <a href='/strategies/s20'>s20</a> &middot; "
            "<a href='/strategies/s21'>s21</a> &middot; <a href='/strategies/s22'>s22</a> &middot; "
            "<a href='/strategies/s23' style='font-weight:600'>s23</a></p>"
            f"{head}"
            "<div class='tablewrap s23'><table><thead>"
            "<tr><th>Ln</th><th>Crop</th>"
            "<th>Fable-5</th><th>Truth</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table></div></div>")


@app.route("/pipeline_v2")
@app.route("/pipeline_v2/<int:frame>")
def pipeline_v2_view(frame=None):
    """View pipeline_v2 output for one frame: full row table with escalation
    badges + ground-truth pass/fail highlighting on the fixture cells.

    Optional query param `?partial=<suffix>` loads a partial run written by
    `pipeline_v2 --lines ...` (filename `_<frame>.pipeline_v2.partial.<suffix>.json`).
    """
    reel = REEL
    partial = request.args.get("partial", "").strip() or None

    files = sorted(OUT.glob(f"{reel}_*.pipeline_v2.json"))
    if not files:
        return VIEW_CSS + ("<div class=wrap><h1>pipeline_v2</h1>"
                           "<p>No pipeline_v2 output yet. Run "
                           "<code>src/pipeline_v2.py</code>.</p></div>")
    frames_avail = sorted({
        int(p.name.replace(".pipeline_v2.json", "").split("_")[-1])
        for p in files})
    if frame is None:
        frame = frames_avail[0]
    if partial:
        p = OUT / f"{reel}_{int(frame):04d}.pipeline_v2.partial.{partial}.json"
    else:
        p = OUT / f"{reel}_{int(frame):04d}.pipeline_v2.json"
    if not p.exists():
        return VIEW_CSS + (f"<p>No pipeline_v2 output for frame {frame}"
                           + (f" partial={partial}" if partial else "")
                           + f" (looked for {p.name}).</p>")

    d = json.loads(p.read_text())
    tm = _fresh_truth()
    truth_row_map = {}
    for k, v in tm.items():
        line, field = k
        truth_row_map.setdefault(str(line), {})[field] = v

    meta = d.get("metadata", {}) or {}
    models = d.get("models", {})

    def _cell_html(row, field):
        val = row.get(field)
        display = html.escape(str(val)) if val not in (None, "") else ""
        # Fixture ground truth if any → pass/fail highlight
        gt = truth_row_map.get(row["line_number"], {}).get(field)
        extras = ""
        cls = ""
        if field in ("interpreted_first_name", "interpreted_last_name"):
            meta_key = "_first_name_meta" if field == "interpreted_first_name" else "_last_name_meta"
            m = row.get(meta_key, {}) or {}
            conf = m.get("confidence")
            escalated = m.get("escalated")
            needs_review = m.get("needs_review")
            badges = []
            if escalated:
                badges.append("<span class=badge-fable>FABLE</span>")
            if conf == "LOW":
                badges.append("<span class=badge-low>LOW</span>")
            elif conf == "MEDIUM":
                badges.append("<span class=badge-med>MED</span>")
            if needs_review:
                badges.append("<span class=badge-rev>⚑</span>")
            if badges:
                extras = "<div class=badges>" + " ".join(badges) + "</div>"
            # If escalated, show the original Opus read
            opus_read = m.get("opus_read") or {}
            if opus_read.get("name") and opus_read["name"] != val:
                extras += (f"<div class=opus-orig>opus: "
                           f"{html.escape(str(opus_read['name']))}</div>")
        if gt is not None:
            cls = "pass" if _match(val, gt) else "fail"
        return f'<td class="{cls}">{display}{extras}</td>'

    trs = ""
    for row in d.get("rows", []):
        if row.get("_missing"):
            trs += (f'<tr class=missing><td class=ln>{row["line_number"]}</td>'
                    + "<td colspan=14 class=meta>(band not read)</td></tr>")
            continue
        tds = "".join(_cell_html(row, f) for f, _ in VIEW_COLS)
        trs += f'<tr><td class=ln>{row["line_number"]}</td>{tds}</tr>'

    ths = "".join(f"<th>{lbl}</th>" for _, lbl in VIEW_COLS)

    # Escalation summary
    n_escalated = sum(1 for r in d.get("rows", [])
                      for m in ((r.get("_first_name_meta") or {}),
                                (r.get("_last_name_meta") or {}))
                      if m.get("escalated"))
    n_low = sum(1 for r in d.get("rows", [])
                for m in ((r.get("_first_name_meta") or {}),
                          (r.get("_last_name_meta") or {}))
                if m.get("confidence") == "LOW")

    # Fixture score
    tot = correct = 0
    for r in d.get("rows", []):
        for field in ("interpreted_first_name", "interpreted_last_name"):
            gt = truth_row_map.get(r["line_number"], {}).get(field)
            if gt is None: continue
            tot += 1
            if _match(r.get(field), gt): correct += 1

    nav = " ".join(
        f'<a class="{"on" if int(fr) == int(frame) else ""}" href="/pipeline_v2/{fr}">{fr:04d}</a>'
        for fr in frames_avail)
    # discover partial runs for the CURRENT frame
    partial_files = sorted(OUT.glob(
        f"{reel}_{int(frame):04d}.pipeline_v2.partial.*.json"))
    partial_suffixes = [pf.name.replace(f"{reel}_{int(frame):04d}.pipeline_v2.partial.", "")
                          .replace(".json", "") for pf in partial_files]
    partial_nav = ""
    if partial_suffixes:
        parts = []
        parts.append(f'<a class="{"on" if not partial else ""}" '
                     f'href="/pipeline_v2/{int(frame)}">full</a>')
        for suffix in partial_suffixes:
            parts.append(f'<a class="{"on" if partial == suffix else ""}" '
                         f'href="/pipeline_v2/{int(frame)}?partial={suffix}">'
                         f'partial: {suffix}</a>')
        partial_nav = f'<p class=nav>Runs for {int(frame):04d}: {" ".join(parts)}</p>'
    head = (f"<h2>pipeline_v2 &mdash; frame {int(frame):04d} &mdash; "
            f"{html.escape(str(meta.get('location_town') or '?'))}, "
            f"{html.escape(str(meta.get('location_county') or '?'))} County</h2>"
            f"<p class=meta>models: primary={html.escape(str(models.get('primary','')))} · "
            f"name escalation={html.escape(str(models.get('name_escalation','')))} · "
            f"date={html.escape(str(meta.get('enumeration_date','?')))}</p>"
            f"<p class=meta>fixture score: <b>{correct}/{tot}</b> on the marquee cells · "
            f"{n_escalated} name cells escalated to Fable · "
            f"{n_low} name cells at LOW confidence</p>"
            f"<p class='meta legend'>"
            f"<span style='background:#e6f5d9;padding:0 4px'>green</span> = fixture cell PASS "
            f"<span style='background:#fbe6e6;padding:0 4px'>red</span> = fixture cell FAIL "
            f"<span class=badge-fable>FABLE</span> = escalated · "
            f"<span class=badge-low>LOW</span>/<span class=badge-med>MED</span> = confidence · "
            f"<span class=badge-rev>⚑</span> = needs_review</p>")
    css = VIEW_CSS + """<style>
     .pipeline_v2 td.pass{background:#e6f5d9}
     .pipeline_v2 td.fail{background:#fbe6e6}
     .badges{margin-top:2px}
     .badge-fable{background:#4a3aa8;color:white;font-size:10px;padding:1px 4px;border-radius:3px;font-weight:600}
     .badge-low{background:#a03000;color:white;font-size:10px;padding:1px 4px;border-radius:3px;font-weight:600}
     .badge-med{background:#a08000;color:white;font-size:10px;padding:1px 4px;border-radius:3px}
     .badge-rev{background:#8a5a00;color:white;font-size:10px;padding:1px 4px;border-radius:3px}
     .opus-orig{font-size:10px;color:#666;font-style:italic;margin-top:2px}
     .missing td{background:#f6f6f6;color:#999}
    </style>"""
    return (css + "<div class=wrap><h1>Scriptorium — pipeline_v2</h1>"
            "<p><a href='/'>&larr; API playground</a> &middot; "
            "<a href='/view'>consensus view</a> &middot; "
            "<a href='/pipeline_v2' style='font-weight:600'>pipeline_v2</a></p>"
            f"<p class=nav>Frames: {nav}</p>{partial_nav}{head}"
            f"<div class='tablewrap pipeline_v2'><table><thead>"
            f"<tr><th>Ln</th>{ths}</tr></thead>"
            f"<tbody>{trs}</tbody></table></div></div>")


if __name__ == "__main__":
    host = os.environ.get("WEBPLAY_HOST", "127.0.0.1")
    print(f"Gemini playground -> http://{host}:5001")
    app.run(host=host, port=5001, debug=False)
