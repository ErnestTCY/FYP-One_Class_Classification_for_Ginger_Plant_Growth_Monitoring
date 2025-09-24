import os
import time
import logging
from datetime import datetime, date
from pathlib import Path
import requests
from flask import Response
from werkzeug.utils import secure_filename
from requests.exceptions import RequestException
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash

# -----------------------------------------------------------------------------
# App / config
# -----------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

app = Flask(__name__)
app.config["SECRET_KEY"] = "replace-me"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------------------------------------------------------
# HTTP helpers with detailed logging
# -----------------------------------------------------------------------------
def fetch_json(url, default=None, timeout=8):
    t0 = time.time()
    try:
        r = requests.get(url, timeout=timeout)
        dt = (time.time() - t0) * 1000
        if r.ok:
            try:
                j = r.json()
                app.logger.info(
                    f"[UI] GET {url} -> {r.status_code} in {dt:.1f}ms; "
                    f"type={'dict' if isinstance(j, dict) else 'list'} keys={list(j) if isinstance(j, dict) else '—'}"
                )
                return j
            except ValueError:
                app.logger.error(
                    f"[UI] GET {url} non-JSON in {dt:.1f}ms; body[:200]={r.text[:200]!r}"
                )
                return default
        else:
            app.logger.warning(
                f"[UI] GET {url} -> HTTP {r.status_code} in {dt:.1f}ms; body[:200]={r.text[:200]!r}"
            )
            return default
    except RequestException as e:
        dt = (time.time() - t0) * 1000
        app.logger.exception(f"[UI] GET {url} exception after {dt:.1f}ms: {e}")
        return default

def post_json(url, payload=None, timeout=12):
    t0 = time.time()
    try:
        r = requests.post(url, json=(payload or {}), timeout=timeout)
        dt = (time.time() - t0) * 1000
        app.logger.info(
            f"[UI] POST {url} payload={payload} -> {r.status_code} in {dt:.1f}ms; body[:200]={r.text[:200]!r}"
        )
        return r
    except RequestException as e:
        dt = (time.time() - t0) * 1000
        app.logger.exception(f"[UI] POST {url} exception after {dt:.1f}ms: {e}")
        return None

def delete_call(url, timeout=12):
    t0 = time.time()
    try:
        r = requests.delete(url, timeout=timeout)
        dt = (time.time() - t0) * 1000
        app.logger.info(
            f"[UI] DELETE {url} -> {r.status_code} in {dt:.1f}ms; body[:200]={r.text[:200]!r}"
        )
        return r
    except RequestException as e:
        dt = (time.time() - t0) * 1000
        app.logger.exception(f"[UI] DELETE {url} exception after {dt:.1f}ms: {e}")
        return None

def _try_delete(url):
    # Prefer DELETE; if some proxies block it, fall back to POST /delete
    r = delete_call(url)
    if r is not None and r.status_code >= 400:
        try:
            rr = requests.post(url + "/delete", timeout=10)
            app.logger.info(f"[UI] POST fallback {url}/delete -> {rr.status_code}")
            return rr
        except Exception as e:
            app.logger.exception(f"[UI] POST fallback {url}/delete failed: {e}")
            return r
    return r

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/sessions/<int:sid>/report_overall")
def session_report_overall(sid):
    url = f"{BACKEND}/api/sessions/{sid}/report_overall"
    # give it time to render
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        return f"Report generation failed (HTTP {r.status_code}).", r.status_code

    filename = r.headers.get("X-Filename", f"session_{sid}_report.pdf")
    filename = secure_filename(filename)

    return Response(
        r.content,
        headers={
            "Content-Type": "application/pdf",
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
        status=200
    )
    
@app.get("/")
def home():
    sessions = fetch_json(f"{BACKEND}/api/sessions", [])
    latest   = fetch_json(f"{BACKEND}/api/detections", [])
    weather  = fetch_json(f"{BACKEND}/api/weather", None)
    return render_template("home.html", sessions=sessions, latest=latest, weather=weather, backend=BACKEND)

@app.get("/detect")
def detect_page():
    sessions = fetch_json(f"{BACKEND}/api/sessions", [])
    return render_template("detect.html", sessions=sessions, backend=BACKEND)

@app.post("/detect")
def do_detect():
    week = request.form.get("week")
    session_id = request.form.get("session_id")
    f = request.files.get("image")
    if not f:
        flash("Choose an image")
        return redirect(url_for("detect_page"))

    t0 = time.time()
    try:
        r = requests.post(
            f"{BACKEND}/api/jobs",
            data={"week": week, "session_id": session_id},
            files={"image": (f.filename, f.stream, f.mimetype)},
            timeout=30
        )
        dt = (time.time() - t0) * 1000
        app.logger.info(f"[UI] POST /api/jobs sid={session_id} week={week} -> {r.status_code} in {dt:.1f}ms")
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", "unknown")
            except Exception:
                msg = r.text[:200]
            flash(f"Error: {msg}")
    except Exception as e:
        app.logger.exception(f"[UI] detect upload failed: {e}")
        flash(f"Upload failed: {e}")
    return redirect(url_for("history"))

@app.get("/sessions")
def sessions():
    sessions = fetch_json(f"{BACKEND}/api/sessions", [])
    return render_template("sessions.html", sessions=sessions, backend=BACKEND)

@app.get("/sessions/<int:sid>")
def session_detail(sid):
    # actual session record
    sess = fetch_json(f"{BACKEND}/api/sessions/{sid}", {})
    # planned/present list that drives the cards
    weeks = fetch_json(f"{BACKEND}/api/sessions/{sid}/weeks", {"planned": [], "present": []})
    # histogram (safe default)
    sess_hist = fetch_json(
        f"{BACKEND}/api/sessions/{sid}/hist",
        {"labels": ["Normal plant","Abnormal plant","Empty Bag Detected"], "counts": [0,0,0]}
    )

    app.logger.info(
        f"[UI] /sessions/{sid}: sess={{name:{sess.get('name')}, start:{sess.get('started_at')}, cw:{sess.get('current_week')}}}, "
        f"planned={weeks.get('planned')}, present={weeks.get('present')}"
    )

    started_at = sess.get("started_at")
    now = datetime.utcnow()

    return render_template(
        "session_detail.html",
        sid=sid,
        sess=sess,
        weeks=weeks,
        sess_hist=sess_hist,
        started_at=started_at,
        now=now,
        backend=BACKEND,
    )

@app.get("/sessions/<int:sid>/week/<int:week>")
def session_week(sid, week):
    # detections list
    dets = fetch_json(f"{BACKEND}/api/sessions/{sid}/weeks/{week}/detections", [])
    # summary for week (dates/day buckets/hist)
    wsum = fetch_json(f"{BACKEND}/api/sessions/{sid}/weeks/{week}/summary", {})

    det_by_id = {d.get("id"): d for d in (dets or []) if isinstance(d, dict) and "id" in d}

    # build day buckets with actual detection objects
    days = []
    for day in (wsum.get("days") or []):
        ids = day.get("ids", [])
        items = [det_by_id[i] for i in ids if i in det_by_id]
        days.append({"name": day.get("name"), "date": day.get("date"), "items": items})

    app.logger.info(
        f"[UI] /sessions/{sid}/week/{week}: dets={len(dets) if isinstance(dets, list) else 'n/a'} "
        f"days={len(days)} week_start={wsum.get('week_start_date')}"
    )

    return render_template(
        "session_week.html",
        sid=sid, week=week,
        dets=dets,
        days=days,
        week_start=wsum.get("week_start_date"),
        backend=BACKEND
    )
@app.get("/sessions/<int:sid>/weeks/<int:week>/report")
def session_week_report(sid, week):
    url = f"{BACKEND}/api/sessions/{sid}/weeks/{week}/report"
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        return f"Week report failed (HTTP {r.status_code}).", r.status_code
    filename = r.headers.get("X-Filename", f"session_{sid}_week_{week}.pdf")
    filename = secure_filename(filename)
    return Response(
        r.content,
        headers={
            "Content-Type": "application/pdf",
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
        status=200
    )

@app.post("/sessions/<int:sid>/weeks/add")
def session_add_week(sid):
    week = request.form.get("week", type=int)
    if not week:
        flash("Please enter a week number (1–52)")
        return redirect(url_for("session_detail", sid=sid))

    r = post_json(f"{BACKEND}/api/sessions/{sid}/weeks/plan", {"week": week})
    ok = (r is not None and r.ok)
    app.logger.info(f"[UI] ADD planned week sid={sid} week={week} ok={ok}")
    if ok:
        flash("Week added.")
    else:
        flash(f"Failed to add week. ({r.status_code if r else 'no response'})")
    return redirect(url_for("session_detail", sid=sid))

@app.post("/sessions/<int:sid>/weeks/<int:week>/delete")
def session_delete_week(sid, week):
    r = delete_call(f"{BACKEND}/api/sessions/{sid}/weeks/{week}")
    ok = (r is not None and r.ok)
    app.logger.info(f"[UI] DELETE planned week sid={sid} week={week} ok={ok}")
    if ok:
        flash("Week deleted.")
    else:
        flash(f"Failed to delete week. ({r.status_code if r else 'no response'})")
    return redirect(url_for("session_detail", sid=sid))

@app.post("/sessions")
def create_session():
    name  = request.form.get("name","Session")
    start_week = request.form.get("start_week","1")
    r = post_json(f"{BACKEND}/api/sessions", {"name": name, "start_week": int(start_week)})
    app.logger.info(f"[UI] create_session name={name!r} start_week={start_week} -> {r.status_code if r else 'no response'}")
    return redirect(url_for("sessions"))

@app.post("/sessions/<int:sid>/set_week")
def session_set_week(sid):
    w = request.form.get("current_week", type=int)
    r = requests.patch(f"{BACKEND}/api/sessions/{sid}", json={"current_week": w})
    app.logger.info(f"[UI] set_current_week sid={sid} -> {w} (HTTP {r.status_code})")
    return redirect(url_for("session_detail", sid=sid))



@app.post("/sessions/<int:sid>/week/<int:week>/gemini")
def session_week_gemini(sid, week):
    r = post_json(f"{BACKEND}/api/sessions/{sid}/weeks/{week}/reasoning_gemini")
    app.logger.info(f"[UI] gemini_reason sid={sid} week={week} -> {r.status_code if r else 'no response'}")
    return redirect(url_for("session_week", sid=sid, week=week))


@app.post("/sessions/<int:sid>/week/<int:week>/bulk")
def session_week_bulk(sid, week):
    files = request.files.getlist("images")
    if not files:
        flash("Please choose at least one image.")
        return redirect(url_for("session_week", sid=sid, week=week))

    day = request.form.get("day", "")
    mfiles = [("images", (f.filename, f.stream, f.mimetype)) for f in files]
    data = {"day": day}

    t0 = time.time()
    try:
        r = requests.post(
            f"{BACKEND}/api/sessions/{sid}/weeks/{week}/bulk",
            data=data, files=mfiles, timeout=120
        )
        dt = (time.time() - t0) * 1000
        app.logger.info(f"[UI] BULK sid={sid} week={week} files={len(files)} day={day!r} -> {r.status_code} in {dt:.1f}ms")
        r.raise_for_status()
        flash("Images uploaded and queued for processing.")
    except Exception as e:
        app.logger.exception(f"[UI] BULK upload failed sid={sid} week={week}: {e}")
        flash(f"Upload failed: {e}")

    return redirect(url_for("session_week", sid=sid, week=week))

@app.post("/sessions/<int:sid>/reason")
def add_reasoning(sid):
    text = request.form.get("text","")
    r = post_json(f"{BACKEND}/api/sessions/{sid}/reasoning", {"text": text})
    app.logger.info(f"[UI] add_reasoning sid={sid} len(text)={len(text)} -> {r.status_code if r else 'no response'}")
    return redirect(url_for("session_detail", sid=sid))

@app.post("/sessions/<int:sid>/report")
def make_report(sid):
    r = post_json(f"{BACKEND}/api/sessions/{sid}/report")
    app.logger.info(f"[UI] make_report sid={sid} -> {r.status_code if r else 'no response'}")
    return redirect(url_for("session_detail", sid=sid))

@app.get("/history")
def history():
    dets = fetch_json(f"{BACKEND}/api/detections", [])
    return render_template("history.html", dets=dets, backend=BACKEND)

@app.get("/ginger/<int:det_id>")
def ginger_detail(det_id):
    d = fetch_json(f"{BACKEND}/api/detections/{det_id}", {})
    return render_template("ginger_detail.html", d=d, backend=BACKEND,  sid=d.get("session_id"))

@app.post("/ginger/<int:det_id>/reason")
def ginger_reason(det_id):
    text = request.form.get("text","")
    r = post_json(f"{BACKEND}/api/detections/{det_id}/reasoning", {"text": text})
    app.logger.info(f"[UI] ginger_reason det_id={det_id} -> {r.status_code if r else 'no response'}")
    return redirect(url_for("ginger_detail", det_id=det_id))

@app.post("/ginger/<int:det_id>/report")
def ginger_report(det_id):
    r = post_json(f"{BACKEND}/api/detections/{det_id}/report")
    app.logger.info(f"[UI] ginger_report det_id={det_id} -> {r.status_code if r else 'no response'}")
    return redirect(url_for("ginger_detail", det_id=det_id))

@app.template_filter("datetime")
def _fmt_dt(ts):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%H:%M")
    except Exception:
        return ts

@app.post("/sessions/<int:sid>/delete")
def ui_delete_session(sid):
    r = _try_delete(f"{BACKEND}/api/sessions/{sid}")
    app.logger.info(f"[UI] delete_session sid={sid} -> {(r.status_code if r else 'no response')}")
    return redirect(url_for("sessions"))

@app.post("/ginger/<int:det_id>/delete")
def ui_delete_detection(det_id):
    next_url = request.form.get("next") or url_for("history")
    r = _try_delete(f"{BACKEND}/api/detections/{det_id}")
    app.logger.info(f"[UI] delete_detection det_id={det_id} -> {(r.status_code if r else 'no response')}")
    return redirect(next_url)

@app.post("/sessions/<int:sid>/weeks/<int:week>/detections/<int:det_id>/delete")
def ui_delete_detection_in_week(sid, week, det_id):
    r = _try_delete(f"{BACKEND}/api/detections/{det_id}")
    app.logger.info(f"[UI] delete_detection_in_week sid={sid} week={week} det_id={det_id} -> {(r.status_code if r else 'no response')}")
    return redirect(url_for("session_week", sid=sid, week=week))

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
