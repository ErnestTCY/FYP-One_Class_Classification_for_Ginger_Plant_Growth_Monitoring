from runpy import run_path
import os, io, json, uuid
from queue import Queue, Empty
from threading import Thread
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask import send_from_directory
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from flask_sqlalchemy import SQLAlchemy
from database import db, Session, Detection, Reasoning, Report
from inference_engine import InferenceEngine, PhaseModel
from model_paths import phase_ckpt_map, PHASE_EARLY, PHASE_VG, PHASE_BP, PHASE_RM
from report_gen import make_report
from datetime import datetime,timedelta
from week_utils import phase_from_week
import os
from gemini_client import gemini_reasoning
import time, requests
from pathlib import Path
import logging,traceback
from werkzeug.exceptions import NotFound
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from flask import send_file, abort
import textwrap 
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.units import mm
# --- Histogram helpers -------------------------------------------------------
from collections import Counter

LABELS_HIST = ["Normal plant", "Abnormal plant", "Empty Bag Detected"]


WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(Path(__file__).with_name(".env")) 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)

@app.before_request
def _log_incoming():
    try:
        body = request.get_json(silent=True)
    except Exception:
        body = None
    app.logger.info(f"[API IN] {request.method} {request.path} args={dict(request.args)} json={body}")

CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # optional
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///ginger_occ.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
with app.app_context():
    db.create_all()
    try:
        from sqlalchemy import inspect, text
        insp = inspect(db.engine)
        cols = [c["name"] for c in insp.get_columns("session")]
        if "details" not in cols:
            db.session.execute(text("ALTER TABLE session ADD COLUMN details TEXT"))
            db.session.commit()
            app.logger.info("[DB] Added 'details' column to 'session' table.")
    except Exception as e:
        app.logger.warning(f"[DB] Could not ensure 'details' column: {e}")

_WEATHER_CACHE = {"ts": 0, "data": None}
_WEATHER_TTL   = 600  # seconds
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR   = os.getenv("UPLOAD_DIR",   "./uploads");   os.makedirs(UPLOAD_DIR, exist_ok=True)
RESULTS_DIR  = os.getenv("RESULTS_DIR",  "./results");   os.makedirs(RESULTS_DIR, exist_ok=True)
ARTIFACTS_DIR= os.getenv("ARTIFACTS_DIR","./artifacts"); os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Inference setup (load YOLO + three MAML checkpoints)
yolo_w = os.getenv("YOLO_WEIGHTS", "models/yolo_ginger_bag.pt")
conf_thresh = float(os.getenv("CONF_THRESH", "0.5"))
area_thresh = int(os.getenv("AREA_THRESH", "500"))
tau_override= float(os.getenv("TAU_OVERRIDE", "0.6"))
tau_scale   = float(os.getenv("TAU_SCALE",   "0.9"))
box_width   = int(os.getenv("BOX_WIDTH",     "6"))

phase_models = {}
for p, ck in phase_ckpt_map().items():
    if ck and os.path.exists(ck):
        phase_models[p] = PhaseModel(ck, tau_override=tau_override, tau_scale=tau_scale)

ENGINE = InferenceEngine(yolo_w, phase_models, conf_thresh=conf_thresh, area_thresh=area_thresh, box_width=box_width)

# Background job infra
JOB_Q = Queue()
JOB_RES = {}
def _sess_weeks_planned(sess):
    d = sess.details or {}
    weeks = d.get("plan_weeks") or d.get("weeks_planned")
    if not weeks:
        weeks = [int(sess.start_week)]
    try:
        weeks = [int(w) for w in weeks]
    except Exception:
        weeks = [int(sess.start_week)]
    return sorted(set(weeks))

def _generate_gemini_reasoning(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini not configured. Set GEMINI_API_KEY to enable automatic reasoning."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        return text or "No content returned by Gemini."
    except Exception as e:
        return f"Generation error: {e}"
    
# shared builder so GET and POST (and both paths) behave the same
def _build_and_send_week_pdf(sid: int, week: int):
    sess = Session.query.get_or_404(sid)

    q = Detection.query.filter_by(session_id=sess.id)
    if hasattr(Detection, "week"):
        q = q.filter(Detection.week == week)
    dets = q.order_by(Detection.id.asc()).all()

    reports_dir = Path(ARTIFACTS_DIR) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / f"session_{sid}_week_{week}.pdf"

    # Use your actual week-report builder here
    # make_week_report(str(pdf_path), sess, dets, title=f"Week {week} Report")
    make_report(str(pdf_path), sess, dets, None, artifacts_root=".")

    if not pdf_path.exists():
        abort(500, "Report was not created")

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=pdf_path.name,
        max_age=0,
        etag=False,
        conditional=False,
    )

# ------------------ GEMINI HELPERS ------------------

def _gemini_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def _safe_iso_to_dt(val):
    """Accept datetime | str | None. Return datetime | None."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        s = val.strip()
        try:
            # handle 'Z'
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None

def _weather_line_for_llm(d):
    parts = []
    if getattr(d, "wx_summary", None): parts.append(str(d.wx_summary))
    if getattr(d, "wx_temp_c", None) is not None: parts.append(f"{d.wx_temp_c:.1f}°C")
    if getattr(d, "wx_humidity", None) is not None: parts.append(f"{d.wx_humidity}% RH")
    if getattr(d, "wx_rain_1h_mm", None) is not None: parts.append(f"{d.wx_rain_1h_mm} mm rain(1h)")
    if getattr(d, "wx_wind_kmh", None) is not None: parts.append(f"{d.wx_wind_kmh:.0f} km/h wind")
    return ", ".join(parts) if parts else "n/a"

def _call_gemini_flash(prompt_text, temperature=0.4, timeout=25):
    """Call Gemini 1.5 Flash. Returns string on success, None on failure."""
    api_key = _gemini_key()
    if not api_key:
        return None
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    try:
        r = requests.post(
            url,
            params={"key": api_key},
            json={"contents": [{"parts":[{"text": prompt_text}]}]},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        # extract text
        candidates = (data.get("candidates") or [])
        for c in candidates:
            parts = (((c.get("content") or {}).get("parts")) or [])
            for p in parts:
                t = p.get("text")
                if t:
                    return t.strip()
        return None
    except Exception:
        return None

def _build_overall_gemini_summary(sess, detections, scope_label):
    """
    Compose a compact prompt from saved reasoning+weather and ask Gemini to summarize.
    scope_label: 'Session' or 'Week X'
    """
    items = []
    for d in detections:
        dt = d.details or {}
        reason = None
        if isinstance(dt, dict):
            reason = dt.get("reasoning_text")
        ts = _safe_iso_to_dt(getattr(d, "ts", None))
        ts_s = ts.strftime("%Y-%m-%d %H:%M") if ts else str(getattr(d, "ts", ""))
        items.append(
            f"- Det #{d.id} | {ts_s} | Verdict: {getattr(d,'verdict','?')} | "
            f"Phase: {getattr(d,'phase','?')} | Weather: {_weather_line_for_llm(d)}\n"
            f"  Reasoning: { (reason.strip() if reason else '—') }"
        )
    if not items:
        return None

    prompt = (
        f"You are an agronomy assistant for ginger plants. Summarize the following per-plant observations for this {scope_label}. "
        f"Give: 1) overall health status and notable issues, 2) environmental factors that likely influenced it (use listed weather), "
        f"3) concise recommendations for the next few days. Avoid repeating every detail; synthesize.\n\n"
        "OBSERVATIONS:\n" + "\n".join(items)
    )
    return _call_gemini_flash(prompt)

# ------------------ PDF BUILDERS ------------------

def _format_weather_line(d):
    parts = []
    if getattr(d, "wx_summary", None): parts.append(str(d.wx_summary))
    if getattr(d, "wx_temp_c", None) is not None: parts.append(f"{d.wx_temp_c:.1f}°C")
    if getattr(d, "wx_humidity", None) is not None: parts.append(f"{d.wx_humidity}%")
    if getattr(d, "wx_rain_1h_mm", None) is not None: parts.append(f"{d.wx_rain_1h_mm} mm (1h)")
    if getattr(d, "wx_wind_kmh", None) is not None: parts.append(f"{d.wx_wind_kmh:.0f} km/h")
    return " | ".join(parts) if parts else "—"

def _image_fs(path: str):
    if not path: return None
    p1 = os.path.join(app.root_path, path)
    if os.path.exists(p1): return p1
    p2 = os.path.join(app.root_path, "static", path)
    if os.path.exists(p2): return p2
    return None

def _wrap_lines(text, width=100):
    lines = []
    for para in (text or "").splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, replace_whitespace=False))
    return lines

def _group_detections(detections, group_by):
    """
    group_by: 'week' -> dict[week_number] = [d,...]
              'day'  -> dict['Mon, 2025-09-18'] = [d,...]
              None   -> {'All': [d,...]}
    """
    out = defaultdict(list)
    if group_by == "week":
        for d in detections:
            wk = getattr(d, "week", None)
            key = f"Week {wk}" if wk is not None else "Week —"
            out[key].append(d)
        # order by numeric week if possible
        def key_sort(k):
            try:
                return int(k.split()[-1])
            except Exception:
                return 10**9
        return dict(sorted(out.items(), key=lambda kv: key_sort(kv[0])))

    if group_by == "day":
        for d in detections:
            ts = _safe_iso_to_dt(getattr(d, "ts", None))
            if ts:
                key = ts.strftime("%a, %Y-%m-%d")
            else:
                key = "Unknown day"
            out[key].append(d)
        return dict(sorted(out.items(), key=lambda kv: kv[0]))
    # default
    out["All"] = list(detections)
    return out

def _build_detections_report_pdf(sess, detections, title="Detection Report", group_by=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from collections import defaultdict
    import io, os

    def _wrap_lines(text, width=80):
        import textwrap
        return textwrap.wrap(text, width=width)

    buffer = io.BytesIO()
    W, H = pagesize = A4
    c = canvas.Canvas(buffer, pagesize=pagesize)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, H - 50, title)
    y = H - 80

    # Session metadata
    if sess:
        sid = getattr(sess, "id", "N/A")
        sname = getattr(sess, "name", "N/A")
        sphase = getattr(sess, "phase", "N/A")
        sstart = getattr(sess, "started_at", "N/A")
        c.setFont("Helvetica", 10)
        c.drawString(30, y, f"Session ID: {sid}  |  Name: {sname}  |  Phase: {sphase}")
        y -= 14
        c.drawString(30, y, f"Started: {sstart}")
        y -= 20

    # Summary counts
    verdict_counts = defaultdict(int)
    for dt in detections:
        verdict = getattr(dt, "verdict", "Unknown")
        verdict_counts[verdict] += 1

    c.setFont("Helvetica-Bold", 11)
    c.drawString(30, y, "Summary counts")
    y -= 14
    c.setFont("Helvetica", 10)
    for verdict in ["Normal", "Abnormal", "Empty", "Unknown"]:
        count = verdict_counts.get(verdict, 0)
        c.drawString(40, y, f"{verdict}: {count}")
        y -= 12
    y -= 12

    # Gemini summary
    gem_text = ""
    if sess and detections:
        try:
            gem_text = _build_overall_gemini_summary(sess, detections, scope=title)
        except Exception as e:
            print("Gemini summary failed:", e)

    # Defect rate
    defect_rate = 0.0
    total = len(detections)
    abnormal = sum(1 for d in detections if "Abnormal" in getattr(d, "verdict", ""))
    if total > 0:
        defect_rate = (abnormal / total) * 100

    # Green summary box
    box_x = 30
    box_y = y - 10
    box_w = W - 2 * box_x
    box_h = 120
    c.setFillColorRGB(0.85, 1.0, 0.85)
    c.roundRect(box_x, box_y, box_w, box_h, radius=10, stroke=0, fill=1)

    c.setFillColorRGB(0, 0.4, 0)
    c.setFont("Helvetica-Bold", 12)
    label = "Session Summary" if "Session" in title else "Week Summary"
    c.drawString(box_x + 10, box_y + box_h - 20, f"{label} (Gemini + Defect Rate)")

    c.setFont("Helvetica", 10)
    c.drawString(box_x + 10, box_y + box_h - 38, f"Detected Images: {total}")
    c.drawString(box_x + 10, box_y + box_h - 52, f"Abnormal Detections: {abnormal}")
    c.drawString(box_x + 10, box_y + box_h - 66, f"Defect Rate: {defect_rate:.2f}%")

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.1, 0.3, 0.1)
    gy = box_y + box_h - 84
    for line in _wrap_lines(gem_text or "No Gemini summary available.", width=85):
        c.drawString(box_x + 10, gy, line)
        gy -= 12

    c.setFillColor(colors.black)
    y = box_y - 30

    # Grouping
    if group_by == "week":
        grouped = defaultdict(list)
        for d in detections:
            grouped[getattr(d, "week", 0)].append(d)
        groups = sorted(grouped.items())
    else:
        groups = [(None, detections)]

    for group, group_dets in groups:
        if group_by == "week":
            c.setFont("Helvetica-Bold", 13)
            c.drawString(30, y, f"Week {group}")
            y -= 20

        for d in group_dets:
            imgpath = getattr(d, "annotated_path", None) or getattr(d, "image_path", None)
            imgfile = os.path.join(UPLOAD_DIR, imgpath) if imgpath else None
            if imgfile and os.path.exists(imgfile):
                try:
                    c.drawImage(imgfile, 30, y - 180, width=160, height=120, preserveAspectRatio=True)
                except Exception as e:
                    print("Image draw failed:", e)

            c.setFont("Helvetica-Bold", 10)
            c.drawString(200, y - 10, f"Det #{getattr(d, 'id', '?')} — {getattr(d, 'verdict', '?')} — {getattr(d, 'phase', '?')}")

            c.setFont("Helvetica", 9)
            try:
                ts = d.get("ts", "N/A")
            except Exception:
                ts = getattr(d, "ts", "N/A")
            c.drawString(200, y - 25, f"Time: {ts}")

            try:
                wx = d.get("weather", {}) or {}
            except Exception:
                wx = getattr(d, "weather", {}) or {}
            wtext = f"{wx.get('summary', 'N/A')} | {wx.get('temp_c', '?')}°C | {wx.get('humidity', '?')}% | {wx.get('rain_1h', '?')} mm | {wx.get('wind_kph', '?')} km/h"
            c.drawString(200, y - 40, f"Weather: {wtext}")

            try:
                reasoning = d.get("reasoning_text") or d.get("reasoning") or "No reasoning provided."
            except Exception:
                reasoning = getattr(d, "reasoning_text", None) or getattr(d, "reasoning", None) or "No reasoning provided."

            ry = y - 58
            c.setFont("Helvetica-Oblique", 8)
            for line in _wrap_lines(reasoning, width=90):
                c.drawString(200, ry, line)
                ry -= 10

            y -= 200
            if y < 200:
                c.showPage()
                y = H - 60

        y -= 20

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if "Week" in title:
        fname = f"report_weekly_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    elif "Session" in title:
        fname = f"report_session_{sess.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    else:
        fname = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    return pdf_bytes, fname


def _ensure_week_meta(sess, week: int):
    """Make sure sess.details['week_meta'][str(week)] exists with phase."""
    # normalize details
    raw = getattr(sess, "details", None)
    if isinstance(raw, str):
        try:
            details = json.loads(raw) if raw else {}
        except Exception:
            details = {}
    elif isinstance(raw, dict):
        details = dict(raw)
    else:
        details = {}

    meta = details.get("week_meta") or {}
    wkey = str(int(week))
    if wkey not in meta:
        meta[wkey] = {
            "phase": phase_from_week(int(week)),  # your existing function
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        details["week_meta"] = meta
        sess.details = details
        db.session.add(sess)


def _rm(relpath: str):
    """Remove a relative file if it exists."""
    if not relpath:
        return
    relpath = relpath.replace("\\", "/")
    p = os.path.join(BASE_DIR, relpath)
    try:
        if os.path.isfile(p):
            os.remove(p)
            app.logger.info(f"[DEL] file removed: {relpath}")
    except Exception as e:
        app.logger.warning(f"[DEL] file remove failed {relpath}: {e}")
        

def _get_plan_weeks(sess):
    raw = getattr(sess, "details", None)
    if isinstance(raw, str):
        try:
            details = json.loads(raw) if raw else {}
        except Exception:
            details = {}
    elif isinstance(raw, dict):
        details = raw
    else:
        details = {}

    weeks = details.get("plan_weeks") or details.get("weeks_planned") or []
    # bootstrap default if empty
    sw = int(getattr(sess, "start_week", 1) or 1)
    cw = int(getattr(sess, "current_week", sw) or sw)
    if not weeks and cw >= sw:
        weeks = list(range(sw, cw + 1))

    out = []
    for w in weeks:
        try:
            out.append(int(w))
        except Exception:
            pass
    return sorted(set(out))

def _set_plan_weeks(sess, weeks):
    uniq = sorted({int(w) for w in weeks if str(w).isdigit()})
    raw = getattr(sess, "details", None)
    if isinstance(raw, str):
        try:
            details = json.loads(raw) if raw else {}
        except Exception:
            details = {}
    elif isinstance(raw, dict):
        details = dict(raw)
    else:
        details = {}
    details["plan_weeks"] = uniq
    sess.details = details
    db.session.add(sess)




def _week_start_dt(sess, week: int) -> datetime:
    """Absolute datetime for start of the given week (# respects sess.start_week)."""
    base = sess.started_at if isinstance(sess.started_at, datetime) \
           else datetime.fromisoformat(str(sess.started_at).replace("Z", ""))
    start_week = int(getattr(sess, "start_week", 1) or 1)
    delta_days = (int(week) - start_week) * 7
    return base + timedelta(days=delta_days)

def _norm(p):
    return p.replace("\\", "/") if isinstance(p, str) else p

def _ginger_advisory(current):
    """Return simple agronomic advisories for ginger."""
    adv = []
    t = current.get("temp")
    rh = current.get("humidity")
    uvi = current.get("uvi")
    wind = current.get("wind_speed")  # m/s
    rain1h = current.get("rain_1h", 0.0)

    if t is not None and rh is not None:
        if t >= 32 and rh <= 45:
            adv.append("Heat & dry stress: irrigate and mulch to conserve moisture.")
        if t >= 28 and rh >= 85:
            adv.append("Hot & humid: elevated fungal risk (leaf spot). Improve airflow, avoid evening overhead watering.")
        if t <= 15:
            adv.append("Cool conditions: slow growth risk. Consider protection or later irrigations.")
    if rain1h and rain1h >= 2:
        adv.append("Heavy rain: watch for waterlogging. Ensure drainage; avoid compacted soils.")
    if uvi is not None and uvi >= 7:
        adv.append("High UV: consider partial shade during midday to reduce leaf scorch.")
    if wind and wind >= 10:  # ~36 km/h
        adv.append("Strong wind: provide windbreak/support to avoid lodging.")
    if not adv:
        adv.append("Conditions acceptable. Maintain regular irrigation and monitor for pests/disease.")
    return adv


def _mask(s):
    if not s: return s
    return s[:4] + "…" + s[-4:] if len(s) > 8 else "***"

def _classify_item(it) -> str | None:
    """
    Normalize a single detection item into one of:
      'Normal plant' | 'Abnormal plant' | 'Empty Bag Detected' | None
    Accepts dicts OR simple tuple/list forms like ('plant', ...).
    """
    # Coerce simple tuple/list: assume first element is a label
    if not isinstance(it, dict):
        try:
            lbl = str(it[0]).lower()
            if "empty" in lbl and "bag" in lbl:
                return "Empty Bag Detected"
            if "abnormal" in lbl or "diseased" in lbl:
                return "Abnormal plant"
            if any(k in lbl for k in ("plant", "ginger", "shoot", "sprout", "seedling")):
                return "Normal plant"
        except Exception:
            pass
        return None

    typ    = (it.get("type") or "").lower()          # e.g. 'bag', 'plant'
    label  = (it.get("label") or it.get("cls_name") or "").lower()
    status = (it.get("status") or it.get("health") or "").lower()

    # Empty bag
    if "bag" in typ or "bag" in label:
        if it.get("is_empty") is True or "empty" in label:
            return "Empty Bag Detected"
        return None  # non-empty bag handled by caller when needed

    # Plant-like
    if "plant" in typ or any(k in label for k in ("ginger", "plant", "shoot", "sprout", "seedling")):
        if "abnormal" in label or "diseased" in label or "abnormal" in status or "diseased" in status:
            return "Abnormal plant"
        return "Normal plant"

    return None


def _extract_items_from_details(details):
    """Support list or dict-with-items shapes."""
    if isinstance(details, list):
        return details
    if isinstance(details, dict):
        if "items" in details and isinstance(details["items"], list):
            return details["items"]
        # Some code stores the list directly under another key:
        if "detections" in details and isinstance(details["detections"], list):
            return details["detections"]
    return []

def _counts_from_row(row) -> tuple[int, int, int]:

    det = getattr(row, "details", None)

    # If it's a JSON string, parse it
    if isinstance(det, str):
        try:
            det = json.loads(det) if det else {}
        except Exception:
            det = {}

    # Precomputed summary?
    if isinstance(det, dict) and "counts" in det and isinstance(det["counts"], dict):
        c = det["counts"]
        return (
            int(c.get("normal", 0)),
            int(c.get("abnormal", 0)),
            int(c.get("empty_bag", c.get("empty", 0))),
        )

    # Otherwise derive from items
    items = _extract_items_from_details(det)
    normal = abnormal = empty_bag = 0
    for it in items:
        bucket = _classify_item(it)
        if bucket == "Normal plant":
            normal += 1
        elif bucket == "Abnormal plant":
            abnormal += 1
        elif bucket == "Empty Bag Detected":
            empty_bag += 1
    return (normal, abnormal, empty_bag)


def _hist_from_rows(rows):
    total = Counter({"Normal plant":0, "Abnormal plant":0, "Empty Bag Detected":0})
    for r in rows:
        n,a,e = _counts_from_row(r)
        total["Normal plant"] += n
        total["Abnormal plant"] += a
        total["Empty Bag Detected"] += e
    return [total["Normal plant"], total["Abnormal plant"], total["Empty Bag Detected"]]


def _wxcode_to_emoji(code):
    try:
        code = int(code)
    except Exception:
        return "🌡️"
    return {
        0:"☀️", 1:"🌤️", 2:"⛅", 3:"☁️",
        45:"🌫️", 48:"🌫️",
        51:"🌦️", 53:"🌦️", 55:"🌧️",
        61:"🌧️", 63:"🌧️", 65:"🌧️",
        66:"🌧️", 67:"🌧️",
        71:"🌨️", 73:"🌨️", 75:"❄️",
        77:"🌨️",
        80:"🌧️", 81:"🌧️", 82:"🌧️",
        85:"🌨️", 86:"🌨️",
        95:"⛈️", 96:"⛈️", 99:"⛈️",
    }.get(code, "🌡️")

def _owm_fetch(lat, lon, units, api_key):
    base = "https://api.openweathermap.org/data/2.5"
    common = {"lat": lat, "lon": lon, "appid": api_key, "units": units or "metric"}

    rc = requests.get(f"{base}/weather", params=common, timeout=10)
    app.logger.info(f"[WX] OWM free current status={rc.status_code}")
    rc.raise_for_status()
    cur = rc.json()

    rf = requests.get(f"{base}/forecast", params=common, timeout=10)
    app.logger.info(f"[WX] OWM free forecast status={rf.status_code}")
    rf.raise_for_status()
    fc = rf.json()

    w0   = (cur.get("weather") or [{}])[0]
    icon = w0.get("icon")                 # e.g. '04d'
    city = cur.get("name") or _owm_reverse_geocode(lat, lon, api_key) or _nominatim_reverse(lat, lon)

    # 🔹 bigger icon (128px) and emoji fallback
    emoji_map = {"01":"☀️","02":"🌤️","03":"⛅","04":"☁️","09":"🌧️","10":"🌦️","11":"⛈️","13":"❄️","50":"🌫️"}
    icon_url  = f"https://openweathermap.org/img/wn/{icon}@4x.png" if icon else None
    icon_emoji = emoji_map.get(icon[:2]) if icon else None

    current = {
        "summary": w0.get("description", "").title(),
        "temp": (cur.get("main") or {}).get("temp"),
        "humidity": (cur.get("main") or {}).get("humidity"),
        "wind_speed": (cur.get("wind") or {}).get("speed"),
        "uvi": None,
        "rain_1h": (cur.get("rain") or {}).get("1h", 0.0),
        "dt": cur.get("dt"),
        "units": units or "metric",
        "icon": icon,
        "icon_url": icon_url,
        "icon_emoji": icon_emoji,
        "location_name": city or "Your location",
    }

    hours = []
    for it in (fc.get("list") or [])[:4]:  # ~12h (3h steps)
        hours.append({
            "dt": it.get("dt"),
            "humidity": (it.get("main") or {}).get("humidity"),
            "pop": it.get("pop"),
            "temp": (it.get("main") or {}).get("temp"),
        })

    return current, hours


def _iso_date(dt_or_str):
    if not dt_or_str:
        return ""
    if isinstance(dt_or_str, str):
        # "2025-09-16T18:41:53.632474" -> "2025-09-16"
        return dt_or_str.split("T")[0]
    try:
        return dt_or_str.date().isoformat()
    except Exception:
        return str(dt_or_str)[:10]

def _week_start_date(sess, week:int) -> str:
    """Week-1 begins on 'started_at' day, each week = 7 days."""
    base = sess.started_at
    if isinstance(base, str):
        # for safety
        base = datetime.fromisoformat(base.replace("Z", ""))
    start = base + timedelta(days=(int(week)-1)*7)
    return start.date().isoformat()

def _openmeteo_fetch(lat, lon, units):
    from datetime import datetime
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "hourly": "relative_humidity_2m,precipitation_probability,temperature_2m,weather_code",
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=10)
    app.logger.info(f"[WX] Open-Meteo status={r.status_code}")
    r.raise_for_status()
    j = r.json()

    curj = j.get("current", {}) or {}
    code = curj.get("weather_code")

    current = {
        "summary": "Local conditions",
        "temp": curj.get("temperature_2m"),
        "humidity": curj.get("relative_humidity_2m"),
        "wind_speed": curj.get("wind_speed_10m"),
        "uvi": None,
        "rain_1h": 0.0,
        "dt": None,
        "units": "metric" if (units or "metric") == "metric" else "imperial",
        "icon_emoji": _wxcode_to_emoji(code),
        "icon_url": None,
        "location_name": j.get("timezone") or "Your location",
    }

    hours = []
    times = (j.get("hourly") or {}).get("time", []) or []
    rh    = (j.get("hourly") or {}).get("relative_humidity_2m", []) or []
    pop   = (j.get("hourly") or {}).get("precipitation_probability", []) or []
    temp  = (j.get("hourly") or {}).get("temperature_2m", []) or []
    n = min(12, len(times), len(rh), len(temp))
    for i in range(n):
        ts = None
        try:
            ts = int(datetime.fromisoformat(times[i].replace("Z", "+00:00")).timestamp())
        except Exception:
            pass
        hours.append({"dt": ts, "humidity": rh[i], "pop": (pop[i] if i < len(pop) else None), "temp": temp[i]})
    return current, hours

@app.post("/api/chat")
def api_chat():
    """
    Lightweight chat endpoint. Request JSON:
    {
      "messages": [{"role":"user"/"assistant"/"system","content":"..."}],
      "include_weather": true|false
    }
    We do not persist history on the server — client sends it each call.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        msgs = data.get("messages", [])
        include_weather = bool(data.get("include_weather"))

        # Optional context: current local weather to ground answers.
        weather_blob = ""
        if include_weather:
            try:
                # reuse your existing weather endpoint logic
                w = api_weather()  
                if w and "current" in w:
                    cur = w["current"]
                    weather_blob = (
                        f"\n\n[Current Weather]\n"
                        f"Summary: {cur.get('summary')}\n"
                        f"Temp: {cur.get('temp')}°{ 'C' if cur.get('units')=='metric' else 'F'}\n"
                        f"Humidity: {cur.get('humidity')}%\n"
                        f"Rain(1h): {cur.get('rain_1h', 0)} mm\n"
                        f"Wind: {round(cur.get('wind_speed',0)*3.6)} km/h\n"
                    )
            except Exception:
                weather_blob = ""

        # Compose a safe system prompt that keeps the model on ginger-care topic
        system_prompt = (
            "You are a helpful agronomy assistant focused on ginger (Zingiber officinale). "
            "Give short, practical answers with bullet points when useful. "
            "Consider growth phase differences (Sprouting, VegetativeGrowth, BulkingPhase, Maturity). "
            "Default climate is tropical monsoon unless user states otherwise."
            f"{weather_blob}"
            "\nIf the user asks for diagnosis, list likely causes and quick checks. "
            "If the user asks for actions, give clear steps (1–2 sentences per step). "
            "Never invent measurements; if unknown, say so."
        )

        # Build the Gemini-style message list
        # We’ll prepend a system message and then the user/assistant turns sent by the client.
        chat = [{"role":"system","content":system_prompt}]
        for m in msgs:
            role = m.get("role","user")
            content = (m.get("content") or "").strip()
            if content:
                chat.append({"role": role, "content": content})

        # Call Gemini (same lib as your gen_reason)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return jsonify({"ok": False, "message": "Gemini API key not configured."}), 500

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Turn our list into a single prompt (Gemini Python chat API also supports histories,
        # but this keeps the server stateless—history stays on client).
        prompt = ""
        for turn in chat:
            tag = turn["role"].upper()
            prompt += f"\n\n[{tag}]\n{turn['content']}"
        prompt += "\n\n[ASSISTANT]\n"

        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()

        return jsonify({"ok": True, "reply": text})

    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500
    
def _haversine_m(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def _nominatim_reverse(lat, lon):
    """OSM reverse geocode (no key)."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format":"jsonv2", "lat":lat, "lon":lon, "zoom":14, "addressdetails":1},
            headers={"User-Agent":"ginger-monitor/1.0"},
            timeout=8,
        )
        if r.ok:
            j = r.json()
            addr = j.get("address", {}) or {}
            # Prefer fine-grained locality names
            name = addr.get("neighbourhood") or addr.get("suburb") or addr.get("village") or \
                   addr.get("town") or addr.get("city") or addr.get("county")
            state   = addr.get("state")
            cc      = (addr.get("country_code") or "").upper()
            if name:
                if state and cc:
                    return f"{name}, {state} {cc}"
                if state:
                    return f"{name}, {state}"
                if cc:
                    return f"{name}, {cc}"
                return name
    except Exception:
        pass
    return None

def _owm_reverse_geocode(lat, lon, api_key):
    """OpenWeather reverse geocode with nearest-of-multiple selection."""
    try:
        r = requests.get(
            "https://api.openweathermap.org/geo/1.0/reverse",
            params={"lat": lat, "lon": lon, "limit": 5, "appid": api_key},
            timeout=8,
        )
        if r.ok:
            arr = r.json()
            if isinstance(arr, list) and arr:
                # pick the closest entry
                best = min(
                    arr,
                    key=lambda a: _haversine_m(lat, lon, a.get("lat", lat), a.get("lon", lon))
                )
                local = (best.get("local_names") or {}).get("en") or best.get("name")
                state = best.get("state")
                cc    = best.get("country")
                if local:
                    if state and cc:
                        return f"{local}, {state} {cc}"
                    if state:
                        return f"{local}, {state}"
                    if cc:
                        return f"{local}, {cc}"
                    return local
    except Exception:
        pass
    return None

# put near _week_start_date(...)
def _week_start_dt(sess, week: int) -> datetime:
    """Datetime for the start of the requested week (respects sess.start_week)."""
    base = sess.started_at if isinstance(sess.started_at, datetime) \
           else datetime.fromisoformat(str(sess.started_at).replace("Z",""))
    start_week = int(getattr(sess, "start_week", 1) or 1)
    delta_days = (int(week) - start_week) * 7
    return base + timedelta(days=delta_days)

def _weather_snapshot_now():
    """
    Returns a dict with keys: summary, temp_c, humidity, rain_1h_mm, wind_kmh
    Uses the existing /api/weather endpoint to avoid duplicating logic.
    Safe-fails to None on any error.
    """
    try:
        base = os.getenv("SELF_BASE_URL", "http://127.0.0.1:8001")  # your backend itself
        lat  = os.getenv("WEATHER_LAT", None)
        lon  = os.getenv("WEATHER_LON", None)
        params = {}
        if lat and lon:
            params.update({"lat": float(lat), "lon": float(lon)})
        params["nocache"] = 1

        r = requests.get(f"{base}/api/weather", params=params, timeout=4)
        j = r.json()
        if not j.get("ok"):
            return None

        cur   = j.get("current", {}) or {}
        units = (j.get("units") or "metric").lower()

        # Normalize to Celsius and km/h
        temp_c = cur.get("temp")
        if temp_c is not None and units == "imperial":  # F → C
            temp_c = (float(temp_c) - 32.0) / 1.8

        wind_kmh = None
        if cur.get("wind_speed") is not None:
            # our UI used m/s * 3.6 to show km/h
            wind_kmh = float(cur["wind_speed"]) * 3.6

        return {
            "summary":    cur.get("summary"),
            "temp_c":     float(temp_c) if temp_c is not None else None,
            "humidity":   cur.get("humidity"),
            "rain_1h_mm": float(cur.get("rain_1h") or 0.0),
            "wind_kmh":   float(wind_kmh) if wind_kmh is not None else None,
        }
    except Exception as e:
        app.logger.warning("[WX] snapshot failed: %s", e)
        return None
    
def worker_loop():
    with app.app_context():
        while True:
            job = JOB_Q.get()
            if job is None:
                break
            job_id = job["id"]
            try:
                img_bytes = job["image_bytes"]
                phase = job["phase"]
                session_id = job.get("session_id")
                week = int(job.get("week", 1))
                ts_override = job.get("ts_override")            # NEW

                # normalize ts_override to datetime if present
                if isinstance(ts_override, str):
                    try: ts_override = datetime.fromisoformat(ts_override)
                    except Exception: ts_override = None

                out = ENGINE.run(img_bytes, phase)
                # ENGINE.run should return detections + an annotated image (bytes or PIL). Adapt as needed:
                items = (
                    out.get("detections")
                    or out.get("items")
                    or out.get("objects")
                    or out.get("boxes")
                    or []
                )
                # count objects
# --- counts: prefer engine-provided counts if available ---
                normal_cnt = abnormal_cnt = empty_cnt = 0
                eng_counts = out.get("counts") or out.get("plant_counts") or out.get("stats")
                if isinstance(eng_counts, dict):
                    normal_cnt   = int(eng_counts.get("normal", 0))
                    abnormal_cnt = int(eng_counts.get("abnormal", 0))
                    empty_cnt    = int(eng_counts.get("empty_bag", eng_counts.get("empty", 0)))
                else:
                    # derive from returned items (dicts or tuples)
                    for it in (items or []):
                        b = _classify_item(it)
                        if b == "Normal plant":
                            normal_cnt += 1; continue
                        if b == "Abnormal plant":
                            abnormal_cnt += 1; continue
                        if b == "Empty Bag Detected":
                            empty_cnt += 1; continue

                        # ---- Early/Sprouting fallback ----
                        try:
                            from model_paths import PHASE_EARLY
                        except Exception:
                            PHASE_EARLY = "Early"

                        phase_lc = str(phase).lower() if phase is not None else ""
                        if (phase == PHASE_EARLY) or ("early" in phase_lc) or ("sprout" in phase_lc):
                            # Treat ANY non-empty bag as a normal plant in early phase
                            # (engines usually mark empties explicitly)
                            if isinstance(it, dict):
                                typ   = str(it.get("type") or "").lower()
                                label = str(it.get("label") or it.get("cls_name") or "").lower()
                                is_bag   = ("bag" in typ) or ("bag" in label)
                                is_empty = (it.get("is_empty") is True) or ("empty" in label)
                            else:
                                # tuple/list: rely on label text
                                lbl = str(it[0]).lower() if isinstance(it, (list, tuple)) and it else ""
                                is_bag   = "bag" in lbl
                                is_empty = "empty" in lbl
                            if is_bag and not is_empty:
                                normal_cnt += 1


                # decide verdict headline
                phase = phase  # your existing variable
                is_normal = bool(out.get("is_normal"))
                if not is_normal:
                    v0 = (out.get("verdict") or "")
                    is_normal = v0.lower().startswith("normal")

                if empty_cnt > 0:
                    out["verdict"] = f"Empty Bag Detected — {phase}"
                elif abnormal_cnt > 0:
                    out["verdict"] = f"Abnormal — {phase}"
                else:
                    out["verdict"] = f"Normal — {phase}"

                # store details with counts summary (works with both shapes)
                details_payload = {"items": items, "counts": {
                    "normal": normal_cnt,
                    "abnormal": abnormal_cnt,
                    "empty_bag": empty_cnt
                }}

                
                uid = uuid.uuid4().hex
                uploads_dir = Path("uploads"); uploads_dir.mkdir(parents=True, exist_ok=True)
                results_dir = Path("results"); results_dir.mkdir(parents=True, exist_ok=True)

                # Save original
                up_path = uploads_dir / f"orig_{uid}.png"
                with open(up_path, "wb") as f:
                    f.write(img_bytes)

                # Save annotated
                an_path = results_dir / f"annot_{uid}.png"

                # Try common keys your ENGINE might return
                annotated_bytes = (
                    out.get("annotated_bytes")
                    or out.get("annotated_png")
                    or out.get("vis_png")
                )

                if annotated_bytes:
                    with open(an_path, "wb") as f:
                        f.write(annotated_bytes)
                else:
                    # If ENGINE returned a PIL image
                    img = out.get("annotated_image") or out.get("vis_image")
                    if img is not None:
                        img.save(an_path)
                    else:
                        # FINAL FALLBACK: save the original bytes so at least something shows
                        with open(an_path, "wb") as f:
                            f.write(img_bytes)

                # after saving up_path/an_path
                image_path_rel = _norm(os.path.relpath(up_path, start="."))
                annot_path_rel = _norm(os.path.relpath(an_path, start="."))
                wx = _weather_snapshot_now()

                det = Detection(
                    phase=phase,
                    week=week,                         # <-- not week_number
                    verdict=out["verdict"],
                    image_path=image_path_rel,
                    annotated_path=annot_path_rel,
                    details=details_payload,
                    is_manual=False,
                    session_id=session_id,
                    wx_summary = wx and wx.get("summary"),
                    wx_temp_c = wx and wx.get("temp_c"),
                    wx_humidity = wx and wx.get("humidity"),
                    wx_rain_1h_mm = wx and wx.get("rain_1h_mm"),
                    wx_wind_kmh = wx and wx.get("wind_kmh"),
                )



                db.session.add(det); db.session.commit()

                JOB_RES[job_id] = {
                    "status": "done",
                    "detection_id": det.id,
                    "verdict": det.verdict,
                    "week": week
                }
            except Exception as e:
                JOB_RES[job_id] = {"status": "failed", "error": str(e)}


Thread(target=worker_loop, daemon=True).start()

def generate_week_reasoning_fallback(detections):
    total = len(detections)
    abn   = sum(1 for d in detections if d.verdict.lower()=="abnormal")
    empty_bags = 0
    for d in detections:
        for it in (d.details or []):
            if it.get("type")=="bag" and "Empty" in it.get("label",""):
                empty_bags += 1
    return (
        f"Weekly summary: {total} images processed; {abn} abnormal detections. "
        f"Empty bags observed: {empty_bags}. "
        f"Recommendation: remove empty bags promptly; re-check irrigation and pests for abnormal plants."
    )

# ===== API =====
@app.delete("/api/sessions/<int:sid>")
@app.post("/api/sessions/<int:sid>/delete")  # HTML form fallback
def api_delete_session(sid):
    """Delete a session and all its detections/children."""
    sess = Session.query.get(sid)
    if not sess:
        raise NotFound("Session not found")

    # Collect detections (via relationship or query)
    dets = getattr(sess, "detections", None)
    if dets is None:
        dets = Detection.query.filter_by(session_id=sid).all()

    # Delete children files & rows
    for d in dets:
        # child tables first (if any)
        try:
            Reasoning.query.filter_by(detection_id=d.id).delete(synchronize_session=False)
        except Exception:
            pass
        try:
            Report.query.filter_by(detection_id=d.id).delete(synchronize_session=False)
        except Exception:
            pass
        # files
        _rm(getattr(d, "image_path", None))
        _rm(getattr(d, "annotated_path", None))
        # row
        db.session.delete(d)

    # finally the session
    db.session.delete(sess)
    db.session.commit()
    return jsonify({"ok": True, "deleted_session": sid, "deleted_detections": len(dets)}), 200

@app.delete("/api/detections/<int:det_id>")
@app.post("/api/detections/<int:det_id>/delete")  # HTML form fallback
def api_delete_detection(det_id):
    """Delete a single detection and its artifacts."""
    d = Detection.query.get(det_id)
    if not d:
        raise NotFound("Detection not found")

    # child tables first (if any)
    try:
        Reasoning.query.filter_by(detection_id=d.id).delete(synchronize_session=False)
    except Exception:
        pass
    try:
        Report.query.filter_by(detection_id=d.id).delete(synchronize_session=False)
    except Exception:
        pass

    # files
    _rm(getattr(d, "image_path", None))
    _rm(getattr(d, "annotated_path", None))

    # row
    db.session.delete(d)
    db.session.commit()
    return jsonify({"ok": True, "deleted_detection": det_id}), 200

@app.get("/api/sessions/<int:sid>/report_overall")
def api_report_overall(sid):
    try:
        sess = Session.query.get_or_404(sid)
        detections = Detection.query.filter_by(session_id=sess.id).order_by(Detection.id.asc()).all()
        pdf_bytes, fname = _build_detections_report_pdf(sess, detections, title="Session Report", group_by="week")
        return send_file(io.BytesIO(pdf_bytes), download_name=fname, as_attachment=True, mimetype="application/pdf")
    except Exception as e:
        print("PDF generation failed:", e)
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/sessions/<int:sid>/week/<int:week>/report")
def api_report_week(sid, week):
    sess = Session.query.get_or_404(sid)
    q = Detection.query.filter_by(session_id=sess.id)
    if hasattr(Detection, "week"):
        q = q.filter_by(week=week)
    detections = q.order_by(Detection.ts.asc()).all()
    # group by day in a week report
    pdf_bytes, fname = _build_detections_report_pdf(sess, detections, title=f"Week {week} Report", group_by="day")
    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name=fname)


@app.post("/api/sessions")
def create_session():
    data = request.get_json(force=True)
    start_week = int(data.get("start_week", 1))
    s = Session(
        name=data.get("name", "Session"),
        start_week=start_week,
        current_week=start_week,
        current_phase=phase_from_week(start_week),  # or 'phase' if that’s your column
        started_at=datetime.utcnow(),
        details={"plan_weeks": [start_week], "weeks_planned": [start_week]}
    )
    db.session.add(s)
    db.session.commit()
    return jsonify({
        "id": s.id,
        "name": s.name,
        "start_week": s.start_week,
        "current_week": s.current_week,
        "phase": getattr(s, "current_phase", None) or getattr(s, "phase", None),
        "started_iso": s.started_at.isoformat()
    })


@app.get("/api/sessions")
def list_sessions():
    rows = Session.query.order_by(Session.id.desc()).all()
    out = []
    for s in rows:
        # some dbs/models used s.phase, some had s.current_phase – normalise here
        phase = getattr(s, "phase", None) or getattr(s, "current_phase", None)

        started_iso = None
        started_display = ""
        if getattr(s, "started_at", None):
            dt = s.started_at  # datetime
            started_iso = dt.isoformat()
            # nice, human readable (no tz? drop %Z)
            started_display = dt.strftime("%a, %d %b %Y %H:%M:%S")

        out.append({
            "id": s.id,
            "name": s.name,
            "start_week": s.start_week,
            "current_week": s.current_week,
            "phase": phase,                          # <— IMPORTANT
            "started_iso": started_iso,              # ISO for machines
            "started_display": started_display,      # string for UI
        })
    return jsonify(out), 200




@app.patch("/api/sessions/<int:sid>")
def update_session(sid):
    s = Session.query.get_or_404(sid)
    data = request.get_json(force=True)
    if "current_week" in data:
        s.current_week = int(data["current_week"])
        s.current_phase = phase_from_week(s.current_week)
    if "ended" in data and data["ended"] is True:
        s.ended_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"ok": True, "current_week": s.current_week, "current_phase": s.current_phase})

@app.post("/api/sessions/<int:sid>/weeks/<int:week>/bulk")
def api_bulk_week(sid, week):
    """Bulk upload images for a week. Optional form field 'day' (1=Mon..7=Sun)."""
    sess = Session.query.get_or_404(sid)
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no images[] provided"}), 400

    # optional weekday selector
    day = request.form.get("day", type=int)
    ts_override = None
    if day and 1 <= day <= 7:
        ts_override = _week_start_dt(sess, week) + timedelta(days=day - 1)

    phase_ = phase_from_week(week)  # your existing helper

    ids = []
    for f in files:
        job_id = uuid.uuid4().hex
        JOB_RES[job_id] = {"status": "queued"}
        JOB_Q.put({
            "id": job_id,
            "image_bytes": f.read(),
            "phase": phase_,
            "week": int(week),
            "session_id": int(sess.id),
            "ts_override": ts_override.isoformat() if ts_override else None
        })
        ids.append(job_id)

    return jsonify({"job_ids": ids, "status": "queued"}), 200


@app.post("/api/jobs")
def create_job():
    # prefer week
    week = request.form.get("week", type=int)
    session_id = request.form.get("session_id", type=int)
    phase = request.form.get("phase")  # fallback for old clients
    f = request.files.get("image")
    if not f: return jsonify({"error":"image required"}), 400

    if week is None:
        # derive from session current_week if session provided
        if session_id:
            s = Session.query.get(session_id)
            if not s: return jsonify({"error":"invalid session_id"}), 400
            week = s.current_week
        else:
            return jsonify({"error":"week (1..20) or session_id required"}), 400

    try:
        phase_ = phase_from_week(week)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    img_bytes = f.read()
    job_id = uuid.uuid4().hex
    JOB_RES[job_id] = {"status":"queued"}
    JOB_Q.put({"id":job_id, "image_bytes": img_bytes, "phase": phase_, "week": week, "session_id": session_id})
    return jsonify({"job_id": job_id, "status":"queued"})

@app.post("/api/sessions/<int:sid>/weeks/<int:week>/bulk_jobs")
def bulk_jobs(sid, week):
    s = Session.query.get_or_404(sid)
    try: phase_ = phase_from_week(week)
    except Exception as e: return jsonify({"error": str(e)}), 400
    files = request.files.getlist("images")
    if not files: return jsonify({"error":"no images[] provided"}), 400
    ids = []
    for f in files:
        job_id = uuid.uuid4().hex
        JOB_RES[job_id] = {"status":"queued"}
        JOB_Q.put({"id":job_id, "image_bytes": f.read(), "phase": phase_, "week": week, "session_id": s.id})
        ids.append(job_id)
    return jsonify({"job_ids": ids, "status":"queued"})


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    st = JOB_RES.get(job_id) or {"status":"unknown"}
    return jsonify(st)

@app.get("/api/detections/<int:det_id>/hist")
def api_hist_detection(det_id):
    d = Detection.query.get_or_404(det_id)
    counts = _hist_from_rows([d])
    return jsonify({"ok": True, "labels": LABELS_HIST, "counts": counts}), 200

@app.get("/api/sessions/<int:sid>/hist")
def api_hist_session(sid):
    rows = Detection.query.filter_by(session_id=sid).all()
    counts = _hist_from_rows(rows)  # sums details['counts'] when present, else derives from items
    return jsonify({"labels": LABELS_HIST, "counts": counts}), 200

@app.get("/api/sessions/<int:sid>/detections")
def detections_for_session(sid):
    rows = Detection.query.filter_by(session_id=sid)\
                          .order_by(Detection.id.desc()).all()
    return jsonify([{
        "id": d.id,
        "ts": d.created_at.isoformat(),
        "week": d.week,
        "phase": d.phase,
        "verdict": d.verdict,
        "image_path": _norm(d.image_path),
        "annotated_path": _norm(d.annotated_path),
        "details": d.details,
        "session_id": d.session_id,
        "is_manual": d.is_manual,
    } for d in rows]), 200


@app.get("/api/sessions/<int:sid>/weeks/<int:week>/hist")
def api_hist_session_week(sid, week):
    rows = Detection.query.filter_by(session_id=sid, week=week).all()
    counts = _hist_from_rows(rows)
    return jsonify({"ok": True, "labels": LABELS_HIST, "counts": counts, "n": len(rows)}), 200


@app.get("/api/detections")
def list_detections():
    sid = request.args.get("session_id", type=int)
    q = Detection.query
    if sid: q = q.filter_by(session_id=sid)
    rows = q.order_by(Detection.id.desc()).limit(200).all()
    return jsonify([{
        "id": d.id, "ts": d.created_at.isoformat(), "week": d.week,
        "phase": d.phase, "verdict": d.verdict,
        "image_path": _norm(d.image_path),
        "annotated_path": _norm(d.annotated_path),
        "details": d.details, "session_id": d.session_id, "is_manual": d.is_manual
    } for d in rows])

@app.get("/api/sessions/<int:sid>")
def get_session(sid):
    sess = Session.query.get_or_404(sid)
    payload = sess.to_dict() if hasattr(sess, "to_dict") else {
        "id": sess.id,
        "name": sess.name,
        "started_at": sess.started_at,
        "start_week": sess.start_week,
        "current_week": sess.current_week,
        "phase": sess.current_phase or sess.phase,
    }
    payload["started_date"] = _iso_date(sess.started_at)       # NEW
    payload["now_date"] = datetime.now().date().isoformat()    # NEW
    return jsonify(payload)


@app.get("/api/weather")
def api_weather():
    # Env + query
    api_key = os.getenv("WEATHER_API_KEY")
    units   = os.getenv("WEATHER_UNITS", "metric")


    qlat = request.args.get("lat", type=float)
    qlon = request.args.get("lon", type=float)
    env_lat = os.getenv("WEATHER_LAT", "0")
    env_lon = os.getenv("WEATHER_LON", "0")
    try:
        env_lat = float(env_lat); env_lon = float(env_lon)
    except Exception:
        env_lat = 0.0; env_lon = 0.0

    lat = qlat if qlat is not None else env_lat
    lon = qlon if qlon is not None else env_lon

    app.logger.info(f"[WX] .env path={ENV_PATH} exists={ENV_PATH.exists()}")
    app.logger.info(f"[WX] key?={'yes' if api_key else 'NO'} units={units} lat={lat} (q={qlat}/env={env_lat}) lon={lon} (q={qlon}/env={env_lon})")

    if lat == 0 and lon == 0:
        return jsonify({"ok": False, "message": "Set WEATHER_LAT/LON in backend/.env or pass lat/lon in query"}), 200

    now = time.time()
    if request.args.get("nocache"):
        _WEATHER_CACHE.update(ts=0, data=None)
    # cache normalized payload ...
    if _WEATHER_CACHE["data"] and (now - _WEATHER_CACHE["ts"] < _WEATHER_TTL):
        app.logger.info(f"[WX] Cache HIT age={int(now - _WEATHER_CACHE['ts'])}s")
        return jsonify(_WEATHER_CACHE["data"]), 200

    # Try OpenWeather FREE endpoints first (needs API key); else Open-Meteo
    try:
        if api_key:
            current, hours = _owm_fetch(lat, lon, units, api_key)
            source = "openweather_free"
        else:
            raise RuntimeError("No OWM key; using Open-Meteo")
    except Exception as e:
        app.logger.warning(f"[WX] OWM free failed ({e}); falling back to Open-Meteo")
        current, hours = _openmeteo_fetch(lat, lon, units)
        source = "openmeteo"

    payload = {
        "ok": True,
        "source": source,
        "current": current,
        "hours": hours,
        "advice": _ginger_advisory(current)
    }
    _WEATHER_CACHE.update(ts=now, data=payload)
    app.logger.info(f"[WX] OK source={source} temp={current.get('temp')} hum={current.get('humidity')} hours={len(hours)}")
    app.logger.info(f"[WX] OK source={source} city={current.get('location_name')} "
                f"icon={current.get('icon')} icon_url={'yes' if current.get('icon_url') else 'no'}")

    return jsonify(payload), 200


@app.get("/api/sessions/<int:sid>/weeks/<int:week>/summary")
def week_summary(sid, week):
    sess = Session.query.get_or_404(sid)
    # rows for that week
    rows = Detection.query.filter_by(session_id=sid, week=week).order_by(Detection.id.desc()).all()

    # histogram (per-plant; uses your existing _hist_from_rows)
    labels = ["Normal plant", "Abnormal plant", "Empty Bag Detected"]
    counts = _hist_from_rows(rows)

    # week start date and days
    wstart = _week_start_date(sess, week)
    wstart_dt = datetime.fromisoformat(wstart)

    # group rows by exact calendar day
    def _row_date(r):
        v = getattr(r, "created_at", None)
        v2 = getattr(r, "ts", None)   # if you ever add an explicit ts column
        return _iso_date(v2 or v)

    day_items = { (wstart_dt + timedelta(days=i)).date().isoformat(): [] for i in range(7) }
    for r in rows:
        d = _row_date(r)
        if d in day_items:
            day_items[d].append(r.id)

    days = []
    for i in range(7):
        d = (wstart_dt + timedelta(days=i)).date().isoformat()
        days.append({
            "name": WEEKDAYS[i],
            "date": d,
            "count": len(day_items[d]),
            "ids": day_items[d],
        })

    return jsonify({
        "week_start_date": wstart,
        "hist": {"labels": labels, "counts": counts},
        "days": days
    })


@app.post("/api/sessions/<int:sid>/reasoning")
def add_reasoning(sid):
    s = Session.query.get_or_404(sid)
    text = request.get_json(force=True).get("text","")
    r = Reasoning(text=text, session_id=s.id)
    db.session.add(r); db.session.commit()
    return jsonify({"id": r.id, "created_at": r.created_at.isoformat()})

@app.post("/api/sessions/<int:sid>/report")
def make_pdf(sid):
    s = Session.query.get_or_404(sid)
    latest = Detection.query.filter_by(session_id=s.id).order_by(Detection.id.desc()).limit(5).all()
    r = Reasoning.query.filter_by(session_id=s.id).order_by(Reasoning.id.desc()).first()
    pdf_name = f"report_session_{sid}.pdf"
    pdf_path = os.path.join(ARTIFACTS_DIR, pdf_name)
    make_report(pdf_path, s, latest, r.text if r else None, artifacts_root=".")
    rp = Report(pdf_path=os.path.relpath(pdf_path, start="."), session_id=s.id)
    db.session.add(rp); db.session.commit()
    return jsonify({"report_id": rp.id, "pdf_path": rp.pdf_path})

@app.get("/api/reports/<int:rid>")
def download_report(rid):
    r = Report.query.get_or_404(rid)
    return send_file(r.pdf_path, as_attachment=True)

@app.get("/files/<path:fp>")
def files(fp):
    fp = fp.replace("\\", "/")
    allowed = ("uploads/", "results/", "artifacts/")
    if not any(fp.startswith(a) or fp == a.rstrip("/") for a in allowed):
        return jsonify({"error":"forbidden"}), 403

    p = Path(fp)
    if not p.exists() and fp.startswith("results/annot_"):
        alt = "uploads/orig_" + fp.split("results/annot_", 1)[1]
        if Path(alt).exists():
            app.logger.info(f"[FILES] {fp} → fallback to {alt}")
            fp = alt

    return send_from_directory(".", fp, as_attachment=False)
# Get one detection
@app.get("/api/detections/<int:det_id>")
def get_detection(det_id):
    d = Detection.query.get_or_404(det_id)
    return jsonify({
        "id": d.id,
        "ts": (d.created_at or d.ts).isoformat() if getattr(d, "created_at", None) else str(d.ts),
        "phase": d.phase,
        "verdict": d.verdict,
        "image_path": d.image_path,
        "annotated_path": d.annotated_path,
        "details": d.details,
        "session_id": d.session_id,
        # NEW — include weather snapshot
        "wx_summary":    d.wx_summary,
        "wx_temp_c":     d.wx_temp_c,
        "wx_humidity":   d.wx_humidity,
        "wx_rain_1h_mm": d.wx_rain_1h_mm,
        "wx_wind_kmh":   d.wx_wind_kmh,
    })

@app.post("/api/detections/<int:det_id>/gen_reason")
def api_generate_reason(det_id):
    d = Detection.query.get_or_404(det_id)

    # Counts — prefer what you store during inference; fall back to zeros.
    det_details = d.details or {}
    counts = (det_details.get("counts") or
              det_details.get("plant_counts") or
              {"normal": 0, "abnormal": 0, "empty": 0})
    normal   = int(counts.get("normal", 0))
    abnormal = int(counts.get("abnormal", 0))
    empty    = int(counts.get("empty_bag", counts.get("empty", 0)))

    # Weather snapshot
    wx_summary = d.wx_summary or "Unknown"
    wx_temp    = f"{d.wx_temp_c:.1f}°C" if d.wx_temp_c is not None else "—"
    wx_hum     = f"{d.wx_humidity}%"   if d.wx_humidity is not None else "—"
    wx_rain    = f"{d.wx_rain_1h_mm} mm" if d.wx_rain_1h_mm is not None else "0 mm"
    wx_wind    = f"{round(d.wx_wind_kmh)} km/h" if d.wx_wind_kmh is not None else "—"

    prompt = f"""
You are an agronomy assistant specialized in ginger. Analyze this SINGLE detection snapshot and write a concise reasoning and advice.

Phase: {d.phase}
Verdict: {d.verdict}

Counts:
- Normal plants: {normal}
- Abnormal plants: {abnormal}
- Empty bags detected: {empty}

Weather at capture:
- Summary: {wx_summary}
- Temp: {wx_temp}
- Humidity: {wx_hum}
- Rain (last 1h): {wx_rain}
- Wind: {wx_wind}

Write:
1) 2–3 paragraphs explaining what the numbers and weather imply for the plants this week.
2) Up to 3 short, actionable tips (bulleted).
Keep it under 150 words total; stay specific to ginger.
3) Tells the number of ginger plants detected at the begining, number of ginger plants = {abnormal} + {normal}. 
"""

    text = _generate_gemini_reasoning(prompt)

    # Persist into this detection so it survives navigation
    # Persist (REASSIGN a fresh dict so SQLAlchemy detects a change)
    det_details = dict(d.details or {})
    det_details["reasoning_text"] = text
    det_details["reasoning_ts"] = datetime.utcnow().isoformat()
    d.details = det_details
    db.session.add(d)
    db.session.commit()


    return jsonify({"ok": True, "text": text}), 200

# Save reasoning for a detection
@app.post("/api/detections/<int:det_id>/reasoning")
def add_detection_reasoning(det_id):
    d = Detection.query.get_or_404(det_id)
    text = request.get_json(force=True).get("text", "").strip()
    if not text:
        return jsonify({"error": "reasoning text required"}), 400

    # Persist in DB (attach to the same session for now)
    r = Reasoning(text=text, session_id=d.session_id if d.session_id else None)
    db.session.add(r); db.session.commit()

    # Also save a .txt alongside artifacts in a timestamped dir
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    det_dir = Path(ARTIFACTS_DIR) / f"det_{d.id}_{ts}"
    det_dir.mkdir(parents=True, exist_ok=True)
    txt_path = det_dir / "reasoning.txt"
    txt_path.write_text(text, encoding="utf-8")

    # Remember the path in detection.details for easy linking
    det_details = dict(d.details or {})
    det_details["reasoning_path"] = str(txt_path).replace("\\", "/")
    d.details = det_details
    db.session.commit()

    return jsonify({"ok": True, "reasoning_path": det_details["reasoning_path"], "reasoning_id": r.id})

# Generate a PDF report for one detection
@app.post("/api/detections/<int:det_id>/report")
def det_report(det_id):
    d = Detection.query.get_or_404(det_id)
    # Use latest reasoning in same session (if any)
    r = None
    if d.session_id:
        r = Reasoning.query.filter_by(session_id=d.session_id).order_by(Reasoning.id.desc()).first()

    # Put report alongside reasoning folder if present; else general artifacts
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    det_dir = Path(ARTIFACTS_DIR) / f"det_{d.id}_{ts}"
    det_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = det_dir / f"report_det_{d.id}.pdf"

    # Reuse report_gen to build a single-detection report
    make_report(str(pdf_path), session=Session.query.get(d.session_id) if d.session_id else type("S", (), {"id":0,"name":"N/A","current_phase":d.phase}),
                detections=[d], reasoning_text=(r.text if r else None), artifacts_root=".")

    # Also register a Report row (tie to session if exists)
    rp = Report(pdf_path=str(pdf_path).replace("\\","/"), session_id=d.session_id if d.session_id else None)
    db.session.add(rp); db.session.commit()

    return jsonify({"ok": True, "pdf_path": rp.pdf_path, "report_id": rp.id})

@app.get("/api/sessions/<int:sid>/weeks")
def api_weeks_get(sid):
    sess = Session.query.get_or_404(sid)
    planned = _get_plan_weeks(sess)

    # weeks that actually have detections
    present = (
        db.session.query(Detection.week)
        .filter(Detection.session_id == sid)
        .distinct()
        .all()
    )
    present = sorted({int(w[0]) for w in present if w and w[0] is not None})

    return jsonify({"planned": planned, "present": present}), 200


    
@app.get("/api/sessions/<int:sid>/weeks/<int:week>/detections")
def detections_for_week(sid, week):
    rows = Detection.query.filter_by(session_id=sid, week=week)\
                          .order_by(Detection.id.desc()).all()
    return jsonify([{
        "id":d.id, "ts": d.created_at.isoformat(), "week": d.week, "phase": d.phase,
        "verdict": d.verdict, "annotated_path": d.annotated_path, "image_path": d.image_path
    } for d in rows])
    
def api_session_weeks(sid):
    """Return planned + present weeks for a session."""
    sess = Session.query.get_or_404(sid)

    # present = weeks that actually have detections
    q = db.session.query(Detection.week).filter(Detection.session_id == sid).distinct()
    present = sorted(set(int(w or 0) for (w,) in q.all() if w is not None))

    planned = _get_plan_weeks(sess)
    # bootstrap default if nothing stored yet
    if not planned:
        planned = [int(getattr(sess, "start_week", 1) or 1)]

    return jsonify({"planned": planned, "present": present}), 200


@app.post("/api/sessions/<int:sid>/weeks")
def api_weeks_add_idempotent(sid):
    """Idempotent: adds 'week' into planned if missing; returns {'ok', 'planned'}"""
    sess = Session.query.get_or_404(sid)
    data = request.get_json(silent=True) or {}
    week = int(data.get("week", 0))
    app.logger.info(f"[API] ADD planned week sid={sid} payload={data} parsed_week={week}")
    if week < 1 or week > 52:
        app.logger.warning(f"[API] ADD planned week bad_week sid={sid} week={week}")
        return jsonify({"ok": False, "error": "bad_week"}), 400

    weeks = _get_plan_weeks(sess)
    if week not in weeks:
        before = weeks[:]
        weeks.append(week)
        _set_plan_weeks(sess, weeks)
        db.session.commit()
        app.logger.info(f"[API] ADD planned week sid={sid} before={before} after={weeks}")
    else:
        app.logger.info(f"[API] ADD planned week sid={sid} week={week} already_present")

    return jsonify({"ok": True, "planned": _get_plan_weeks(sess)}), 200

@app.delete("/api/sessions/<int:sid>/weeks/<int:week>")
def api_weeks_plan_delete(sid, week):
    sess = Session.query.get_or_404(sid)
    planned = _get_plan_weeks(sess)
    before = set(planned)
    after = sorted(w for w in planned if int(w) != int(week))
    if set(after) != before:
        _set_plan_weeks(sess, after)
        db.session.commit()
    return jsonify({"ok": True, "planned": after}), 200

@app.post("/api/sessions/<int:sid>/weeks/plan")
def api_weeks_plan_add(sid):
    app.logger.info("[API IN] POST /api/sessions/%s/weeks/plan args=%r json=%r",
                    sid, request.args, request.get_json(silent=True))
    sess = Session.query.get_or_404(sid)

    # 1) Read payload from JSON (or form as fallback)
    payload = request.get_json(silent=True) if request.is_json else request.form
    if not payload:
        payload = {}

    # 2) Extract & validate "week"
    week_raw = payload.get("week")
    try:
        week = int(week_raw)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_week", "value": week_raw}), 400
    if week < 1 or week > 52:
        return jsonify({"ok": False, "error": "out_of_range"}), 400

    # 3) Overwrite switch: either query ?overwrite=1 or JSON { "force": true }
    overwrite = (request.args.get("overwrite") == "1") or \
                str(payload.get("force", "")).lower() in ("1", "true", "yes")

    # 4) Load/save planned weeks
    planned = _get_plan_weeks(sess)
    if week in planned and not overwrite:
        return jsonify({"ok": False, "error": "exists"}), 409

    planned = [w for w in planned if int(w) != week] + [week]
    _set_plan_weeks(sess, planned)
    _ensure_week_meta(sess, week)
    db.session.commit()
    return jsonify({
        "ok": True,
        "planned": _get_plan_weeks(sess),
        "week_meta": (getattr(sess, "details", {}) or {}).get("week_meta", {})
    }), 200


@app.route("/api/sessions/<int:sid>/weeks/<int:week>/report", methods=["GET", "POST"])
def api_week_report_plural(sid, week):
    return _build_and_send_week_pdf(sid, week)

@app.post("/api/sessions/<int:sid>/report_overall")
def session_overall_report(sid):
    s = Session.query.get_or_404(sid)
    dets = Detection.query.filter_by(session_id=sid)\
                          .order_by(Detection.id.desc()).all()
    ts_dir = Path(ARTIFACTS_DIR) / f"session_{sid}_overall"
    ts_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = ts_dir / f"report_session_{sid}_overall.pdf"
    # pick latest reasoning across weeks (optional)
    r = Reasoning.query.filter_by(session_id=sid).order_by(Reasoning.id.desc()).first()
    make_report(str(pdf_path), s, dets, (r.text if r else None), artifacts_root=".")
    rp = Report(pdf_path=str(pdf_path).replace("\\","/"), session_id=s.id, week=None)
    db.session.add(rp); db.session.commit()
    return jsonify({"report_id": rp.id, "pdf_path": rp.pdf_path})

@app.post("/api/sessions/<int:sid>/weeks/<int:week>/reasoning_gemini")
def weekly_gemini_reasoning(sid, week):
    s = Session.query.get_or_404(sid)
    dets = Detection.query.filter_by(session_id=sid, week=week).all()

    # Build a compact prompt
    lines = [f"Weekly ginger field analysis — session {s.id}, week {week}."]
    verdicts = {"Normal":0, "Abnormal":0, "NoDetections":0}
    for d in dets:
        verdicts[d.verdict] = verdicts.get(d.verdict,0)+1
    lines.append(f"Counts: {verdicts}.")
    for d in dets[:30]:  # cap prompt size
        lines.append(f"- {d.created_at.isoformat()} {d.verdict} {d.phase} {d.week}")
    prompt = "\n".join(lines) + "\nProvide concise agronomic insights and next actions."

    text = gemini_reasoning(prompt)

    # store in DB with week
    r = Reasoning(text=text, session_id=s.id, week=week)
    db.session.add(r); db.session.commit()

    # write file alongside weekly artifacts
    ts_dir = Path(ARTIFACTS_DIR) / f"session_{sid}_week_{week}"
    ts_dir.mkdir(parents=True, exist_ok=True)
    txt_path = ts_dir / "reasoning_gemini.txt"
    txt_path.write_text(text, encoding="utf-8")

    return jsonify({"ok": True, "reasoning": text, "path": str(txt_path).replace("\\","/")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
