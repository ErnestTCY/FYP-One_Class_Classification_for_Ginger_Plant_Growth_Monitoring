# backend/database.py
import datetime as dt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.sqlite import JSON

db = SQLAlchemy()
from sqlalchemy import Index

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    started_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    ended_at   = db.Column(db.DateTime, nullable=True)
    # NEW: week orchestration
    start_week   = db.Column(db.Integer, default=1)     # user-chosen
    current_week = db.Column(db.Integer, default=1)     # can be advanced/edited
    current_phase = db.Column(db.String(50), default="EarlySprouting")
    details      = db.Column(db.JSON, default=dict)



    detections = db.relationship("Detection", backref="session", lazy=True, cascade="all, delete-orphan")
    reasonings = db.relationship("Reasoning", backref="session", lazy=True, cascade="all, delete-orphan")
    reports    = db.relationship("Report",    backref="session", lazy=True, cascade="all, delete-orphan")

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    phase   = db.Column(db.String(50), nullable=False)
    week    = db.Column(db.Integer, nullable=False)   # NEW: which week (1..20)
    verdict = db.Column(db.String(32), nullable=False)
    image_path     = db.Column(db.String(255), nullable=False)
    annotated_path = db.Column(db.String(255), nullable=False)
    details = db.Column(JSON, nullable=True)
    is_manual = db.Column(db.Boolean, default=False)
    session_id = db.Column(db.Integer, db.ForeignKey("session.id"), nullable=True)
        # --- inside class Detection(db.Model) ---
    wx_summary    = db.Column(db.String(120))
    wx_temp_c     = db.Column(db.Float)
    wx_humidity   = db.Column(db.Integer)
    wx_rain_1h_mm = db.Column(db.Float)
    wx_wind_kmh   = db.Column(db.Float)

# quick index to fetch by session/week
Index("idx_detection_session_week", Detection.session_id, Detection.week)

class Reasoning(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    text = db.Column(db.Text, nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey("session.id"), nullable=True)
    week = db.Column(db.Integer, nullable=True)   # NEW: weekly reasoning

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    pdf_path = db.Column(db.String(255), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey("session.id"), nullable=True)
    week = db.Column(db.Integer, nullable=True)  # NEW: weekly or overall
