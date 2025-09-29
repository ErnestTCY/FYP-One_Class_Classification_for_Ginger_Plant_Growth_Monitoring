import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def make_report(pdf_path, session, detections, reasoning_text=None, artifacts_root="."):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4
    y = H - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, f"Session Report: {session.name} (#{session.id})")
    y -= 25
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Phase: {session.current_phase}")
    y -= 18

    if reasoning_text:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "AI Reasoning Summary:")
        y -= 16
        c.setFont("Helvetica", 10)
        for line in reasoning_text.splitlines():
            c.drawString(50, y, line[:100])
            y -= 12

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Latest Detections:")
    y -= 16
    c.setFont("Helvetica", 10)

    for d in detections[:5]:
        c.drawString(50, y, f"[{d.verdict}] {d.phase}  file={os.path.basename(d.image_path)}")
        y -= 12
        img_path = os.path.join(artifacts_root, d.annotated_path)
        if os.path.exists(img_path) and y > 150:
            c.drawImage(ImageReader(img_path), 50, y-120, width=240, height=120, preserveAspectRatio=True, mask='auto')
            y -= 130
        if y < 120:
            c.showPage()
            y = H - 50
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
