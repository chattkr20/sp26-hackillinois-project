"""
Modal.com GPU endpoint for CAT inspection report generation.
Uses transformers + torch directly (no vLLM) to avoid tokenizer conflicts.
Model: Phi-3.5-mini-instruct on A10G GPU.
"""

import base64
import io
import json
from datetime import datetime

import modal

app = modal.App("cat-report-generator")

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"


def _download_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    AutoTokenizer.from_pretrained(MODEL_ID)
    AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.48.3",
        "accelerate",
        "fastapi[standard]",
        "fpdf2",
        "huggingface_hub",
    )
    .run_function(_download_model)
    .add_local_file("backend/prompt.txt", "/root/prompt.txt")
)


def _build_prompt(transcript: str, audio_anomaly: dict, visual_anomaly: dict) -> str:
    """Load prompt.txt and fill in the two placeholders with combined anomaly data."""
    with open("/root/prompt.txt", "r") as f:
        template = f.read()

    visual_str = str(visual_anomaly) if visual_anomaly else "Not available"
    audio_str = json.dumps(audio_anomaly, indent=2) if audio_anomaly else "Not available"
    combined_anomaly = f"Visual Anomaly:\n\n{visual_str}\n\nAudio Anomaly:\n\n{audio_str}"

    return (
        template
        .replace("{{INSERT_TRANSCRIPT_HERE}}", transcript)
        .replace("{{INSERT_ANOMALOUS_REPORT_HERE}}", combined_anomaly)
    )


@app.cls(
    image=image,
    gpu="A10G",
    timeout=300,
    scaledown_window=300,
)
class ReportGenerator:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    @modal.fastapi_endpoint(method="POST")
    def generate_report(self, payload: dict) -> dict:
        from fpdf import FPDF

        def sanitize(text: str) -> str:
            """Strip characters not supported by fpdf2 Helvetica (latin-1)."""
            return (
                str(text)
                .replace("\u2014", "-")   # em dash
                .replace("\u2013", "-")   # en dash
                .replace("\u2018", "'")   # left single quote
                .replace("\u2019", "'")   # right single quote
                .replace("\u201c", '"')   # left double quote
                .replace("\u201d", '"')   # right double quote
                .encode("latin-1", errors="replace").decode("latin-1")
            )

        anomaly = payload.get("anomaly", {})
        stt = payload.get("stt", {})
        image_anomaly = payload.get("image_anomaly", {})
        part_name = sanitize(payload.get("part_name") or "Not specified")
        operator_name = sanitize(payload.get("operator_name") or "Unknown")
        operator_id = sanitize(payload.get("operator_id") or "N/A")
        transcript = sanitize(stt.get("transcript", ""))
        anomaly_score = anomaly.get("anomaly_score", 0.0)
        anomaly_status = anomaly.get("status", "unknown")
        machine_type = sanitize(anomaly.get("machine_type") or "Unknown")
        anomaly_subtype = sanitize(anomaly.get("anomaly_subtype") or "None detected")
        image_data_b64 = payload.get("image_data_b64", "")
        img_status = image_anomaly.get("status", "")
        img_score = image_anomaly.get("anomaly_score", None)
        img_confidence = image_anomaly.get("confidence", None)
        img_mode = sanitize(image_anomaly.get("mode") or "N/A")

        # ── 1. Build prompt from prompt.txt ──────────────────────────────────
        prompt = _build_prompt(
            transcript=transcript,
            audio_anomaly=anomaly,
            visual_anomaly=image_anomaly,
        )
        messages = [{"role": "user", "content": prompt}]

        # ── 2. Run inference ─────────────────────────────────────────────────
        result = self.pipe(
            messages,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=True,
        )
        raw = result[0]["generated_text"][-1]["content"].strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        # ── Derive overall status from anomaly scores ────────────────────────
        audio_score = anomaly.get("anomaly_score", 0.0)
        visual_score = image_anomaly.get("anomaly_score", 0.0) if image_anomaly else 0.0
        visual_status = image_anomaly.get("status", "") if image_anomaly else ""
        if audio_score >= 1.0 or visual_status == "anomaly":
            derived_status = "FAIL"
        elif audio_score >= 0.3 or visual_score >= 0.5:
            derived_status = "MONITOR"
        else:
            derived_status = "PASS"

        try:
            llm_data = json.loads(raw)
        except Exception:
            llm_data = []

        # New prompt returns a flat list [{part_name, part_details}]
        summary = ""
        if isinstance(llm_data, list):
            raw_parts = llm_data
        elif isinstance(llm_data, dict):
            # Backwards-compat if model still returns {summary, parts}
            summary = sanitize(llm_data.get("summary", ""))
            raw_parts = llm_data.get("parts", [])
        else:
            raw_parts = []

        parts = [
            {
                "part_name": sanitize(p.get("part_name", "")),
                "part_details": sanitize(p.get("part_details", "") or ""),
                "status": derived_status,
            }
            for p in raw_parts
        ]

        if not parts and part_name != "Not specified":
            parts = [{"part_name": part_name, "part_details": transcript[:120] or "See transcript", "status": derived_status}]

        # ── 3. Generate PDF ──────────────────────────────────────────────────
        today = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        class ReportPDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 16)
                self.cell(0, 10, "CATalyze Inspection Report", new_x="LMARGIN", new_y="NEXT", align="C")
                self.set_font("Helvetica", "", 9)
                self.set_text_color(140, 140, 140)
                self.cell(0, 6, f"Generated: {today}", new_x="LMARGIN", new_y="NEXT", align="C")
                self.set_text_color(0, 0, 0)
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(140, 140, 140)
                self.cell(0, 10, f"Page {self.page_no()} | CATalyze", align="C")

            def section_title(self, title):
                self.set_font("Helvetica", "B", 11)
                self.set_fill_color(230, 240, 255)
                self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
                self.ln(2)

            def kv_row(self, key, value):
                self.set_font("Helvetica", "B", 9)
                self.cell(55, 6, key + ":")
                self.set_font("Helvetica", "", 9)
                self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

            def part_row(self, name, status, details):
                self.set_font("Helvetica", "", 9)
                colors = {"FAIL": (220, 50, 50), "MONITOR": (230, 160, 0), "PASS": (30, 160, 60)}
                self.cell(70, 6, name)
                r, g, b = colors.get(status.upper(), (80, 80, 80))
                self.set_text_color(r, g, b)
                self.set_font("Helvetica", "B", 9)
                self.cell(22, 6, status.upper())
                self.set_text_color(0, 0, 0)
                self.set_font("Helvetica", "", 9)
                self.multi_cell(0, 6, details or "—")
                self.ln(1)

        pdf = ReportPDF()
        pdf.add_page()

        pdf.section_title("Inspection Details")
        pdf.kv_row("Operator", f"{operator_name} (ID: {operator_id})")
        pdf.kv_row("Part / Component", part_name)
        pdf.kv_row("Machine Type", machine_type)
        pdf.kv_row("Inspection Date", today)
        pdf.ln(6)

        pdf.section_title("Acoustic Anomaly Analysis")
        pdf.kv_row("Result", "ANOMALY DETECTED" if anomaly_status == "anomaly" else "NORMAL")
        pdf.kv_row("Anomaly Score", f"{anomaly_score * 100:.1f}%")
        pdf.kv_row("Fault Subtype", anomaly_subtype)
        pdf.ln(6)

        if image_anomaly:
            pdf.section_title("Visual (Image) Anomaly Analysis")
            img_result_text = "ANOMALY DETECTED" if img_status == "anomaly" else ("NORMAL" if img_status == "normal" else "N/A")
            pdf.kv_row("Result", img_result_text)
            if img_score is not None:
                pdf.kv_row("Anomaly Score", f"{img_score * 100:.1f}%")
            if img_confidence is not None:
                pdf.kv_row("Confidence", f"{img_confidence * 100:.1f}%")
            pdf.kv_row("Mode", img_mode)
            pdf.ln(6)

        if image_data_b64:
            import base64 as _b64
            try:
                img_bytes = _b64.b64decode(image_data_b64)
                # Detect format from magic bytes so fpdf2 knows the type
                if img_bytes[:2] == b'\xff\xd8':
                    ext = 'jpg'
                elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                    ext = 'png'
                elif img_bytes[:6] in (b'GIF87a', b'GIF89a'):
                    ext = 'gif'
                else:
                    ext = 'jpg'  # safe default
                img_buf = io.BytesIO(img_bytes)
                img_buf.name = f"photo.{ext}"  # fpdf2 uses .name to determine format
                pdf.section_title("Inspection Photo")
                pdf.image(img_buf, w=170)
                pdf.ln(4)
            except Exception as e:
                print(f"[report] Could not embed image: {e}")

        if summary:
            pdf.section_title("AI Inspection Summary")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 6, summary)
            pdf.ln(4)

        if parts:
            pdf.section_title("Part-by-Part Analysis")
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(245, 245, 245)
            pdf.cell(70, 6, "Component", fill=True)
            pdf.cell(22, 6, "Status", fill=True)
            pdf.cell(0, 6, "Details", fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            for part in parts:
                pdf.part_row(
                    str(part.get("part_name", "")),
                    str(part.get("status", "PASS")),
                    str(part.get("part_details", "")),
                )
            pdf.ln(4)

        if transcript:
            pdf.section_title("Operator Transcript (Verbatim)")
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 5, transcript)
            pdf.set_text_color(0, 0, 0)

        # ── 4. Return ────────────────────────────────────────────────────────
        pdf_bytes = pdf.output()
        return {
            "pdf_base64": base64.b64encode(bytes(pdf_bytes)).decode("utf-8"),
            "summary": summary,
            "parts": parts,
        }
