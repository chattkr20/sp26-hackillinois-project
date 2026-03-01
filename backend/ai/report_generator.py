"""
Modal.com GPU endpoint for CAT inspection report generation.
Uses transformers + torch directly (no vLLM) to avoid tokenizer conflicts.
Model: Phi-3.5-mini-instruct on A10G GPU.
"""

import base64
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
)

PROMPT_TEMPLATE = """You are a CAT equipment inspection AI. Extract structured information from the transcript and anomaly report below.

Transcript: "{{TRANSCRIPT}}"

Anomaly Report:
{{ANOMALY_JSON}}

Return ONLY a JSON object with exactly two keys:
1. "summary" (string): 2-3 sentence plain English inspection summary.
2. "parts" (array): [{"part_name": str, "part_details": str, "status": "PASS"|"MONITOR"|"FAIL"}]
If no parts are mentioned, infer from context. No markdown fences, no extra text."""


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

        anomaly = payload.get("anomaly", {})
        stt = payload.get("stt", {})
        part_name = payload.get("part_name") or "Not specified"
        operator_name = payload.get("operator_name") or "Unknown"
        operator_id = payload.get("operator_id") or "—"
        transcript = stt.get("transcript", "")
        anomaly_score = anomaly.get("anomaly_score", 0.0)
        anomaly_status = anomaly.get("status", "unknown")
        machine_type = anomaly.get("machine_type") or "Unknown"
        anomaly_subtype = anomaly.get("anomaly_subtype") or "None detected"

        # ── 1. Build prompt ──────────────────────────────────────────────────
        prompt = (
            PROMPT_TEMPLATE
            .replace("{{TRANSCRIPT}}", transcript)
            .replace("{{ANOMALY_JSON}}", json.dumps(anomaly, indent=2))
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

        try:
            llm_data = json.loads(raw)
        except Exception:
            llm_data = {"summary": raw, "parts": []}

        summary = llm_data.get("summary", "")
        parts = llm_data.get("parts", [])

        if not parts and part_name != "Not specified":
            status = "FAIL" if anomaly_status == "anomaly" else "PASS"
            parts = [{"part_name": part_name, "part_details": transcript[:120] or "See transcript", "status": status}]

        # ── 3. Generate PDF ──────────────────────────────────────────────────
        today = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        class ReportPDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 16)
                self.cell(0, 10, "CAT Equipment Inspection Report", new_x="LMARGIN", new_y="NEXT", align="C")
                self.set_font("Helvetica", "", 9)
                self.set_text_color(140, 140, 140)
                self.cell(0, 6, f"Generated: {today}", new_x="LMARGIN", new_y="NEXT", align="C")
                self.set_text_color(0, 0, 0)
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(140, 140, 140)
                self.cell(0, 10, f"Page {self.page_no()} | CAT Inspection Tool", align="C")

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
