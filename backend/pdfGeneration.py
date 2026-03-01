from fpdf import FPDF
from pypdf import PdfReader, PdfWriter

class InspectionReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Wheel Loader: Safety & Maintenance', new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(5)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)

    def item_row(self, name, status, comments=None):
        self.set_font('Helvetica', '', 10)
        # Render Name
        self.cell(140, 6, name)

        # Render Status with Color
        if status =='FAIL':
            self.set_text_color(255, 0, 0) # Red
        elif status == 'MONITOR':
             self.set_text_color(255, 191, 0) # Yellow
        elif status == 'PASS':
             self.set_text_color(0, 128, 0) # Green
        else:
             self.set_text_color(0, 0, 0) # Black

        self.cell(0, 6, status, new_x="LMARGIN", new_y="NEXT", align='R')
        self.set_text_color(0, 0, 0) # Reset color

        # Render Comments if they exist
        if comments:
            self.set_font('Helvetica', 'I', 9)
            self.multi_cell(0, 5, f"Details: {comments}")
        self.ln(2)

# --- Data Extracted from Original PDF ---
metadata = {
    "Inspection Number": "22892110", "Customer No": "2969507567",
    "Serial Number": "W8210127", "Customer Name": "BORAL RESOURCES P/L",
    "Make": "CATERPILLAR", "Work Order": "FW12076",
    "Model": "982", "Completed On": "28/06/2025 11:07:00 AM",
    "Equipment Family": "Medium Wheel Loader", "Inspector": "John Doe",
    "Asset ID": "FL-3062", "PDF Generated On": "28/06/2025"
}

sections = [
    ("PART-BY-PART ANALYSIS", [
        ("1.1 Tires and Rims", "NORMAL"),
        ("1.2 Bucket Cutting Edge, Tips, or Moldboard", "MONITOR", "Inspect tips for wear."),
        ("1.3 Bucket Tilt Cylinders and Hoses", "NORMAL"),
        ("1.4 Bucket, Lift Cylinders and Hoses", "PASS", "Functioning correctly."),
        ("1.5 Lift arm attachment to frame", "MONITOR", "Attachment points showing minor wear."),
        ("1.6 Underneath of Machine", "FAIL")
    ])
]

# --- Generate PDF ---
pdf = InspectionReportPDF()
pdf.add_page()

# Metadata Grid
pdf.set_font('Helvetica', '', 10)
keys = list(metadata.keys())
for i in range(0, len(keys), 2):
    k1 = keys[i]
    v1 = metadata[k1]
    pdf.cell(50, 6, f"{k1}:", border=0)
    pdf.cell(45, 6, v1, border=0)

    if i + 1 < len(keys):
        k2 = keys[i+1]
        v2 = metadata[k2]
        pdf.cell(50, 6, f"{k2}:", border=0)
        pdf.cell(45, 6, v2, border=0, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.ln()

pdf.ln(10)

# Sections
for title, items in sections:
    pdf.section_title(title)
    for item in items:
        name = item[0]
        status = item[1]
        comment = item[2] if len(item) > 2 else None
        pdf.item_row(name, status, comment)
    pdf.ln(5)

output_filename = "recreated_inspection_report.pdf"
pdf.output(output_filename)
print(f"PDF generated: {output_filename}")