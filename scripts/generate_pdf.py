import re
import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfgen import canvas

# A Canvas subclass to implement professional Page Numbering ("Page X of Y")
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Suppress page number on the cover page (Page 1)
        if self._pageNumber > 1:
            # Header
            self.drawString(inch, 10.5 * inch, "EmotionSense AI: The Complete Project Handbook")
            self.setStrokeColor(colors.HexColor("#DDDDDD"))
            self.setLineWidth(0.5)
            self.line(inch, 10.4 * inch, 7.5 * inch, 10.4 * inch)
            
            # Footer
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(7.5 * inch, 0.5 * inch, page_text)
            self.drawString(inch, 0.5 * inch, "Confidential - Study Guide")
            self.line(inch, 0.6 * inch, 7.5 * inch, 0.6 * inch)
            
        self.restoreState()

def clean_text_for_pdf(text: str) -> str:
    """Removes unsupported emoji characters and formats markdown tags into HTML-like tags for ReportLab Paragraphs."""
    # Remove emojis
    text = re.sub(r'[^\x00-\x7F\u03B1-\u03C9\u2200-\u22FF\u2190-\u21FF\u2260-\u2265]', '', text)
    
    # Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 1. First format inline code backticks `code` using a placeholder to avoid inner match conflicts
    # `code` -> <font face="Courier" color="#c7254e"><b>code</b></font>
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" color="#c7254e"><b>\1</b></font>', text)
    
    # 2. Format bold **bold** -> <b>bold</b>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    
    # 3. Format italic *italic* -> <i>italic</i> (using single asterisks, avoiding underscores to prevent filename clashes)
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
    
    return text

def parse_markdown_to_flowables(md_path: Path, styles) -> list:
    """Parses markdown text and builds ReportLab flowables dynamically."""
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    flowables = []
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []
    
    # Custom heading styles
    h1_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E3D59'),
        spaceAfter=15,
        keepWithNext=True
    )
    
    vol_style = ParagraphStyle(
        'VolumeHeader',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#17B890'),
        spaceBefore=20,
        spaceAfter=15,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'ChapterHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E3D59'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#435058'),
        spaceBefore=8,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'CodeCustom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#222222'),
        spaceAfter=10
    )
    
    quote_style = ParagraphStyle(
        'QuoteCustom',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#555555'),
        leftIndent=20,
        rightIndent=10,
        spaceAfter=8
    )
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 1. Handle Code Blocks
        if stripped.startswith("```"):
            if in_code_block:
                # End of code block, construct flowable
                code_text = "\n".join(code_lines)
                code_p = Paragraph(clean_text_for_pdf(code_text).replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style)
                
                # Render code in a nice wrapped background box
                code_table = Table([[code_p]], colWidths=[6.5 * inch])
                code_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E9ECEF')),
                    ('LEFTPADDING', (0,0), (-1,-1), 10),
                    ('RIGHTPADDING', (0,0), (-1,-1), 10),
                    ('TOPPADDING', (0,0), (-1,-1), 8),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ]))
                flowables.append(code_table)
                flowables.append(Spacer(1, 10))
                
                in_code_block = False
                code_lines = []
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            # Preserve spacing in code blocks
            code_lines.append(line.rstrip())
            i += 1
            continue
            
        # 2. Handle Tables
        if stripped.startswith("|") and not in_code_block:
            if not in_table:
                in_table = True
                table_rows = []
            
            # Parse table rows
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            
            # Check if this is a separator line (e.g. | :--- | :---: |)
            is_separator = all(re.match(r'^:?-+:?$', c) for c in cells)
            if not is_separator:
                table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            # Table ended, build reportlab Table flowable
            if table_rows:
                # Convert cell contents to Paragraph flowables for auto-wrapping
                formatted_rows = []
                col_widths = []
                num_cols = len(table_rows[0])
                
                # Calculate dynamic column widths (total 6.5 inches)
                # Assign proportional widths based on cell text length estimation
                lengths = [0] * num_cols
                for r in table_rows:
                    for idx, c in enumerate(r):
                        lengths[idx] = max(lengths[idx], len(c))
                
                total_len = sum(lengths)
                if total_len > 0:
                    col_widths = [(max(0.5, (l / total_len) * 6.5)) * inch for l in lengths]
                else:
                    col_widths = [float(6.5 / num_cols) * inch] * num_cols
                
                for r_idx, row in enumerate(table_rows):
                    formatted_row = []
                    for c_idx, cell in enumerate(row):
                        cell_clean = clean_text_for_pdf(cell)
                        # Bold headings for the first row
                        cell_style = ParagraphStyle(
                            f'TableCell_{r_idx}_{c_idx}',
                            parent=body_style,
                            fontName='Helvetica-Bold' if r_idx == 0 else 'Helvetica',
                            fontSize=9,
                            leading=11,
                            spaceAfter=0
                        )
                        formatted_row.append(Paragraph(cell_clean, cell_style))
                    formatted_rows.append(formatted_row)
                    
                t = Table(formatted_rows, colWidths=col_widths, repeatRows=1)
                t_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3D59')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E5E5')),
                ]
                
                # Make header text white by wrapping paragraph color
                for c_idx in range(num_cols):
                    t_style.append(('TEXTCOLOR', (c_idx, 0), (c_idx, 0), colors.white))
                    
                # Add alternating row background colors
                for r_idx in range(1, len(table_rows)):
                    bg = colors.HexColor('#F8F9FA') if r_idx % 2 == 1 else colors.white
                    t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), bg))
                    
                t.setStyle(TableStyle(t_style))
                flowables.append(t)
                flowables.append(Spacer(1, 10))
                
            in_table = False
            table_rows = []
            
        # 3. Handle Empty Lines
        if not stripped:
            i += 1
            continue
            
        # 4. Handle Headings
        if stripped.startswith("# "):
            title = clean_text_for_pdf(stripped[2:])
            # If it's a Volume, insert a PageBreak to keep sections beautifully partitioned
            if "VOLUME" in title or "Volume" in title:
                flowables.append(PageBreak())
                flowables.append(Spacer(1, 40))
                flowables.append(Paragraph(title, vol_style))
                flowables.append(Spacer(1, 15))
            else:
                # Cover Page title or master header
                flowables.append(Paragraph(title, h1_style))
                flowables.append(Spacer(1, 15))
            i += 1
            continue
        elif stripped.startswith("## "):
            flowables.append(Paragraph(clean_text_for_pdf(stripped[3:]), h2_style))
            flowables.append(Spacer(1, 8))
            i += 1
            continue
        elif stripped.startswith("### "):
            flowables.append(Paragraph(clean_text_for_pdf(stripped[4:]), h3_style))
            flowables.append(Spacer(1, 6))
            i += 1
            continue
            
        # 5. Handle Blockquotes
        if stripped.startswith(">"):
            # Accumulate full quote block
            quote_text = clean_text_for_pdf(stripped[1:].strip())
            flowables.append(Paragraph(quote_text, quote_style))
            flowables.append(Spacer(1, 8))
            i += 1
            continue
            
        # 6. Handle Bullet points
        if stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = clean_text_for_pdf(stripped[2:])
            flowables.append(Paragraph(f"&bull; {bullet_text}", bullet_style))
            i += 1
            continue
            
        # 7. Standard Paragraph Text
        p_text = clean_text_for_pdf(stripped)
        flowables.append(Paragraph(p_text, body_style))
        i += 1
        
    return flowables

def generate_pdf(handbook_md_path: Path, output_pdf_path: Path):
    # Setup document geometry with 0.75-inch margins
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Generate flowables
    print("[PDF Builder] Parsing markdown and mapping ReportLab flowables...")
    flowables = parse_markdown_to_flowables(handbook_md_path, styles)
    
    # Build document using our custom page numbering canvas
    print(f"[PDF Builder] Generating PDF output at: {output_pdf_path}...")
    doc.build(flowables, canvasmaker=NumberedCanvas)
    print("[PDF Builder] PDF Generation complete!")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    handbook_md = project_root / "emotionsense_ai_handbook.md"
    
    # Fallback to appdata brain folder if not in root
    if not handbook_md.exists():
        # Search parent directories
        appdata_dir = Path("C:/Users/ASHIK/.gemini/antigravity/brain/c4895291-b019-4851-a6e8-453362b4edf8")
        handbook_md = appdata_dir / "emotionsense_ai_handbook.md"
        
    if not handbook_md.exists():
        print(f"Error: Could not locate handbook markdown file.")
        sys.exit(1)
        
    output_pdf = handbook_md.with_suffix(".pdf")
    generate_pdf(handbook_md, output_pdf)
