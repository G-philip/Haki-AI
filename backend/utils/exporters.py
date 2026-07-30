import os
from datetime import datetime
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_UNDERLINE


class DocumentExporter:
    """Export legal documents to DOCX and PDF"""
    
    @staticmethod
    def export_docx(data, doc_type, deponent_name):
        os.makedirs('exports', exist_ok=True)
        safe_name = deponent_name.replace(' ', '_')[:30]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join('exports', f"{doc_type.replace(' ', '_')}_{safe_name}_{timestamp}.docx")
        
        doc = Document()
        
        section = doc.sections[0]
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.5)
        section.right_margin = Inches(1)
        
        style = doc.styles['Normal']
        style.font.name = 'Times New Roman'
        style.font.size = Pt(12)
        style.paragraph_format.line_spacing = 1.5
        
        text_content = data.get('document_text', 'Document content')
        
        center_keywords = [
            'REPUBLIC OF KENYA', 'AFFIDAVIT', 'IN THE MATTER OF',
            'CAP. 15', 'AND', 'CASE NO.', 'APPLICATION NO.',
        ]
        
        for line in text_content.split('\n'):
            stripped = line.strip()
            
            p = doc.add_paragraph()
            
            if not stripped:
                run = p.add_run(' ')
                run.font.name = 'Times New Roman'
                run.font.size = Pt(6)
                continue
            
            is_numbered = False
            if len(stripped) > 5 and stripped[0].isdigit():
                if '. THAT ' in stripped[:10]:
                    is_numbered = True
            
            is_centered = any(kw in stripped for kw in center_keywords)
            
            if is_numbered:
                # Bigger indent: number at 1.0", text wraps at 1.5"
                p.paragraph_format.left_indent = Inches(1.0)
                p.paragraph_format.first_line_indent = Inches(-0.5)
            else:
                p.paragraph_format.left_indent = Inches(0)
                p.paragraph_format.first_line_indent = Inches(0)
            
            run = p.add_run(stripped)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            
            if is_centered:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run.bold = True
                run.underline = WD_UNDERLINE.SINGLE
        
        doc.save(filename)
        return filename
    
    @staticmethod
    def export_pdf(data, doc_type, deponent_name):
        os.makedirs('exports', exist_ok=True)
        safe_name = deponent_name.replace(' ', '_')[:30]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join('exports', f"{doc_type.replace(' ', '_')}_{safe_name}_{timestamp}.pdf")
        
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        
        doc = SimpleDocTemplate(
            filename, pagesize=A4, rightMargin=72, leftMargin=108,
            topMargin=72, bottomMargin=72
        )
        
        center_style = ParagraphStyle(
            'Center', fontName='Times-Bold', fontSize=12,
            alignment=TA_CENTER, leading=18,
        )
        
        normal_style = ParagraphStyle(
            'Normal', fontName='Times-Roman', fontSize=12,
            alignment=TA_JUSTIFY, leading=18,
        )
        
        numbered_style = ParagraphStyle(
            'Numbered', fontName='Times-Roman', fontSize=12,
            alignment=TA_JUSTIFY, leading=18,
            leftIndent=108, firstLineIndent=-36,
        )
        
        story = []
        text_content = data.get('document_text', 'Document content')
        
        center_keywords = [
            'REPUBLIC OF KENYA', 'AFFIDAVIT', 'IN THE MATTER OF',
            'CAP. 15', 'AND', 'CASE NO.', 'APPLICATION NO.',
        ]
        
        for line in text_content.split('\n'):
            stripped = line.strip()
            
            if not stripped:
                story.append(Spacer(1, 6))
                continue
            
            is_numbered = False
            if len(stripped) > 5 and stripped[0].isdigit():
                if '. THAT ' in stripped[:10]:
                    is_numbered = True
            
            is_centered = any(kw in stripped for kw in center_keywords)
            
            if is_centered:
                story.append(Paragraph(f"<b><u>{stripped}</u></b>", center_style))
            elif is_numbered:
                story.append(Paragraph(stripped, numbered_style))
            else:
                story.append(Paragraph(stripped, normal_style))
        
        doc.build(story)
        return filename