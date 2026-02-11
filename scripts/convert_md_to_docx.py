"""
Markdown to Word Document Converter 📄

Purpose:
    Converts markdown manuscript to Word (.docx) format suitable for
    Google Docs upload. Preserves formatting, headings, and tables.

Usage:
    python convert_md_to_docx.py <input.md> <output.docx>
    python convert_md_to_docx.py GSM_Manuscript_v2.md GSM_Manuscript_v2.docx
"""

import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import re


def set_cell_border(cell, **kwargs):
    """Set cell borders in a table."""
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        if edge in kwargs:
            edge_element = OxmlElement(f'w:{edge}')
            edge_element.set(qn('w:val'), 'single')
            edge_element.set(qn('w:sz'), '12')
            edge_element.set(qn('w:space'), '0')
            edge_element.set(qn('w:color'), '000000')
            tcBorders.append(edge_element)
    
    tcPr.append(tcBorders)


def parse_markdown(md_file: Path) -> list[dict]:
    """Parse markdown file into structured content."""
    content = md_file.read_text(encoding='utf-8')
    
    blocks = []
    current_block = None
    in_table = False
    table_lines = []
    
    for line in content.split('\n'):
        # Handle tables
        if line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        else:
            if in_table:
                blocks.append({'type': 'table', 'data': table_lines})
                in_table = False
                table_lines = []
            
            # Handle headings
            if line.startswith('# '):
                blocks.append({'type': 'heading1', 'text': line[2:].strip()})
            elif line.startswith('## '):
                blocks.append({'type': 'heading2', 'text': line[3:].strip()})
            elif line.startswith('### '):
                blocks.append({'type': 'heading3', 'text': line[4:].strip()})
            elif line.startswith('#### '):
                blocks.append({'type': 'heading4', 'text': line[5:].strip()})
            
            # Handle unordered lists
            elif line.strip().startswith('- '):
                blocks.append({'type': 'list_item', 'text': line.strip()[2:]})
            
            # Handle horizontal rules
            elif line.strip() in ('---', '***', '___'):
                blocks.append({'type': 'page_break', 'text': ''})
            
            # Handle code blocks
            elif line.strip().startswith('```'):
                blocks.append({'type': 'code_block_start', 'text': ''})
            elif line.strip().endswith('```') and current_block != 'code':
                blocks.append({'type': 'code_block_end', 'text': ''})
            
            # Handle paragraphs
            elif line.strip():
                blocks.append({'type': 'paragraph', 'text': line.strip()})
            
            # Handle blank lines
            elif not line.strip() and blocks and blocks[-1]['type'] != 'blank':
                blocks.append({'type': 'blank', 'text': ''})
    
    if in_table:
        blocks.append({'type': 'table', 'data': table_lines})
    
    return blocks


def parse_table(lines: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """Parse markdown table into header and rows."""
    if len(lines) < 2:
        return [], []
    
    header = [cell.strip() for cell in lines[0].split('|')[1:-1]]
    rows = []
    
    for line in lines[2:]:  # Skip separator line
        if line.strip().startswith('|'):
            row = [cell.strip() for cell in line.split('|')[1:-1]]
            rows.append(row)
    
    return header, rows


def add_formatted_text(paragraph, text: str):
    """Add text with markdown-style formatting (bold, italic)."""
    # Handle **bold**
    bold_pattern = r'\*\*(.+?)\*\*'
    # Handle *italic*
    italic_pattern = r'\*(.+?)\*'
    # Handle code
    code_pattern = r'`(.+?)`'
    
    last_end = 0
    result = []
    
    # Process formatting in order
    for match in re.finditer(r'\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`', text):
        if match.start() > last_end:
            result.append({'type': 'text', 'content': text[last_end:match.start()]})
        
        if match.group(1):  # bold
            result.append({'type': 'bold', 'content': match.group(1)})
        elif match.group(2):  # italic
            result.append({'type': 'italic', 'content': match.group(2)})
        elif match.group(3):  # code
            result.append({'type': 'code', 'content': match.group(3)})
        
        last_end = match.end()
    
    if last_end < len(text):
        result.append({'type': 'text', 'content': text[last_end:]})
    
    for item in result:
        run = paragraph.add_run(item['content'])
        
        if item['type'] == 'bold':
            run.bold = True
        elif item['type'] == 'italic':
            run.italic = True
        elif item['type'] == 'code':
            run.font.name = 'Courier New'
            run.font.size = Pt(10)


def md_to_docx(input_file: Path, output_file: Path):
    """Convert markdown file to Word document."""
    print(f"📖 Reading markdown file: {input_file}")
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return False
    
    # Parse markdown
    blocks = parse_markdown(input_file)
    print(f"✅ Parsed {len(blocks)} content blocks")
    
    # Create Word document
    doc = Document()
    
    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    
    # Add content
    for block in blocks:
        try:
            if block['type'] == 'heading1':
                heading = doc.add_heading(block['text'], level=1)
                heading.runs[0].font.color.rgb = RGBColor(0, 51, 102)
            
            elif block['type'] == 'heading2':
                heading = doc.add_heading(block['text'], level=2)
                heading.runs[0].font.color.rgb = RGBColor(0, 102, 153)
            
            elif block['type'] == 'heading3':
                doc.add_heading(block['text'], level=3)
            
            elif block['type'] == 'heading4':
                doc.add_heading(block['text'], level=4)
            
            elif block['type'] == 'paragraph':
                p = doc.add_paragraph(block['text'])
                add_formatted_text(p, block['text'])
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            
            elif block['type'] == 'list_item':
                doc.add_paragraph(block['text'], style='List Bullet')
            
            elif block['type'] == 'table':
                header, rows = parse_table(block['data'])
                
                if header:
                    table = doc.add_table(rows=1, cols=len(header))
                    table.style = 'Light Grid Accent 1'
                    
                    # Add header
                    header_cells = table.rows[0].cells
                    for i, cell_text in enumerate(header):
                        header_cells[i].text = cell_text
                        # Style header
                        for paragraph in header_cells[i].paragraphs:
                            for run in paragraph.runs:
                                run.bold = True
                                run.font.color.rgb = RGBColor(255, 255, 255)
                            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        # Header background color
                        shading_elm = OxmlElement('w:shd')
                        shading_elm.set(qn('w:fill'), '003366')
                        header_cells[i]._element.get_or_add_tcPr().append(shading_elm)
                    
                    # Add rows
                    for row in rows:
                        row_cells = table.add_row().cells
                        for i, cell_text in enumerate(row):
                            row_cells[i].text = cell_text
                            row_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
            
            elif block['type'] == 'blank':
                doc.add_paragraph()
            
            elif block['type'] == 'page_break':
                doc.add_page_break()
        
        except Exception as e:
            print(f"⚠️  Error processing block: {e}")
            continue
    
    # Save document
    doc.save(str(output_file))
    print(f"\n✅ Document saved to: {output_file}")
    print(f"📊 Document contains {len(doc.paragraphs)} paragraphs and {len(doc.tables)} tables")
    
    return True


def main():
    """Main conversion function."""
    if len(sys.argv) < 3:
        print("Usage: python convert_md_to_docx.py <input.md> <output.docx>")
        print("\nExample:")
        print("  python convert_md_to_docx.py GSM_Manuscript_v2.md GSM_Manuscript_v2.docx")
        return
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    print("=" * 70)
    print("📄 Markdown to Word Document Converter")
    print("=" * 70)
    
    success = md_to_docx(input_file, output_file)
    
    if success:
        print("\n✨ Conversion complete!")
        print(f"📤 Ready to upload to Google Docs: {output_file}")
    else:
        print("\n❌ Conversion failed")


if __name__ == "__main__":
    main()
