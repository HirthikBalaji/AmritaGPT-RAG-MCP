"""
Document Intelligence & Ingestion Parser for AmritaGPT.
Parses heterogeneous document formats: PDF, DOCX, XLSX, PPTX, TXT, and Images.
Treats documents as structured knowledge objects rather than raw text.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re
import hashlib

from pypdf import PdfReader
try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None
from docx import Document as DocxDocument
import openpyxl
import pandas as pd
from pptx import Presentation

from amritagpt.config import DATA_DIR


@dataclass
class DocumentSection:
    section_id: str
    title: str
    content: str
    page: Optional[int] = None
    section_type: str = "text"  # text, heading, table, slide, metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    relative_path: str
    file_type: str
    department: str
    category: str
    academic_year: Optional[str]
    version: str
    access_policy: str  # public, student, faculty, admin
    sections: List[DocumentSection]
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """Format-agnostic document parser extracting structured hierarchical content."""

    @staticmethod
    def compute_checksum(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def infer_metadata(file_path: Path, relative_path: str) -> Dict[str, Any]:
        """Infer institutional department, category, year, and access level from path & name."""
        parts = Path(relative_path).parts
        dept = parts[0] if len(parts) > 0 else "General"
        sub_cat = parts[1] if len(parts) > 1 else dept
        
        filename = file_path.name
        
        # Detect Academic Year (e.g. 2019, 2022, 2023, 2024, 2025, 2026)
        year_match = re.search(r"\b(20[12]\d)\b", filename + " " + relative_path)
        academic_year = year_match.group(1) if year_match else None
        
        # Access Policy (All institutional knowledge defaults to student tier)
        access_policy = "student"

        # Detect Category
        category = "General"
        if "Academic" in dept or "curriculum" in lower_path or "syllabus" in lower_path or "regulations" in lower_path:
            category = "Academic Programs & Regulations"
        elif "Exam" in dept or "exam" in lower_path:
            category = "Examinations & Evaluation"
        elif "Faculty" in dept:
            category = "Faculty & Staff"
        elif "Research" in dept:
            category = "Research & Innovation"
        elif "Time Table" in dept or "timetable" in lower_path:
            category = "Class & Lab Schedules"
        elif "Hostel" in dept or "Canteen" in dept or "Student" in dept:
            category = "Student Life & Campus Facilities"
        elif "IQAC" in dept:
            category = "Quality Assurance & Accreditation"
        elif "Library" in dept:
            category = "Library & Resources"
        else:
            category = dept

        version = academic_year or "current"
        
        return {
            "department": dept,
            "category": category,
            "sub_category": sub_cat,
            "academic_year": academic_year,
            "version": version,
            "access_policy": access_policy
        }

    @classmethod
    def parse_pdf(cls, file_path: Path, doc_id: str) -> List[DocumentSection]:
        sections = []
        try:
            pages_data = []
            if pdfium is not None:
                pdf = pdfium.PdfDocument(str(file_path))
                try:
                    for idx, page in enumerate(pdf):
                        text = page.get_textpage().get_text_range() or ""
                        pages_data.append((idx + 1, text.strip()))
                finally:
                    pdf.close()
            else:
                reader = PdfReader(str(file_path))
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages_data.append((idx + 1, text.strip()))

            for page_num, text in pages_data:
                if not text:
                    continue
                
                # Detect structured headings (e.g. "R.1 Admissions", "Section 4", "Table of Contents")
                lines = text.split("\n")
                current_title = f"Page {page_num}"
                buffer = []

                for line in lines:
                    trimmed = line.strip()
                    if not trimmed:
                        continue
                    # Check if line looks like a regulation or section header
                    if re.match(r"^(R\.\d+|Section\s+\d+|Chapter\s+\d+|[0-9]+\.[0-9]+)\s+[A-Z]", trimmed) or (len(trimmed) < 60 and trimmed.isupper() and len(trimmed) > 4):
                        if buffer:
                            sec_content = "\n".join(buffer).strip()
                            if sec_content:
                                sections.append(DocumentSection(
                                    section_id=f"{doc_id}_p{page_num}_s{len(sections)+1}",
                                    title=current_title,
                                    content=sec_content,
                                    page=page_num,
                                    section_type="text"
                                ))
                            buffer = []
                        current_title = trimmed
                    else:
                        buffer.append(trimmed)

                if buffer:
                    sec_content = "\n".join(buffer).strip()
                    if sec_content:
                        sections.append(DocumentSection(
                            section_id=f"{doc_id}_p{page_num}_s{len(sections)+1}",
                            title=current_title,
                            content=sec_content,
                            page=page_num,
                            section_type="text"
                        ))
        except Exception as e:
            sections.append(DocumentSection(
                section_id=f"{doc_id}_err",
                title="Extraction Warning",
                content=f"PDF extraction error: {str(e)}",
                page=1,
                section_type="metadata"
            ))
        return sections

    @classmethod
    def parse_docx(cls, file_path: Path, doc_id: str) -> List[DocumentSection]:
        sections = []
        try:
            doc = DocxDocument(str(file_path))
            current_heading = "Overview"
            current_buffer = []
            
            # 1. Parse paragraphs and headings
            for p_idx, p in enumerate(doc.paragraphs):
                text = p.text.strip()
                if not text:
                    continue
                
                if p.style and ("Heading" in p.style.name or "Title" in p.style.name):
                    if current_buffer:
                        sections.append(DocumentSection(
                            section_id=f"{doc_id}_s{len(sections)+1}",
                            title=current_heading,
                            content="\n".join(current_buffer).strip(),
                            section_type="text"
                        ))
                        current_buffer = []
                    current_heading = text
                else:
                    current_buffer.append(text)
                    
            if current_buffer:
                sections.append(DocumentSection(
                    section_id=f"{doc_id}_s{len(sections)+1}",
                    title=current_heading,
                    content="\n".join(current_buffer).strip(),
                    section_type="text"
                ))

            # 2. Parse tables with structural preservation
            for t_idx, table in enumerate(doc.tables):
                table_lines = []
                for row in table.rows:
                    row_cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
                    if any(row_cells):
                        table_lines.append(" | ".join(row_cells))
                if table_lines:
                    sections.append(DocumentSection(
                        section_id=f"{doc_id}_tbl{t_idx+1}",
                        title=f"{current_heading} (Table {t_idx+1})",
                        content="\n".join(table_lines),
                        section_type="table"
                    ))
        except Exception as e:
            sections.append(DocumentSection(
                section_id=f"{doc_id}_err",
                title="DOCX Warning",
                content=f"DOCX extraction note: {str(e)}",
                section_type="metadata"
            ))
        return sections

    @classmethod
    def parse_excel(cls, file_path: Path, doc_id: str) -> List[DocumentSection]:
        sections = []
        try:
            wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
            for sheet_name in wb.sheetnames:
                df = pd.read_excel(str(file_path), sheet_name=sheet_name)
                df = df.dropna(how='all')
                if df.empty:
                    continue
                
                # Format dataframe into readable structured rows
                summary_lines = [f"Sheet: {sheet_name} (Total rows: {len(df)})"]
                cols = [str(c).strip() for c in df.columns if not str(c).startswith("Unnamed")]
                if cols:
                    summary_lines.append(f"Fields: {', '.join(cols)}")
                
                # Convert rows to descriptive textual representation
                for r_idx, row in df.head(150).iterrows():
                    row_items = [f"{k}: {v}" for k, v in row.items() if pd.notna(v) and str(v).strip()]
                    if row_items:
                        summary_lines.append(" • " + " | ".join(row_items))
                
                sections.append(DocumentSection(
                    section_id=f"{doc_id}_{sheet_name[:15]}",
                    title=f"Spreadsheet Sheet: {sheet_name}",
                    content="\n".join(summary_lines),
                    section_type="table",
                    metadata={"sheet_name": sheet_name, "row_count": len(df)}
                ))
        except Exception as e:
            sections.append(DocumentSection(
                section_id=f"{doc_id}_err",
                title="Excel Warning",
                content=f"Excel extraction note: {str(e)}",
                section_type="metadata"
            ))
        return sections

    @classmethod
    def parse_pptx(cls, file_path: Path, doc_id: str) -> List[DocumentSection]:
        sections = []
        try:
            prs = Presentation(str(file_path))
            for idx, slide in enumerate(prs.slides):
                slide_num = idx + 1
                slide_texts = []
                slide_title = f"Slide {slide_num}"
                
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            t = paragraph.text.strip()
                            if t:
                                slide_texts.append(t)
                
                if slide_texts:
                    if len(slide_texts) > 0 and len(slide_texts[0]) < 80:
                        slide_title = slide_texts[0]
                    sections.append(DocumentSection(
                        section_id=f"{doc_id}_slide{slide_num}",
                        title=slide_title,
                        content="\n".join(slide_texts),
                        page=slide_num,
                        section_type="slide"
                    ))
        except Exception as e:
            sections.append(DocumentSection(
                section_id=f"{doc_id}_err",
                title="Presentation Warning",
                content=f"PPTX extraction note: {str(e)}",
                section_type="metadata"
            ))
        return sections

    @classmethod
    def parse_text(cls, file_path: Path, doc_id: str) -> List[DocumentSection]:
        sections = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    sections.append(DocumentSection(
                        section_id=f"{doc_id}_sec1",
                        title=file_path.stem,
                        content=content,
                        section_type="text"
                    ))
        except Exception as e:
            sections.append(DocumentSection(
                section_id=f"{doc_id}_err",
                title="Text File Warning",
                content=str(e),
                section_type="metadata"
            ))
        return sections

    @classmethod
    def parse_document(cls, file_path: Path, base_data_dir: Path = DATA_DIR) -> Optional[ParsedDocument]:
        """Parse any document into a unified institutional knowledge object."""
        if not file_path.is_file():
            return None
        
        try:
            rel_path = Path(file_path.relative_to(base_data_dir)).as_posix()
        except ValueError:
            rel_path = file_path.name

        ext = file_path.suffix.lower()
        doc_id = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:12]
        meta = cls.infer_metadata(file_path, rel_path)
        checksum = cls.compute_checksum(file_path)

        sections: List[DocumentSection] = []

        if ext == ".pdf":
            sections = cls.parse_pdf(file_path, doc_id)
        elif ext == ".docx":
            sections = cls.parse_docx(file_path, doc_id)
        elif ext in [".xlsx", ".xls"]:
            sections = cls.parse_excel(file_path, doc_id)
        elif ext == ".pptx":
            sections = cls.parse_pptx(file_path, doc_id)
        elif ext in [".txt", ".md", ".csv", ".json"]:
            sections = cls.parse_text(file_path, doc_id)
        elif ext in [".jpg", ".jpeg", ".png"]:
            # Multimodal media placeholder with rich metadata
            sections = [DocumentSection(
                section_id=f"{doc_id}_media",
                title=f"Campus Image / Media: {file_path.stem}",
                content=f"Image asset in department '{meta['department']}'. File: {file_path.name}",
                section_type="media"
            )]
        else:
            return None

        # Filter empty sections
        valid_sections = [s for s in sections if s.content and len(s.content.strip()) > 5]
        if not valid_sections:
            return None

        return ParsedDocument(
            doc_id=doc_id,
            filename=file_path.name,
            relative_path=rel_path,
            file_type=ext.replace(".", ""),
            department=meta["department"],
            category=meta["category"],
            academic_year=meta["academic_year"],
            version=meta["version"],
            access_policy=meta["access_policy"],
            sections=valid_sections,
            checksum=checksum,
            metadata=meta
        )
