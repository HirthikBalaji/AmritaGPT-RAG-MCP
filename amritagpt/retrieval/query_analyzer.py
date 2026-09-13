"""
Query Understanding & Analyzer for AmritaGPT.
Detects user intent, extracts institutional entities (departments, programs, regulation codes, years),
and generates normalized search filters and query expansions.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re


@dataclass
class AnalyzedQuery:
    raw_query: str
    normalized_query: str
    intent: str
    program: Optional[str] = None
    department: Optional[str] = None
    regulation_code: Optional[str] = None
    academic_year: Optional[str] = None
    is_current_requested: bool = False
    expanded_terms: List[str] = field(default_factory=list)
    suggested_category: Optional[str] = None


class QueryAnalyzer:
    """Institutional query understanding engine."""

    DEPARTMENT_MAP = {
        "cse": "CSE",
        "computer science": "CSE",
        "cys": "CSE",
        "cyber security": "CSE",
        "aids": "AIDS",
        "ai&ds": "AIDS",
        "artificial intelligence and data science": "AIDS",
        "aie": "AIE",
        "artificial intelligence": "AIE",
        "ece": "ECE",
        "electronics": "ECE",
        "cce": "ECE",
        "mech": "MECH",
        "mechanical": "MECH",
        "rai": "RAI",
        "robotics": "RAI",
        "s&h": "S&H",
        "sciences and humanities": "S&H",
        "iqac": "IQAC",
        "cir": "CIR department",
        "hostel": "Hostel",
        "library": "Library",
        "exam cell": "Exam_Cell",
        "examination": "Exam_Cell",
        "timetable": "Time Table",
        "canteen": "Canteen",
        "health": "Health Center",
        "transport": "Transport Department"
    }

    INTENT_KEYWORDS = {
        "Academic Regulation": ["attendance", "credit", "grade", "sgpa", "cgpa", "pass", "fail", "malpractice", "condonation", "regulation", "r.", "rule", "probation"],
        "Examination": ["exam", "hall ticket", "schedule", "revaluation", "supplementary", "arrear", "seating", "marks"],
        "Curriculum & Syllabus": ["syllabus", "curriculum", "course", "elective", "prerequisite", "lab course", "btech curriculum"],
        "Class Timetable": ["timetable", "time table", "class timing", "slot", "room number", "period", "lab session"],
        "Faculty & Staff": ["faculty", "professor", "teacher", "hod", "cabin", "email", "designation", "phd supervisor"],
        "Campus & Residential": ["hostel", "room", "mess", "canteen", "bus", "transport", "gatepass", "doctor", "clinic", "health", "library", "book"],
        "Research & Grants": ["research", "paper", "patent", "journal", "project grant", "conference", "publication"],
        "Policies & Circulars": ["circular", "notice", "deadline", "date", "holiday", "fee payment"]
    }

    EXPANSIONS = {
        "attendance": ["minimum attendance percentage", "condonation", "medical certificate", "75% attendance", "80% attendance", "shortage of attendance"],
        "grading": ["letter grade", "grade point", "passed", "failed", "distinction", "first class"],
        "exam": ["end semester examination", "continuous assessment", "evaluation scheme", "hall ticket"],
        "malpractice": ["examination malpractice", "disciplinary action", "inquiry committee"],
        "revaluation": ["answer sheet review", "re-evaluation fee", "grade revision"]
    }

    @classmethod
    def analyze(cls, query: str) -> AnalyzedQuery:
        q_lower = query.lower().strip()

        # 1. Detect Intent
        detected_intent = "General Institutional Query"
        for intent, keywords in cls.INTENT_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", q_lower) for kw in keywords):
                detected_intent = intent
                break

        # 2. Detect Department
        detected_dept = None
        for key, dept_val in cls.DEPARTMENT_MAP.items():
            if re.search(rf"\b{re.escape(key)}\b", q_lower):
                detected_dept = dept_val
                break

        # 3. Detect Program
        program = None
        if re.search(r"\bb\.?tech\b", q_lower):
            program = "B.Tech"
        elif re.search(r"\bm\.?tech\b", q_lower):
            program = "M.Tech"
        elif re.search(r"\bph\.?d\b", q_lower):
            program = "PhD"
        elif re.search(r"\bmca\b", q_lower):
            program = "MCA"

        # 4. Detect Regulation Code (e.g. "R.4", "R.14", "Regulation 12")
        reg_match = re.search(r"\b(r\.\d+|regulation\s+\d+)\b", q_lower)
        regulation_code = reg_match.group(1).upper() if reg_match else None

        # 5. Detect Academic Year & Temporal Request
        year_match = re.search(r"\b(20[12]\d)\b", query)
        academic_year = year_match.group(1) if year_match else None
        
        is_current = bool(re.search(r"\b(current|latest|present|now|today|active|this\s+year)\b", q_lower))
        if is_current and not academic_year:
            academic_year = "2024"  # Default to latest active syllabus/regulation year

        # 6. Query Expansion
        expanded_terms = []
        for term, syns in cls.EXPANSIONS.items():
            if term in q_lower:
                expanded_terms.extend(syns[:3])

        return AnalyzedQuery(
            raw_query=query,
            normalized_query=q_lower,
            intent=detected_intent,
            program=program,
            department=detected_dept,
            regulation_code=regulation_code,
            academic_year=academic_year,
            is_current_requested=is_current,
            expanded_terms=expanded_terms
        )
