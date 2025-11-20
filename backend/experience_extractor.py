"""
Experience Extractor
====================

Pure-Python implementation that parses plain-text resumes, detects every
required experience pattern, normalises the detected employment ranges,
deduplicates overlaps, and ignores education / project timelines.

Supported patterns (per requirements)
-------------------------------------
1. Direct experience mentions (explicit years / months)
2. “To Till Date / To Present” ranges
3. Dash ranges (MMM YYYY - MMM YYYY)
4. Full month names
5. Two-digit year formats
6. “From…To…” sentences
7. Year-only ranges
8. Total experience statements
9. Project durations (ignored)
10. Variant month spellings
11. Timeline before role (2018 – Present | Title)
12. Multiple dates in one sentence (parenthetical)
13. Month+Year without spaces (Jan2014)
14. Slash year formats (2014/16)
15. Non-English month names
16. Hidden dates inside sentences
17. “Since …” statements
18. Month-year lines without ranges
"""

from __future__ import annotations

import logging
import re
from collections import namedtuple
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # dateutil is preferred for accurate relative deltas
    from dateutil.relativedelta import relativedelta  # type: ignore
except ImportError:  # pragma: no cover
    class relativedelta:  # type: ignore
        def __init__(self, days: int = 0, months: int = 0, **_: Any) -> None:
            self.days = days + months * 30

        def __radd__(self, other: date) -> date:
            return other + datetime.timedelta(days=self.days)

DateRange = namedtuple("DateRange", ["start", "end"])

MONTH_LOOKUP = {
    # English & common abbreviations
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "sepr": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
    # Spanish
    "ene": 1,
    "enero": 1,
    "abr": 4,
    "abril": 4,
    "ago": 8,
    "agosto": 8,
    "dic": 12,
    "diciembre": 12,
    # German
    "märz": 3,
    "maerz": 3,
    "mai": 5,
    "okt": 10,
    "oktober": 10,
    "dez": 12,
    "dezember": 12,
    # French
    "février": 2,
    "fevrier": 2,
    "août": 8,
    "aout": 8,
}

EDUCATION_HEADINGS = [
    "education",
    "academic qualifications",
    "academics",
    "certifications",
    "training",
    "courses",
    "graduation",
]

PROJECT_HEADINGS = [
    "project",
    "projects",
    "project experience",
    "project details",
    "project portfolio",
]

PROJECT_IGNORE_PATTERNS = [
    r"\bproject\s+duration\s+\d+\s*(?:months?|yrs?)",
    r"\b\d+\s*-\s*(?:month|year)\s+project\b",
]

DIRECT_EXPERIENCE_PATTERNS = [
    # Pattern 1: "Overall/Total Experience: X years Y months"
    r"(?:overall|total)\s+experience[:\s]+(?P<years>\d+)\s*(?:years?|yrs?)\s+(?P<months>\d+)\s*(?:months?|mths?)",
    
    # Pattern 2: "Overall/Total Experience: X.Y years" (decimal years)
    r"(?:overall|total)\s+experience[:\s]+(?P<num>\d+\.\d+)\s*(?:years?|yrs?)",
    
    # Pattern 3: "Total Exp: X.Y years"
    r"total\s+exp[:\s]+(?P<num>\d+\.\d+)\s*(?:years?|yrs?)",
    
    # Pattern 4: "X.Y years" (standalone decimal years - MUST come before integer patterns)
    r"(?P<num>\d+\.\d+)\s*(?:years?|yrs?)(?:\s+(?:of\s+)?(?:experience|expertise|exp))?",
    
    # Pattern 5: "X.Y years of experience" (decimal years with "of experience")
    r"(?P<num>\d+\.\d+)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|expertise|exp)",
    
    # Pattern 6: "Over X.Y years of experience"
    r"over\s+(?P<num>\d+\.\d+)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|expertise|exp)",
    
    # Pattern 7: "Nearly/About/Approximately X.Y years"
    r"(?:nearly|about|approximately|around|almost)\s+(?P<num>\d+\.\d+)\s*(?:years?|yrs?)(?:\s+(?:of\s+)?(?:experience|expertise|exp))?",
    
    # Pattern 8: "X.Y+ years" (decimal with plus sign)
    r"(?P<num>\d+\.\d+)\+?\s*(?:years?|yrs?)(?:\s+(?:of\s+)?(?:experience|expertise|exp))?",
    
    # Pattern 9: "X.Y YOE" or "X.Y years of experience" (YOE abbreviation)
    r"(?P<num>\d+\.\d+)\s*(?:years?|yrs?|yoe)(?:\s+(?:of\s+)?(?:experience|expertise))?",
    
    # Pattern 10: "X.Y years in [field]" (experience in specific field)
    r"(?P<num>\d+\.\d+)\s*(?:years?|yrs?)(?:\s+(?:of\s+)?(?:experience|expertise|exp))?\s+in\s+",
    
    # Pattern 11: "Experience: X.Y years"
    r"experience[:\s]+(?P<num>\d+\.\d+)\s*(?:years?|yrs?)",
    
    # Pattern 12: "More than X.Y years" or "Less than X.Y years"
    r"(?:more\s+than|less\s+than|greater\s+than)\s+(?P<num>\d+\.\d+)\s*(?:years?|yrs?)(?:\s+(?:of\s+)?(?:experience|expertise|exp))?",
    
    # Pattern 13: "X years Y months" (convert to decimal - must come after decimal patterns)
    r"(?P<years>\d+)\s*(?:years?|yrs?)\s+(?P<months>\d+)\s*(?:months?|mths?)(?:\s+(?:of\s+)?(?:experience|expertise|exp))?",
    
    # Pattern 14: "Since YYYY, working/serving/acting"
    r"since\s+(?P<year>\d{4})\s*,?\s*(?:working|serving|acting)",
    
    # Pattern 15: "X years" (integer years, with optional +) - MUST check for word boundary to avoid matching "6" from "2.6"
    r"(?<!\d\.)(?P<num>\d+)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|expertise|exp)",
    
    # Pattern 16: "X years' experience" (with apostrophe, including corrupted characters)
    r"(?<!\d\.)(?P<num>\d+)\+?\s*(?:years?|yrs?)\s*['\u2019\uFFFD]\s*(?:experience|expertise|exp)",  # \uFFFD is replacement character for corrupted text
    
    # Pattern 17: "X years experience" (with corrupted characters between)
    r"(?<!\d\.)(?P<num>\d+)\s*\+?\s*(?:years?|yrs?)\s*[^\w\s]{0,3}\s*(?:experience|expertise|exp)",  # Handles corrupted characters between years and experience
    
    # Pattern 18: "Overall/Total Experience: X years" (integer, fallback)
    r"(?:overall|total)\s+experience[:\s]+(?P<num>\d+)\s*(?:years?|yrs?)",
    
    # Pattern 19: "Total Exp: X years" (integer, fallback)
    r"total\s+exp[:\s]+(?P<num>\d+)\s*(?:years?|yrs?)",
    
    # Pattern 20: "Experience: X years" (integer, fallback)
    r"experience[:\s]+(?P<num>\d+)\s*(?:years?|yrs?)",
]


class ExperienceExtractor:
    """Production-quality experience extractor."""

    def __init__(self, resume_text: str) -> None:
        self.raw_text = resume_text
        self.cleaned_text = self._clean_text(resume_text)
        self.lines = [ln.strip() for ln in self.cleaned_text.splitlines() if ln.strip()]
        self.current_date = datetime.now().date()
        self.ignored_entries: List[str] = []

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = re.sub(r"[•\t]", " ", text)
        text = re.sub(r"[ ]{2,}", " ", text)
        return text

    @staticmethod
    def _normalise_year(year_str: str) -> Optional[int]:
        try:
            year = int(year_str)
        except ValueError:
            return None
        if year < 100:
            year += 2000 if year <= 30 else 1900
        if 1900 <= year <= datetime.now().year + 1:
            return year
        return None

    @staticmethod
    def _normalise_month(month_str: str) -> Optional[int]:
        token = (
            month_str.lower()
            .replace(".", "")
            .replace("'", "")
            .replace("’", "")
            .strip()
        )
        return MONTH_LOOKUP.get(token)

    def _parse_token(self, token: str) -> Optional[date]:
        token = token.strip()
        if not token:
            return None

        match = re.match(r"([A-Za-z]+)\s+(\d{2,4})", token)
        if match:
            month = self._normalise_month(match.group(1))
            year = self._normalise_year(match.group(2))
            if month and year:
                return date(year, month, 1)

        match = re.match(r"([A-Za-z]+)(\d{2,4})", token)
        if match:
            month = self._normalise_month(match.group(1))
            year = self._normalise_year(match.group(2))
            if month and year:
                return date(year, month, 1)

        match = re.match(r"(\d{1,2})/(\d{2,4})", token)
        if match:
            month = int(match.group(1))
            year = self._normalise_year(match.group(2))
            if 1 <= month <= 12 and year:
                return date(year, month, 1)

        match = re.match(r"(\d{4})", token)
        if match:
            year = self._normalise_year(match.group(1))
            if year:
                return date(year, 1, 1)

        return None

    @staticmethod
    def _month_end(day: date) -> date:
        if day.month == 12:
            return date(day.year, 12, 31)
        return date(day.year, day.month + 1, 1) - relativedelta(days=1)

    @staticmethod
    def _merge_ranges(ranges: List[DateRange]) -> List[DateRange]:
        if not ranges:
            return []
        ranges.sort(key=lambda rng: (rng.start, rng.end))
        merged = [ranges[0]]
        for rng in ranges[1:]:
            last = merged[-1]
            if rng.start <= last.end:
                merged[-1] = DateRange(last.start, max(last.end, rng.end))
            else:
                merged.append(rng)
        return merged

    def _calc_years(self, ranges: List[DateRange]) -> float:
        total_days = sum((rng.end - rng.start).days + 1 for rng in ranges)
        return round(total_days / 365.25, 2)

    # ----------------------------------------------------------- edu stripping
    def _strip_sections(self, headings: Iterable[str]) -> str:
        """
        Strip education and project sections from text to avoid counting their dates as experience.
        Uses multiple patterns to handle various section formats.
        """
        text = self.raw_text
        # Pattern 1: Standard section with heading followed by content until next major section
        pattern1 = r"(?is)(" + "|".join(re.escape(h) for h in headings) + r")\b.*?(?=\n\s*(?:[A-Z][A-Za-z\s]{2,}:|experience|work|professional|skills|projects|certifications|achievements|awards)\b|\Z)"
        text = re.sub(pattern1, "", text)
        
        # Pattern 2: Section heading on its own line, content until blank line or next section
        pattern2 = r"(?is)^\s*(" + "|".join(re.escape(h) for h in headings) + r")\s*$.*?(?=\n\s*\n|\n\s*(?:experience|work|professional|skills|projects|certifications|achievements|awards)\b|\Z)"
        text = re.sub(pattern2, "", text, flags=re.MULTILINE)
        
        # Pattern 3: More aggressive - find section and remove everything until we see clear work-related keywords
        # This handles cases where section boundaries aren't clear
        work_keywords = r"(?:experience|work\s+experience|professional\s+experience|employment|work\s+history|career)"
        pattern3 = r"(?is)(" + "|".join(re.escape(h) for h in headings) + r")\b[^\n]*(?:\n[^\n]*)*?(?=" + work_keywords + r"|\Z)"
        text = re.sub(pattern3, "", text)
        
        return text

    # ----------------------------------------------------------- explicit exp
    def _extract_explicit_experience(self) -> Optional[float]:
        values: List[float] = []
        for pattern in DIRECT_EXPERIENCE_PATTERNS:
            for match in re.finditer(pattern, self.raw_text, flags=re.IGNORECASE):
                if match.groupdict().get("years") and match.groupdict().get("months"):
                    years = int(match.group("years"))
                    months = int(match.group("months"))
                    values.append(years + months / 12.0)
                elif match.groupdict().get("num"):
                    values.append(float(match.group("num")))
                elif match.groupdict().get("year"):
                    start_year = int(match.group("year"))
                    values.append(float(datetime.now().year - start_year))
        if values:
            value = max(values)
            logger.info("Explicit experience detected: %.2f years", value)
            return value
        return None

    # ----------------------------------------------------------- pattern scan
    def _collect_ranges(self, text: str) -> List[DateRange]:
        ranges: List[DateRange] = []
        add = ranges.append

        def add_range(start_token: str, end_token: str) -> None:
            start = self._parse_token(start_token)
            end = (
                self.current_date
                if end_token.lower() in {"present", "current", "till date", "ongoing"}
                else self._parse_token(end_token)
            )
            if end and end_token.lower() not in {"present", "current", "till date", "ongoing"}:
                end = self._month_end(end)
            if start and end and start <= end:
                add(DateRange(start, end))

        # Pattern 1 & 3 & 10 (full month names, dash ranges, variant spellings)
        dash_pattern = re.compile(
            r"([A-Za-z]+)\s+(\d{2,4})\s*[-–—to]+\s*(present|current|ongoing|till date|[A-Za-z]+\s+\d{2,4}|\d{4})",
            re.IGNORECASE,
        )
        for match in dash_pattern.finditer(text):
            add_range(f"{match.group(1)} {match.group(2)}", match.group(3))

        # Pattern 2 / 6 – explicit “to present” sentences
        to_present_pattern = re.compile(
            r"([A-Za-z0-9/ ]+?)\s+to\s+(present|current|ongoing|till date|[A-Za-z0-9/ ]+)",
            re.IGNORECASE,
        )
        for match in to_present_pattern.finditer(text):
            add_range(match.group(1), match.group(2))

        # Pattern 5 – two digit years
        two_digit_pattern = re.compile(r"([A-Za-z]+)\s+(\d{2})\s*[-–—]\s*([A-Za-z]+)\s+(\d{2})")
        for match in two_digit_pattern.finditer(text):
            add_range(f"{match.group(1)} {match.group(2)}", f"{match.group(3)} {match.group(4)}")

        # Pattern 7 – year only
        year_range_pattern = re.compile(r"(\d{4})\s*[-–—to]+\s*(\d{4}|present|current|till date)")
        for match in year_range_pattern.finditer(text):
            start = self._parse_token(match.group(1))
            end_token = match.group(2)
            if start:
                end = self.current_date if end_token.lower() in {"present", "current", "till date"} else self._parse_token(end_token)
                if end and end_token.isdigit():
                    end = date(end.year, 12, 31)
                if end:
                    add(DateRange(start, end))

        # Pattern 8 / 12 – total exp statements already handled via explicit extraction

        # Pattern 11 – timeline before role
        timeline_pattern = re.compile(
            r"(\d{4}|[A-Za-z]+\s+\d{4})\s*[-–—]\s*(present|current|till date|\d{4}|[A-Za-z]+\s+\d{4})\s*\|",
            re.IGNORECASE,
        )
        for match in timeline_pattern.finditer(text):
            add_range(match.group(1), match.group(2))

        # Pattern 12 – parenthetical
        parenthetical = re.compile(r"\((\d{4})\s*[-–—]\s*(\d{4})\)")
        for match in parenthetical.finditer(text):
            add_range(match.group(1), match.group(2))

        # Pattern 13 – MonthYear without spaces
        condensed = re.compile(
            r"([A-Za-z]+\d{2,4})\s+(?:to|[-–—])\s+(present|current|till date|[A-Za-z]+\d{2,4})",
            re.IGNORECASE,
        )
        for match in condensed.finditer(text):
            add_range(match.group(1), match.group(2))

        # Pattern 14 – slash year formats
        slash_year = re.compile(r"(\d{4})/(\d{2,4})")
        for match in slash_year.finditer(text):
            start = self._parse_token(match.group(1))
            end = self._parse_token(match.group(2))
            if start and end:
                end = date(end.year, 12, 31)
                add(DateRange(start, end))

        # Pattern 15 – handled via MONTH_LOOKUP (non-English)

        # Pattern 16 – hidden dates “at IBM from 2012 – 2015”
        hidden = re.compile(
            r"(?:at|with|in)\s+[A-Za-z &]+\s+from\s+([A-Za-z0-9/ ]+)\s*[-–—to]+\s*(present|current|till date|[A-Za-z0-9/ ]+)",
            re.IGNORECASE,
        )
        for match in hidden.finditer(text):
            add_range(match.group(1), match.group(2))

        # Pattern 17 – since YEAR handled in explicit extraction (converted to range)

        # Pattern 18 – month-year lines without ranges
        ranges.extend(self._consecutive_month_lines(text))

        # Pattern 9 – project durations (ignored)
        for pattern in PROJECT_IGNORE_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                self.ignored_entries.append(match.group().strip())

        # Pattern 2 extra – slash (6/94 – present)
        slash_pattern = re.compile(
            r"(\d{1,2}/\d{2,4})\s*[-–—to]+\s*(present|current|till date|\d{1,2}/\d{2,4})",
            re.IGNORECASE,
        )
        for match in slash_pattern.finditer(text):
            add_range(match.group(1), match.group(2))

        return ranges

    def _consecutive_month_lines(self, text: str) -> List[DateRange]:
        lines = text.splitlines()
        parsed: List[Tuple[int, date]] = []
        for idx, line in enumerate(lines):
            match = re.match(r"([A-Za-z]+)\s+(\d{2,4})$", line.strip())
            if match:
                token = self._parse_token(" ".join(match.groups()))
                if token:
                    parsed.append((idx, token))
        parsed.sort(key=lambda item: item[0])
        ranges: List[DateRange] = []
        i = 0
        while i < len(parsed) - 1:
            start_idx, start_date = parsed[i]
            end_date = start_date
            j = i + 1
            while j < len(parsed):
                prev_line, prev_date = parsed[j - 1]
                current_line, current_date = parsed[j]
                if current_line - prev_line > 3:
                    break
                diff = (current_date.year - prev_date.year) * 12 + (current_date.month - prev_date.month)
                if diff == 1:
                    end_date = current_date
                    j += 1
                else:
                    break
            if end_date > start_date:
                ranges.append(DateRange(start_date, self._month_end(end_date)))
                i = j
            else:
                i += 1
        return ranges

    # ---------------------------------------------------------------- extract
    def extract(self) -> Dict[str, Any]:
        explicit = self._extract_explicit_experience()
        stripped_text = self._strip_sections(EDUCATION_HEADINGS + PROJECT_HEADINGS)
        ranges = self._collect_ranges(stripped_text)
        ranges = self._merge_ranges(ranges)
        total_years = self._calc_years(ranges)

        explicit_used = False
        # Always prioritize explicit experience when found (it's more reliable than date calculations)
        if explicit:
            total_years = explicit
            explicit_used = True
            logger.info(f"Using explicit experience: {explicit} years (from resume text)")
            if ranges:
                logger.info(f"Calculated from dates was {self._calc_years(ranges)} years (ignored, explicit takes priority)")

        segments = [
            {"start": rng.start.strftime("%Y-%m"), "end": rng.end.strftime("%Y-%m")}
            for rng in ranges
        ]

        return {
            "total_experience_years": total_years,
            "segments": segments,
            "ignored": self.ignored_entries,
            "explicit_experience_used": explicit_used,
        }


def extract_experience(resume_text: str) -> Dict[str, Any]:
    """Convenience wrapper."""
    return ExperienceExtractor(resume_text).extract()

