"""
Day 1 - Activity 3: CV / Resume Extractor
==========================================
CONCEPT: Structured output — extract typed data from unstructured text.
         This is one of the highest-value LLM use cases in production.
         You've done this with FastAPI + Pydantic. Same idea here.

TASK: Complete the extract_cv() function using Pydantic + Claude.
      The function must return a fully populated CVData object.
      Run the file to verify your output matches the expected fields.
"""

import json
import anthropic
from pydantic import BaseModel

client = anthropic.Anthropic()


# ── Schema — do not change these models ──────────────────────────────────────

class Experience(BaseModel):
    company: str
    title: str
    years: float          # e.g. 1.5 for 18 months
    technologies: list[str]

class CVData(BaseModel):
    full_name: str
    total_years_experience: float
    primary_language: str       # e.g. "Python"
    top_skills: list[str]       # top 5 skills
    experiences: list[Experience]
    is_remote_worker: bool
    highest_education: str      # e.g. "Bachelor of Science in Computer Science"


# ── Your task ────────────────────────────────────────────────────────────────

def extract_cv(raw_text: str) -> CVData:
    """
    TODO:
    1. Call Claude with a system prompt that enforces JSON-only output
    2. Ask it to extract CVData fields from raw_text
    3. Pass the CVData JSON schema so the model knows the exact structure
       Hint: CVData.model_json_schema() gives you the schema
    4. Parse the JSON response and return a CVData instance

    Tip: If Claude wraps JSON in ```json ``` blocks, strip them first.
    """
    pass


# ── Test with your own CV text ────────────────────────────────────────────────

SAMPLE_CV = """
Sharjeel Sohail — Senior Backend Engineer
5+ years of Python experience. Currently at Uforia (Leadpages) since August 2025.
Expert in FastAPI, Django, PostgreSQL, TimescaleDB, Redis, RabbitMQ, Kafka.
Previously at Merik Solutions (Jun 2024 - Aug 2025) and HyperNym (Sep 2022 - Jun 2024).
At HyperNym worked on IoT platforms with Django, DRF, Apache Kafka for Vodafone Qatar.
Works remotely from Rawalpindi, Pakistan.
BS in Computer Science from COMSATS University Islamabad (2017-2021).
"""


if __name__ == "__main__":
    print("Extracting CV data...\n")
    result = extract_cv(SAMPLE_CV)

    if result is None:
        print("❌ extract_cv() returned None — complete the TODO")
    else:
        print(f"Name                 : {result.full_name}")
        print(f"Total experience     : {result.total_years_experience} years")
        print(f"Primary language     : {result.primary_language}")
        print(f"Top skills           : {', '.join(result.top_skills)}")
        print(f"Remote worker        : {result.is_remote_worker}")
        print(f"Education            : {result.highest_education}")
        print(f"\nExperience entries   : {len(result.experiences)}")
        for exp in result.experiences:
            print(f"  • {exp.title} @ {exp.company} ({exp.years}y) — {', '.join(exp.technologies[:3])}")
