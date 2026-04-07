"""
Day 1 - Exercise 4: Structured output
Goal: extract typed data from LLM output — foundational for agents.
Use Pydantic models (you know these from FastAPI) to define the schema.

This is how agents parse tool results, plans, and decisions reliably.
"""

import json
import anthropic
from pydantic import BaseModel

client = anthropic.Anthropic()


# Define what you want back — same as a Pydantic response model in FastAPI
class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    salary_min: int | None
    salary_max: int | None
    remote: bool
    required_skills: list[str]


def extract_job_posting(raw_text: str) -> JobPosting:
    """
    Use Claude to extract structured data from unstructured text.
    Pattern: instruct model to respond ONLY with JSON, then parse.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="""You are a data extraction engine.
Extract the requested information and respond ONLY with valid JSON.
No explanation, no markdown, no code blocks — just the raw JSON object.""",
        messages=[
            {
                "role": "user",
                "content": f"""Extract job posting details from this text.
Return JSON matching this schema:
{json.dumps(JobPosting.model_json_schema(), indent=2)}

Text:
{raw_text}"""
            }
        ],
    )

    raw_json = response.content[0].text
    data = json.loads(raw_json)
    return JobPosting(**data)


if __name__ == "__main__":
    sample_text = """
    We're hiring a Senior Python Engineer at TechCorp in San Francisco (remote friendly).
    You'll work on distributed backend systems. Salary range: $150k-$200k.
    Requirements: Python, FastAPI, PostgreSQL, Docker, Kubernetes. 5+ years experience.
    """

    result = extract_job_posting(sample_text)

    print("Extracted job posting:")
    print(f"  Title    : {result.title}")
    print(f"  Company  : {result.company}")
    print(f"  Location : {result.location}")
    print(f"  Remote   : {result.remote}")
    print(f"  Salary   : ${result.salary_min:,} - ${result.salary_max:,}")
    print(f"  Skills   : {', '.join(result.required_skills)}")

    # This pattern is used in agents to:
    # 1. Parse LLM "decisions" into typed structs
    # 2. Extract data from tool results
    # 3. Validate agent outputs before acting on them
