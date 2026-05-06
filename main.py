import sys
import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# ─────────────────────────────────────────────
# 1. READ RESUME — extract text from PDF first,
#    BEFORE any agent runs. No FileReadTool needed.
# ─────────────────────────────────────────────

def extract_pdf_text(path: str) -> str:
    """Extract text from a PDF using pdfplumber (best for resumes)."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except ImportError:
        pass

    # Fallback: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except ImportError:
        pass

    raise RuntimeError(
        "No PDF library found. Install one:\n"
        "  pip install pdfplumber\n"
        "  pip install pypdf"
    )


resume_text = ""
resume_path = input("Enter path to your resume PDF (or press Enter to paste text): ").strip().strip('"').strip("'")

if resume_path:
    if not os.path.exists(resume_path):
        print(f"❌ File not found: {resume_path}")
        sys.exit(1)
    print("📄 Reading PDF...")
    resume_text = extract_pdf_text(resume_path)
    if not resume_text:
        print("⚠️  PDF extracted no text (may be a scanned image). Falling back to paste.")
    else:
        print(f"✅ Resume extracted — {len(resume_text)} characters read.\n")

if not resume_text:
    print("Paste your resume text below. Press Enter twice when done:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    resume_text = "\n".join(lines)
    print("✅ Resume text received.\n")

# Truncate to avoid token overload (keep first ~4000 chars)
resume_snippet = resume_text

# ─────────────────────────────────────────────
# 2. AGENTS
# ─────────────────────────────────────────────
search_tool = SerperDevTool()

resume_analyst = Agent(
    role="Resume Analyst",
    goal="Extract all skills from the resume and provide detailed improvement suggestions",
    backstory=(
        "You are a senior HR professional with 15 years of experience reviewing "
        "resumes for top tech companies. You identify skill gaps, weak phrasing, "
        "and missing sections. The resume text is provided directly in the task — "
        "you do NOT need any file tools."
    ),
    tools=[search_tool],
    verbose=True
)

job_matcher = Agent(
    role="Job Matcher",
    goal="Suggest best-fit job roles and identify skill gaps based on the resume",
    backstory=(
        "You are a tech recruiter who matches candidates to roles using real "
        "job market data. You use the resume analysis output to recommend "
        "specific job titles and highlight gaps to close."
    ),
    tools=[search_tool],
    verbose=True
)

interview_coach = Agent(
    role="Interview Coach",
    goal="Prepare the candidate with tailored interview questions and ideal answers",
    backstory=(
        "You are a career coach who has helped hundreds of candidates land jobs "
        "at top companies. You craft role-specific, personalized interview questions "
        "based on the candidate's actual resume and target roles."
    ),
    verbose=True
)

career_strategist = Agent(
    role="Career Strategist",
    goal="Build a detailed 6-month career roadmap with concrete monthly milestones",
    backstory=(
        "You are a career strategist who designs actionable growth plans. "
        "You research current market trends and tailor certifications, resources, "
        "and monthly goals to close the specific gaps found in this candidate's profile."
    ),
    tools=[search_tool],
    verbose=True
)

# ─────────────────────────────────────────────
# 3. TASKS — resume text injected directly, concise output format enforced
# ─────────────────────────────────────────────

task1 = Task(
    description=(
        "Here is the candidate's resume text (extracted from their PDF):\n\n"
        f"--- RESUME START ---\n{resume_snippet}\n--- RESUME END ---\n\n"
        "Be CONCISE. Keep the entire output under 300 words.\n\n"
        "1. SKILLS (one line): List all skills as comma-separated values.\n"
        "2. TOP 5 RESUME FIXES: Exactly 5 bullet points, one line each.\n"
        "   Format: • Fix: <what to do>\n"
        "No long explanations. No paragraphs. Short bullets only."
    ),
    expected_output=(
        "SKILLS: <comma-separated list>\n\n"
        "TOP 5 RESUME FIXES:\n"
        "• Fix: ...\n• Fix: ...\n• Fix: ...\n• Fix: ...\n• Fix: ..."
    ),
    agent=resume_analyst
)

task2 = Task(
    description=(
        "Using the resume analysis from Task 1. Be CONCISE. Under 250 words total.\n\n"
        "1. BEST FIT ROLES: List exactly 3 job roles.\n"
        "   Format: • <Role> | Has: X, Y | Missing: Z\n\n"
        "2. TOP 3 SKILLS TO LEARN: One line each.\n"
        "   Format: • <Skill> — <why in 5 words>\n\n"
        "No paragraphs. No extra headers. Short bullets only."
    ),
    expected_output=(
        "BEST FIT ROLES:\n• ...\n• ...\n• ...\n\n"
        "TOP 3 SKILLS TO LEARN:\n• ...\n• ...\n• ..."
    ),
    agent=job_matcher,
    context=[task1]
)

task3 = Task(
    description=(
        "Using Tasks 1 and 2, write exactly 3 interview questions. Under 300 words total.\n\n"
        "For each question use this exact compact format:\n"
        "Q: <question in one sentence>\n"
        "Answer hint: <ideal answer in 1-2 sentences>\n"
        "Tip: <one specific tip for this candidate in 1 sentence>\n\n"
        "Separate each with a blank line. No long paragraphs. No extra sections."
    ),
    expected_output=(
        "3 interview questions, each with a one-line answer hint and one-line tip."
    ),
    agent=interview_coach,
    context=[task1, task2]
)

task4 = Task(
    description=(
        "Using all previous analysis, write a 6-month roadmap. Under 400 words total.\n\n"
        "Use this exact format for EACH month:\n"
        "Month N — <focus area>\n"
        "• Learn: <skill or topic>\n"
        "• Cert: <certification name or None>\n"
        "• Do: <one concrete deliverable>\n\n"
        "After Month 6, add exactly one line:\n"
        "PRIORITY: Month X → Month Y → Month Z → ... (most urgent first)\n\n"
        "No paragraphs. No tables. Strict bullet format only."
    ),
    expected_output=(
        "6 month entries in compact bullet format, followed by a single priority line."
    ),
    agent=career_strategist,
    context=[task1, task2, task3]
)

# 4. RUN CREW
# ─────────────────────────────────────────────
crew = Crew(
    agents=[resume_analyst, job_matcher, interview_coach, career_strategist],
    tasks=[task1, task2, task3, task4],
    verbose=True
)

print("🚀 Starting AI Career Coach...\n")
crew.kickoff()

# ─────────────────────────────────────────────
# 5. COLLECT & SAVE ALL TASK OUTPUTS
# ─────────────────────────────────────────────
task_labels = [
    "TASK 1 — Resume Analysis & Skill Extraction",
    "TASK 2 — Job Matching & Skill Gaps",
    "TASK 3 — Interview Preparation",
    "TASK 4 — 6-Month Career Roadmap",
]

output_sections = []
for label, task in zip(task_labels, [task1, task2, task3, task4]):
    raw = task.output.raw if task.output else "(No output)"
    section = f"\n{'='*60}\n{label}\n{'='*60}\n{raw}\n"
    output_sections.append(section)

full_output = "===== AI CAREER COACH OUTPUT =====\n" + "".join(output_sections)

print("\n" + full_output)

with open("output.md", "w", encoding="utf-8") as f:
    f.write(full_output)

print("\n✅ Full output saved to output.md")