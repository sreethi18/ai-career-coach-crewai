import os
import tempfile
import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the very first st call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🎯",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — clean, minimal styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 780px; margin: auto; }
    .result-box {
    background: #f8f9fa;
    border-left: 4px solid #4A90D9;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
    font-size: 14px;
    line-height: 1.8;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    word-wrap: break-word;
    overflow-x: auto;
    box-sizing: border-box;
   }
    .task-header {
        font-size: 15px;
        font-weight: 600;
        color: #1a1a2e;
        margin: 1.5rem 0 0.4rem;
        padding-bottom: 4px;
        border-bottom: 1px solid #e0e0e0;
    }
    .status-running {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 6px;
        padding: 8px 14px;
        font-size: 13px;
        color: #7c5c00;
        margin: 6px 0;
    }
    .status-done {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 6px;
        padding: 8px 14px;
        font-size: 13px;
        color: #1b5e20;
        margin: 6px 0;
    }
            .stButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    border: none;
    border-radius: 14px;
    height: 55px;
    font-size: 18px;
    font-weight: 700;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37,99,235,0.35);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PDF EXTRACTION — same logic as main.py
# ─────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    """Write uploaded bytes to a temp file, then extract text with pdfplumber."""
    try:
        import pdfplumber
        # Save the uploaded bytes to a temp file so pdfplumber can open it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        os.unlink(tmp_path)  # clean up temp file
        return text.strip()

    except ImportError:
        st.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""


# ─────────────────────────────────────────────
# CREW BUILDER + RUNNER
# ─────────────────────────────────────────────
def run_career_coach(resume_snippet: str, target_role: str) -> dict:
    """Build agents, tasks, run crew, return all 4 task outputs as a dict."""

    search_tool = SerperDevTool()

    # ── Agents ──────────────────────────────
    resume_analyst = Agent(
        role="Resume Analyst",
        goal="Extract all skills from the resume and provide improvement suggestions",
        backstory=(
            "You are a senior HR professional with 15 years of experience reviewing "
            "resumes for top tech companies. The resume text is provided directly in "
            "the task — you do NOT need any file tools."
        ),
        tools=[search_tool],
        verbose=False   # False keeps terminal clean; UI shows progress instead
    )

    job_matcher = Agent(
        role="Job Matcher",
        goal="Suggest best-fit job roles and identify skill gaps",
        backstory=(
            "You are a tech recruiter who matches candidates to roles using real "
            "job market data. You focus on the candidate's target role when given."
        ),
        tools=[search_tool],
        verbose=False
    )

    interview_coach = Agent(
        role="Interview Coach",
        goal="Prepare the candidate with tailored interview questions and ideal answers",
        backstory=(
            "You are a career coach who crafts role-specific, personalised interview "
            "questions based on the candidate's actual resume and target role."
        ),
        verbose=False
    )

    career_strategist = Agent(
        role="Career Strategist",
        goal="Build a concise 6-month career roadmap with concrete monthly milestones",
        backstory=(
            "You are a career strategist who designs actionable growth plans, "
            "researching current market trends and recommending specific certifications."
        ),
        tools=[search_tool],
        verbose=False
    )

    # Target role line — injected into tasks if user provided one
    role_line = (
        f"\nThe candidate's TARGET ROLE is: {target_role}. Prioritise this role.\n"
        if target_role.strip() else ""
    )

    # ── Tasks ────────────────────────────────
    task1 = Task(
        description=(
            "Here is the candidate's resume text:\n\n"
            f"--- RESUME START ---\n{resume_snippet}\n--- RESUME END ---\n"
            f"{role_line}\n"
            "Be CONCISE. Under 300 words.\n"
            "1. SKILLS (one line): comma-separated list of all skills.\n"
            "2. TOP 5 RESUME FIXES: exactly 5 bullets, one line each.\n"
            "   Format: • Fix: <what to do>\n"
            "No paragraphs. Short bullets only."
        ),
        expected_output=(
            "SKILLS: ...\n\nTOP 5 RESUME FIXES:\n• Fix: ...\n• Fix: ...\n"
            "• Fix: ...\n• Fix: ...\n• Fix: ..."
        ),
        agent=resume_analyst
    )

    task2 = Task(
        description=(
            f"Using Task 1 output.{role_line}"
            "Be CONCISE. Under 250 words.\n"
            "1. BEST FIT ROLES: exactly 3 roles.\n"
            "   Format: • <Role> | Has: X, Y | Missing: Z\n"
            "2. TOP 3 SKILLS TO LEARN: one line each.\n"
            "   Format: • <Skill> — <why in 5 words>\n"
            "No paragraphs. Short bullets only."
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
            f"Using Tasks 1 and 2.{role_line}"
            "Write exactly 3 interview questions. Under 300 words total.\n"
            "For each:\n"
            "Q: <question>\n"
            "Answer hint: <1-2 sentences>\n"
            "Tip: <one specific tip for this candidate>\n\n"
            "Blank line between questions. No paragraphs."
        ),
        expected_output="3 questions each with answer hint and tip.",
        agent=interview_coach,
        context=[task1, task2]
    )

    task4 = Task(
        description=(
            f"Using all previous analysis.{role_line}"
            "Write a 6-month roadmap. Under 400 words.\n"
            "For EACH month:\n"
            "Month N — <focus>\n"
            "• Learn: <skill>\n"
            "• Cert: <name or None>\n"
            "• Do: <one deliverable>\n\n"
            "End with: PRIORITY: Month X → Month Y → ...\n"
            "No paragraphs. No tables."
        ),
        expected_output="6 month entries in bullet format + priority line.",
        agent=career_strategist,
        context=[task1, task2, task3]
    )

    # ── Run ──────────────────────────────────
    crew = Crew(
        agents=[resume_analyst, job_matcher, interview_coach, career_strategist],
        tasks=[task1, task2, task3, task4],
        verbose=False
    )
    crew.kickoff()

    return {
        "Task 1 — Resume Analysis & Skills":   task1.output.raw if task1.output else "",
        "Task 2 — Job Matching & Skill Gaps":  task2.output.raw if task2.output else "",
        "Task 3 — Interview Preparation":       task3.output.raw if task3.output else "",
        "Task 4 — 6-Month Career Roadmap":      task4.output.raw if task4.output else "",
    }


# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────
st.image(
    "banner.png",
    use_container_width=True
)
st.markdown("""
<div style='text-align:center; margin-top:20px; margin-bottom:40px;'>

<h2 style='font-size:40px; color:#1e293b; font-weight:700;'>
Your Personal AI-Powered Career Assistant
</h2>

<p style='font-size:18px; color:gray; max-width:800px; margin:auto; line-height:1.8;'>

Upload your resume and let multiple AI agents analyse your profile,
match career opportunities, prepare interview questions,
and build a personalised roadmap for your future.

</p>

</div>
""", unsafe_allow_html=True)
st.markdown("""
<p style='
text-align:center;
color:gray;
font-size:18px;
margin-top:-10px;
margin-bottom:40px;
'>
Upload your resume PDF and get a personalised career plan in minutes.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='
text-align:center;
margin-top:20px;
margin-bottom:30px;
font-size:28px;
color:#1e293b;
'>
✨ What This Platform Offers
</h3>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

features = [
    ("📄", "Resume Review"),
    ("💼", "Job Matching"),
    ("🎤", "Interview Prep"),
    ("🚀", "Roadmap")
]

colors = ["#2563eb", "#7c3aed", "#ec4899", "#f59e0b"]

for col, (icon, title), color in zip(
    [col1, col2, col3, col4],
    features,
    colors
):
    with col:
        st.markdown(f"""
        <div style='
            background:white;
            padding:22px;
            border-radius:18px;
            text-align:center;
            box-shadow:0 6px 20px rgba(0,0,0,0.08);
            border-top:5px solid {color};
            min-height:140px;
        '>

        <div style='font-size:44px;'>{icon}</div>

        <div style='
            margin-top:14px;
            font-size:18px;
            font-weight:700;
            color:#1e293b;
        '>
            {title}
        </div>

        </div>
        """, unsafe_allow_html=True)
st.divider()

# ── Input section ─────────────────────────────


col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="Your PDF is read locally — nothing is stored."
    )

with col2:
    target_role = st.text_input(
        "Target role (optional)",
        placeholder="e.g. Salesforce Developer",
        help="If filled, all agents focus on this role."
    )

# Show character count once file uploaded
resume_text = ""
if uploaded_file:
    resume_text = extract_pdf_text(uploaded_file.read())
    if resume_text:
        st.success(f"✅ Resume loaded — {len(resume_text):,} characters extracted.")
    else:
        st.error("Could not extract text. The PDF may be a scanned image.")

st.divider()

# ── Run button ────────────────────────────────
run_btn = st.button(
    "🚀 Analyse My Resume",
    disabled=not bool(resume_text),
    use_container_width=True,
    type="primary"
)

# ── Results ───────────────────────────────────
if run_btn and resume_text:

    resume_snippet = resume_text[:4000]

    # Progress display
    progress_placeholder = st.empty()
    results_placeholder  = st.empty()

    task_names = [
        "📋 Analysing resume & extracting skills...",
        "💼 Matching job roles & finding skill gaps...",
        "🎤 Preparing interview questions...",
        "🗺️ Building your 6-month roadmap...",
    ]
    with progress_placeholder.container():
        st.info("⏳ Your AI crew is working — this takes 1–3 minutes...")
        for name in task_names:
            st.markdown(
                f'<div class="status-running">🔄 {name}</div>',
                unsafe_allow_html=True
            )

    # Run the crew
    try:
        outputs = run_career_coach(resume_snippet, target_role)

        # Clear progress, show results
        progress_placeholder.empty()

        with results_placeholder.container():
            st.success("✅ Analysis complete!")

            # Show each task result in its own styled section
            tab1, tab2, tab3, tab4 = st.tabs([
                 "📋 Resume",
                 "💼 Jobs",
                 "🎤 Interview",
                 "🗺️ Roadmap"
            ])

            tabs = [tab1, tab2, tab3, tab4]

            for tab, (label, content) in zip(tabs, outputs.items()):
                with tab:
                    st.markdown(
                        f'<div class="result-box">{content}</div>',
                        unsafe_allow_html=True
                    )

            # Download button — save full output as .txt
            st.divider()
            full_text = "\n\n".join(
                f"=== {label} ===\n{content}"
                for label, content in outputs.items()
            )
            st.download_button(
                label="⬇️ Download full report",
                data=full_text,
                file_name="career_coach_report.txt",
                mime="text/plain",
                use_container_width=True
            )

    except Exception as e:
        progress_placeholder.empty()
        st.error(f"Something went wrong: {e}")
        st.info("Check that your OPENAI_API_KEY and SERPER_API_KEY are set correctly.")

# ── Footer ────────────────────────────────────
st.divider()
st.caption("Built with CrewAI · pdfplumber · Streamlit © 2026 Sreethi")
