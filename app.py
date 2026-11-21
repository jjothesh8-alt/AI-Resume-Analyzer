# ============================================================
# AI Resume Analyzer & ATS Score Checker
# Built with Streamlit + Google Gemini API
# ============================================================

import streamlit as st      # Web application framework
import json                 # For parsing the AI's JSON response
import io                   # For reading the uploaded PDF bytes
import os                   # For reading environment variables

from dotenv import load_dotenv          # Loads variables from the .env file
from pypdf import PdfReader             # Extracts text from PDF files
from google import genai                # Official Google Gen AI Python SDK
from google.genai import types          # For building API request content

# ── Load environment variables from .env (local development) ──────────────────
load_dotenv()


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer & ATS Score Checker",
    page_icon="📄",
    layout="centered",
)


# ── Helper: get API key safely ────────────────────────────────────────────────
def get_api_key() -> str:
    """
    Read the Gemini API key from Streamlit secrets (cloud deployment)
    or from the .env file (local development).
    Returns the key as a string, or an empty string if not found.
    """
    # Streamlit Cloud stores secrets in st.secrets
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # Fallback: read from the .env file (already loaded above)
    return os.getenv("GEMINI_API_KEY", "")


# ── Helper: extract text from an uploaded PDF ─────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    """
    Takes the file object from st.file_uploader and returns all
    text found across every page as a single string.

    pypdf's PdfReader accepts a file-like object (io.BytesIO) so
    we wrap the raw bytes — this avoids saving anything to disk.
    """
    pdf_bytes = uploaded_file.read()          # Raw bytes from the upload
    pdf_file  = io.BytesIO(pdf_bytes)         # Wrap bytes in a file-like object
    reader    = PdfReader(pdf_file)           # pypdf reads the PDF

    all_text = []
    for page in reader.pages:
        page_text = page.extract_text()       # Extract text from one page
        if page_text:                         # Some pages may return None
            all_text.append(page_text)

    return "\n".join(all_text)                # Join pages with a newline


# ── Helper: build the Gemini prompt ──────────────────────────────────────────
def build_prompt(resume_text: str, job_description: str) -> str:
    """
    Constructs the instruction we send to Gemini.
    We tell Gemini exactly what to analyse and the JSON format we expect.
    """
    prompt = f"""
You are an expert ATS (Applicant Tracking System) and career coach.

Carefully compare the resume below against the given job description.

Return ONLY a valid JSON object — no explanation, no markdown, no code fences.
The JSON must follow this exact structure:

{{
  "match_score": <integer between 0 and 100>,
  "matching_skills": [<list of skills/keywords found in BOTH the resume and job description>],
  "missing_skills": [<list of skills/keywords required by the job but NOT explicitly mentioned in the resume>],
  "suggestions": [
    "<Specific suggestion 1>",
    "<Specific suggestion 2>",
    "<Specific suggestion 3>",
    "<Specific suggestion 4>",
    "<Specific suggestion 5>"
  ]
}}

Important notes:
- match_score must be a whole number (integer), not a string.
- missing_skills should list skills the job requires that the resume does not explicitly mention.
  Do NOT say the candidate definitely lacks a skill — they may have it but not listed it.
- suggestions must be exactly 5 items. Each must be specific and actionable.
- Do NOT include any text outside the JSON object.

--- RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{job_description}
"""
    return prompt.strip()


# ── Helper: call the Gemini API ───────────────────────────────────────────────
def analyze_with_gemini(resume_text: str, job_description: str, api_key: str) -> dict:
    """
    Sends the resume and job description to Gemini and returns
    the parsed JSON response as a Python dictionary.

    Raises:
        ValueError  – if the response is not valid JSON or fields are missing.
        Exception   – for API or network errors (re-raised with a friendly message).
    """
    # Create a Gemini client using the API key
    client = genai.Client(api_key=api_key)

    prompt = build_prompt(resume_text, job_description)

    # Call the Gemini model
    response = client.models.generate_content(
        model="gemini-3.5-flash",          # Model to use
        contents=prompt,                    # The full prompt text
        config=types.GenerateContentConfig(
            temperature=0.3,               # Lower = more consistent/deterministic
            max_output_tokens=1024,        # Enough for the JSON response
        ),
    )

    # Extract the raw text from the response
    raw_text = response.text.strip()

    # Sometimes Gemini wraps the JSON in markdown code fences — strip them
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        # Remove the first line (```json or ```) and the last line (```)
        lines = [line for line in lines if not line.startswith("```")]
        raw_text = "\n".join(lines).strip()

    # Parse the JSON string into a Python dictionary
    result = json.loads(raw_text)

    return result


# ── Helper: validate the parsed result ───────────────────────────────────────
def validate_result(result: dict) -> dict:
    """
    Checks that the result dictionary has all required fields
    and that the values are within expected ranges.
    Returns the (possibly corrected) result, or raises ValueError.
    """
    required_keys = ["match_score", "matching_skills", "missing_skills", "suggestions"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Gemini response is missing the field: '{key}'")

    # Ensure match_score is an integer between 0 and 100
    score = result["match_score"]
    if not isinstance(score, (int, float)):
        raise ValueError(f"match_score must be a number, got: {type(score)}")
    result["match_score"] = max(0, min(100, int(score)))   # Clamp to [0, 100]

    # Ensure suggestions has exactly 5 items
    suggestions = result["suggestions"]
    if not isinstance(suggestions, list):
        raise ValueError("suggestions must be a list.")
    # If Gemini returned fewer/more, pad or trim to exactly 5
    while len(suggestions) < 5:
        suggestions.append("Consider tailoring your resume more closely to the job description.")
    result["suggestions"] = suggestions[:5]

    return result


# ── Helper: display a colour-coded badge for the score ────────────────────────
def score_color(score: int) -> str:
    if score >= 75:
        return "🟢"
    elif score >= 50:
        return "🟡"
    else:
        return "🔴"


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

# ── Title & description ───────────────────────────────────────────────────────
st.title("📄 AI Resume Analyzer & ATS Score Checker")
st.markdown(
    """
    **Welcome!** Upload your resume (PDF) and paste the job description below.  
    Our AI will compare them and tell you:
    - ✅ How well your resume matches the job
    - 🔍 Which skills you've already highlighted
    - ⚠️ Which keywords are missing from your resume
    - 💡 Five actionable suggestions to improve your chances
    """
)
st.divider()

# ── Sidebar: API key status ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = get_api_key()
    if api_key:
        st.success("✅ Gemini API key loaded.")
    else:
        st.error("❌ Gemini API key not found.\nAdd it to your .env file or Streamlit secrets.")
    st.markdown("---")
    st.markdown("**How to add your API key:**")
    st.code("GEMINI_API_KEY=your_key_here", language="text")
    st.markdown("Save the above in a file named `.env` in the same folder as `app.py`.")

# ── Section 1: Resume Upload ──────────────────────────────────────────────────
st.subheader("📂 Step 1 — Upload Your Resume")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Only PDF files are accepted. Your file is never saved to disk.",
)

# ── Section 2: Job Description ────────────────────────────────────────────────
st.subheader("📋 Step 2 — Paste the Job Description")
job_description = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the full job description here…",
    help="Copy and paste the complete job posting text.",
)

st.divider()

# ── Section 3: Analyze Button ─────────────────────────────────────────────────
analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

# ── Main analysis logic ───────────────────────────────────────────────────────
if analyze_clicked:

    # --- Input validation ---
    has_error = False

    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF resume before clicking Analyze.")
        has_error = True

    if not job_description.strip():
        st.warning("⚠️ Please paste a job description before clicking Analyze.")
        has_error = True

    if not api_key:
        st.error("❌ Gemini API key is missing. Please add it to your .env file and restart.")
        has_error = True

    if not has_error:
        # --- Extract text from the PDF ---
        with st.spinner("Reading your resume…"):
            try:
                resume_text = extract_pdf_text(uploaded_file)
            except Exception as e:
                st.error(f"❌ Could not read the PDF file. Error: {e}")
                st.stop()

        if not resume_text.strip():
            st.error(
                "❌ No readable text was found in your PDF. "
                "This usually means the PDF is a scanned image. "
                "Please use a text-based PDF or copy-paste your resume text."
            )
            st.stop()

        # --- Call Gemini ---
        with st.spinner("🤖 AI is analyzing your resume… This may take a few seconds."):
            try:
                raw_result = analyze_with_gemini(resume_text, job_description, api_key)
            except json.JSONDecodeError:
                st.error(
                    "❌ The AI returned a response that could not be parsed as JSON. "
                    "Please try again. If this keeps happening, the model may be overloaded."
                )
                st.stop()
            except Exception as e:
                error_msg = str(e)
                if "API_KEY" in error_msg.upper() or "api key" in error_msg.lower() or "INVALID_ARGUMENT" in error_msg:
                    st.error("❌ Invalid API key. Please check your GEMINI_API_KEY in the .env file.")
                elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    st.error("❌ API quota exceeded or rate limit hit. Please wait a moment and try again.")
                elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                    st.error("❌ Network error. Please check your internet connection and try again.")
                else:
                    st.error(f"❌ An unexpected error occurred: {error_msg}")
                st.stop()

            # --- Validate the parsed result ---
            try:
                result = validate_result(raw_result)
            except ValueError as e:
                st.error(f"❌ The AI response had an unexpected format: {e}")
                st.stop()

        # ─────────────────────────────────────────────────────────────────────
        # DISPLAY RESULTS
        # ─────────────────────────────────────────────────────────────────────
        st.success("✅ Analysis complete!")
        st.divider()

        # A. Match Score ──────────────────────────────────────────────────────
        score = result["match_score"]
        st.subheader(f"{score_color(score)} Resume Match Score")

        col_score, col_empty = st.columns([1, 2])
        with col_score:
            st.metric(
                label="ATS Match Score",
                value=f"{score} / 100",
            )

        # Progress bar (value must be between 0.0 and 1.0)
        st.progress(score / 100)

        if score >= 75:
            st.info("🟢 Great match! Your resume aligns well with this job description.")
        elif score >= 50:
            st.warning("🟡 Moderate match. A few additions could significantly improve your score.")
        else:
            st.error("🔴 Low match. Consider tailoring your resume specifically for this role.")

        st.divider()

        # B & C. Skills columns ───────────────────────────────────────────────
        col_match, col_miss = st.columns(2)

        with col_match:
            st.subheader("✅ Matching Skills")
            matching = result.get("matching_skills", [])
            if matching:
                for skill in matching:
                    st.markdown(f"- {skill}")
            else:
                st.markdown("_No strong skill matches detected._")

        with col_miss:
            st.subheader("⚠️ Skills to Add")
            missing = result.get("missing_skills", [])
            if missing:
                st.caption(
                    "These keywords appear in the job description but are not "
                    "explicitly mentioned in your resume. If you have these skills, "
                    "consider adding them."
                )
                for skill in missing:
                    st.markdown(f"- {skill}")
            else:
                st.markdown("_No significant gaps detected. Well done!_")

        st.divider()

        # D. Improvement Suggestions ──────────────────────────────────────────
        st.subheader("💡 5 Suggestions to Improve Your Resume")
        suggestions = result.get("suggestions", [])
        for i, suggestion in enumerate(suggestions, start=1):
            st.markdown(f"**{i}.** {suggestion}")

        st.divider()
        st.caption(
            "⚠️ This analysis is generated by AI and is intended as a guide only. "
            "Always review the results critically and seek advice from career professionals."
        )
