import os
import gradio as gr
from pypdf import PdfReader
from openai import OpenAI

# -------- INIT OPENAI --------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- TOOL 1: EXTRACT PDF --------
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() or ""
        except:
            pass
    return text


# -------- TOOL 2: SKILL EXTRACTION --------
def extract_skills(text):
    prompt = f"""
Extract skills from the resume in JSON format:

Return:
{{
  "technical_skills": [],
  "soft_skills": [],
  "tools": []
}}

Resume:
{text}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# -------- TOOL 3: ATS SCORE --------
def ats_score(resume, job_desc):
    prompt = f"""
Compare resume with job description.

Return:
- Match percentage (0–100)
- Missing skills
- Suggestions

Resume:
{resume}

Job Description:
{job_desc}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# -------- TOOL 4: REWRITE --------
def rewrite_resume(text):
    prompt = f"""
Improve this resume:
- Strong action verbs
- Better bullet points
- Professional tone

Resume:
{text}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# -------- TOOL 5: CHAT --------
def chat_resume(question, resume_text):
    prompt = f"""
Answer based ONLY on the resume.

Resume:
{resume_text}

Question:
{question}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# -------- MAIN AGENT --------
def run_agent(file, job_desc, question):
    if file is None:
        return "Upload a PDF", "", "", ""

    # IMPORTANT: file handling fix
    resume_text = extract_text(file.name)

    skills = extract_skills(resume_text)
    ats = ats_score(resume_text, job_desc)
    improved = rewrite_resume(resume_text)

    chat = ""
    if question:
        chat = chat_resume(question, resume_text)

    return skills, ats, improved, chat


# -------- UI --------
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Resume Agent (ATS + Skills + Rewrite + Chat)")

    file = gr.File(label="Upload Resume (PDF)")
    job_desc = gr.Textbox(label="Job Description")
    question = gr.Textbox(label="Ask about resume (optional)")

    skills_out = gr.Textbox(label="Extracted Skills (JSON)", lines=10)
    ats_out = gr.Textbox(label="ATS Score", lines=10)
    rewrite_out = gr.Textbox(label="Improved Resume", lines=15)
    chat_out = gr.Textbox(label="Chat Answer", lines=5)

    btn = gr.Button("Run Agent")

    btn.click(
        run_agent,
        inputs=[file, job_desc, question],
        outputs=[skills_out, ats_out, rewrite_out, chat_out]
    )

# -------- APP RUNNER FIX --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)