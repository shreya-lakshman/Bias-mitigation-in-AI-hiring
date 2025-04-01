import os
import json
import PyPDF2
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# =======================================
# Configuration
# =======================================
JSON_FILE = "resumes_with_demographics.json"
OUTPUT_SUMMARY_FILE = "resume_summaries.json"
SUMMARY_SENTENCE_COUNT = 5

def get_resume_text(candidate):
    """
    Given a candidate dictionary, constructs the path to the resume PDF based on the domain and file name,
    extracts the full text from the PDF, and returns it.
    If extraction fails, returns the JSON excerpt.
    """
    domain = candidate.get("domain", "").upper()
    file_name = candidate.get("file_name")
    resume_path = os.path.join("data", domain, file_name)
    
    if not os.path.exists(resume_path):
        print(f"Resume file not found for {file_name} in domain {domain}. Using JSON excerpt.")
        return candidate.get("text_excerpt", "")
    
    try:
        with open(resume_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        if not text.strip():
            text = candidate.get("text_excerpt", "")
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {resume_path}: {e}. Using JSON excerpt.")
        return candidate.get("text_excerpt", "")

def extract_education(text):
    """
    Uses a regular expression to extract the Education section from the resume text.
    This regex looks for 'Education' (or 'EDUCATION') followed by optional punctuation
    and then captures text until a double newline or the end of the text.
    """
    pattern = re.compile(r"Education\s*[:\-]?\s*(.*?)(?=\n\s*\n|$)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        edu_section = match.group(1).strip()
        # If the captured section is very short, it might not be useful.
        if len(edu_section) < 30:
            return ""
        return edu_section
    return ""

def summarize_text(text, sentence_count=SUMMARY_SENTENCE_COUNT):
    """
    Uses Sumy's TextRankSummarizer to generate a summary consisting of a given number of sentences.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    summary = " ".join(str(sentence) for sentence in summary_sentences)
    return summary

def main():
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except Exception as e:
        print("Error reading JSON file:", e)
        return

    summaries = []  

    for candidate in candidates:
        print(f"Processing candidate: {candidate.get('file_name')}")
        full_text = get_resume_text(candidate)
        if not full_text:
            print(f"No text available for candidate {candidate.get('file_name')}. Skipping.")
            continue
        
        # Generate a summary using TextRank
        summary = summarize_text(full_text, sentence_count=SUMMARY_SENTENCE_COUNT)

        education_info = extract_education(full_text)

        combined_summary = summary
        if education_info:
            combined_summary += "\nEducation: " + education_info
        
        candidate_summary = {
            "file_name": candidate.get("file_name"),
            "domain": candidate.get("domain"),
            "gender": candidate.get("gender"),
            "ethnicity": candidate.get("ethnicity"),
            "summary": combined_summary
        }
        summaries.append(candidate_summary)
    
    try:
        with open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=4)
        print(f"Summaries saved to {OUTPUT_SUMMARY_FILE}")
    except Exception as e:
        print("Error saving summaries:", e)

if __name__ == "__main__":
    main()
