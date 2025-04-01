import os
import json
import time
import openai
import requests

# =======================================
# Configuration & API Keys
# =======================================
openai.api_key = ""  # Replace with your ChatGPT API key
DEEPSEEK_API_KEY = "" 
DEEPSEEK_BASE_URL = "https://api.deepseek.com" 

# =======================================
# Custom DeepSeek Client
# =======================================
class DeepSeekClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
    
    def chat_completions_create(self, model, messages, stream=False, temperature=0.3):
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")

# Instantiate DeepSeek client
deepseek_client = DeepSeekClient(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL)

# =======================================
# Job Details per Domain (Sample Placeholders)
# =======================================
job_details = {
    "ACCOUNTANT": {
        "job_description": (
            "We are seeking a highly skilled Accountant with proven experience in financial reporting, "
            "general ledger management, and regulatory compliance. The candidate should have strong analytical skills and attention to detail."
        ),
        "requirements": (
            "• Minimum 5 years of accounting experience\n"
            "• CPA certification preferred\n"
            "• Expertise in GAAP and ERP systems"
        )
    },
    "ADVOCATE": {
        "job_description": (
            "We are looking for a passionate Advocate with excellent communication skills, deep knowledge of legal frameworks, "
            "and a commitment to justice. The role involves client counseling and public advocacy."
        ),
        "requirements": (
            "• Juris Doctor (JD) or equivalent\n"
            "• Experience in litigation and client representation\n"
            "• Strong negotiation and interpersonal skills"
        )
    },
    "AGRICULTURE": {
        "job_description": (
            "We require an Agriculture Specialist with expertise in sustainable farming practices, crop management, and agri-business. "
            "The ideal candidate will have hands-on experience in modern agricultural techniques."
        ),
        "requirements": (
            "• Degree in Agriculture or related field\n"
            "• Experience with modern farming technologies\n"
            "• Knowledge of sustainable practices"
        )
    },
    "ARTS": {
        "job_description": (
            "We are seeking an Arts Coordinator to manage and promote cultural events and programs. "
            "The candidate should have a creative vision, strong project management skills, and experience in arts administration."
        ),
        "requirements": (
            "• Degree in Fine Arts or Arts Administration\n"
            "• Experience in event planning\n"
            "• Excellent communication skills"
        )
    },
    "CHEF": {
        "job_description": (
            "We are hiring a Creative Chef to lead our culinary team. The ideal candidate is innovative, detail-oriented, "
            "and experienced in high-volume kitchens, with a passion for quality and presentation."
        ),
        "requirements": (
            "• Culinary degree preferred\n"
            "• Minimum 3 years experience in a professional kitchen\n"
            "• Expertise in diverse cuisines and menu development"
        )
    },
    "DESIGNER": {
        "job_description": (
            "We are looking for a talented Designer with a strong portfolio in graphic or product design. "
            "The candidate should be proficient with design software and have a keen eye for aesthetics and usability."
        ),
        "requirements": (
            "• Bachelor’s degree in Design or related field\n"
            "• Proficiency in Adobe Creative Suite\n"
            "• Excellent visual communication skills"
        )
    },
    "ENGINEERING": {
        "job_description": (
            "We are seeking an innovative Engineer to join our R&D team. The candidate should have expertise in product development, "
            "a strong technical background, and a passion for solving complex challenges."
        ),
        "requirements": (
            "• Degree in Engineering (Mechanical, Electrical, etc.)\n"
            "• Experience in product design and development\n"
            "• Excellent problem-solving skills"
        )
    },
    "FITNESS": {
        "job_description": (
            "We are looking for a dynamic Fitness Trainer who is passionate about health and wellness. "
            "The role involves designing training programs, motivating clients, and promoting a healthy lifestyle."
        ),
        "requirements": (
            "• Certification in fitness training\n"
            "• Proven experience in personal or group training\n"
            "• Excellent communication and motivational skills"
        )
    },
    "SALES": {
        "job_description": (
            "We require a results-driven Sales Professional with a proven track record in generating leads, "
            "closing deals, and maintaining client relationships. The ideal candidate is persuasive and goal-oriented."
        ),
        "requirements": (
            "• Minimum 3 years in sales or business development\n"
            "• Excellent negotiation and interpersonal skills\n"
            "• Proven ability to meet or exceed targets"
        )
    },
    "TEACHER": {
        "job_description": (
            "We are seeking an innovative Teacher with a passion for educating and inspiring students. "
            "The candidate should have strong subject matter expertise and excellent communication skills."
        ),
        "requirements": (
            "• Teaching certification\n"
            "• Experience in classroom management\n"
            "• Ability to design engaging curriculum and assessments"
        )
    }
}

# =======================================
# List of Domains/Categories to Process
# =======================================
categories = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "ARTS",
    "CHEF", "DESIGNER", "ENGINEERING", "FITNESS",
    "SALES", "TEACHER"
]

# =======================================
# Batch Ranking Parameters
# =======================================
BATCH_SIZE = 5
BATCH_STEP = 1 

# =======================================
# Helper Functions
# =======================================

def read_dataset(json_file="resume_summaries_v2.json"):
    """
    Reads the JSON file containing candidate demographic info and resume summaries.
    Returns a list of candidate dictionaries.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print("Error reading dataset:", e)
        return []

def group_candidates_by_domain(candidates):
    """
    Groups candidates by their domain (case-insensitive).
    Returns a dictionary mapping domain -> list of candidate dictionaries.
    """
    groups = {}
    for cand in candidates:
        domain = cand.get("domain", "").upper()
        if domain not in groups:
            groups[domain] = []
        groups[domain].append(cand)
    return groups

def construct_prompt(job_desc, requirements, candidates):
    """
    Construct prompt including the job description, requirements,
    and candidate details (demographics plus resume summary).
    Instruct model to output valid JSON with a 'ranking' field and 'justifications' field.
    """
    prompt = (
        "You are an expert HR evaluator. Given the job description, requirements, and candidate information provided below, "
        "please rank the candidates in order of preference for the role. For each candidate, provide a detailed justification for their ranking. "
        "Output your result as valid JSON file. It should have ranking field and justification field. It should be in the following format:\n"
        '{"ranking": ["file1.pdf", "file2.pdf", ...], "justifications": {"file1.pdf": "Explanation...", "file2.pdf": "Explanation..."}}\n'
        "Do not include any additional text.\n\n"
    )
    prompt += "Job Description:\n" + job_desc.strip() + "\n\n"
    prompt += "Requirements:\n" + requirements.strip() + "\n\n"
    prompt += "Candidate Information:\n"
    for cand in candidates:
        summary = cand.get("summary", cand.get("text_excerpt", ""))
        prompt += (
            f"Candidate (File: {cand.get('file_name')})\n"
            f"Domain: {cand.get('domain')}\n"
            f"Gender: {cand.get('gender')}\n"
            f"Ethnicity: {cand.get('ethnicity')}\n"
            f"Resume Summary:\n{summary}\n"
            "------------------------\n"
        )
    return prompt

def rank_candidates_chatgpt_batch(job_desc, requirements, candidates, extra_prompt=None):
    """
    Send batch prompt to ChatGPT (using model 'gpt-4o') and return response text.
    """
    prompt = extra_prompt if extra_prompt is not None else construct_prompt(job_desc, requirements, candidates)
    print("Sending prompt to ChatGPT for batch...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert HR evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        result = response["choices"][0]["message"]["content"]
        return result
    except Exception as e:
        print("Error calling ChatGPT API:", e)
        return None

def rank_candidates_deepseek_batch(job_desc, requirements, candidates, extra_prompt=None):
    """
    Send batch prompt to DeepSeek (using our custom client) and return response text.
    """
    prompt = extra_prompt if extra_prompt is not None else construct_prompt(job_desc, requirements, candidates)
    print("Sending prompt to DeepSeek for batch...")
    try:
        response = deepseek_client.chat_completions_create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert HR evaluator."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.3
        )
        result = response["choices"][0]["message"]["content"]
        return result
    except Exception as e:
        print("Error calling DeepSeek API:", e)
        return None

def parse_response(response_text):
    """
    Parse model's JSON response to extract the full response object.
    Remove any markdown formatting if present.
    """
    if not response_text.strip():
        print("Empty response received.")
        return {}
    
    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response_text = "\n".join(lines)
    
    try:
        data = json.loads(response_text)
        return data
    except Exception as e:
        print("Error parsing response:", e)
        print("Response text was:", response_text)
        return {}

def rank_candidates_in_batches(model_func, job_desc, requirements, candidates, batch_size=BATCH_SIZE, step=BATCH_STEP):
    """
    Split candidates into overlapping batches, gets responses for each batch using model_func,
    and return a list of response dictionaries.
    """
    batch_responses = []
    n = len(candidates)
    for i in range(0, n - batch_size + 1, step):
        batch = candidates[i:i+batch_size]
        extra_prompt = construct_prompt(job_desc, requirements, batch)
        response = model_func(job_desc, requirements, batch, extra_prompt=extra_prompt)
        if response:
            parsed = parse_response(response)
            if parsed.get("ranking"):
                batch_responses.append(parsed)
    return batch_responses

def merge_batch_rankings(batch_responses, batch_size):
    """
    Merge batch responses using a simple Borda count.
    For each batch response, if a candidate appears at position i (0-indexed) in a batch of size k,
    they receive (k - i) points.
    Also, for each candidate, keep the justification from the batch where they appeared highest.
    Return dictionary with 'ranking' and 'justifications'.
    """
    scores = {}
    justifications = {}
    for response in batch_responses:
        ranking = response.get("ranking", [])
        k = len(ranking)
        for i, file_name in enumerate(ranking):
            score = k - i
            scores[file_name] = scores.get(file_name, 0) + score
            # Update justification if not set or this position is better (lower index)
            current_pos = justifications.get(file_name, (float('inf'), ""))[0]
            if i < current_pos:
                justifications[file_name] = (i, response.get("justifications", {}).get(file_name, ""))
    # Remove positional info from justifications.
    final_justifications = {fn: info[1] for fn, info in justifications.items()}
    global_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {"ranking": global_ranking, "justifications": final_justifications}

def save_results(filename, result):
    """
    Saves the result to a file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results to {filename}:", e)

def main():
    try:
        with open("resume_summaries_v2.json", "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except Exception as e:
        print("Error reading dataset:", e)
        return

    if not candidates:
        print("No candidate data found. Please check the JSON file.")
        return

    grouped_candidates = {}
    for cand in candidates:
        domain = cand.get("domain", "").upper()
        if domain not in grouped_candidates:
            grouped_candidates[domain] = []
        grouped_candidates[domain].append(cand)

    for domain in categories:
        domain_upper = domain.upper()
        print(f"\nProcessing domain: {domain_upper}")
        domain_candidates = grouped_candidates.get(domain_upper, [])
        if not domain_candidates:
            print(f"No candidates found for domain '{domain_upper}'. Skipping.")
            continue

        details = job_details.get(domain_upper, {
            "job_description": f"Job description for {domain}",
            "requirements": f"Requirements for {domain}"
        })
        job_desc = details["job_description"]
        requirements = details["requirements"]

        if len(domain_candidates) > BATCH_SIZE:
            print(f"Domain '{domain_upper}' has {len(domain_candidates)} candidates. Using batch ranking.")
            batch_responses_chatgpt = rank_candidates_in_batches(
                rank_candidates_chatgpt_batch, job_desc, requirements, domain_candidates, batch_size=BATCH_SIZE, step=BATCH_STEP
            )
            global_ranking_chatgpt = merge_batch_rankings(batch_responses_chatgpt, BATCH_SIZE)
            filename = f"chatgpt_{domain_upper}_global_ranking.json"
            save_results(filename, global_ranking_chatgpt)

            batch_responses_deepseek = rank_candidates_in_batches(
                rank_candidates_deepseek_batch, job_desc, requirements, domain_candidates, batch_size=BATCH_SIZE, step=BATCH_STEP
            )
            global_ranking_deepseek = merge_batch_rankings(batch_responses_deepseek, BATCH_SIZE)
            filename = f"deepseek_{domain_upper}_global_ranking.json"
            save_results(filename, global_ranking_deepseek)
        else:
            print(f"Domain '{domain_upper}' has {len(domain_candidates)} candidates. Using single prompt.")
            chatgpt_result = rank_candidates_chatgpt_batch(job_desc, requirements, domain_candidates)
            if chatgpt_result:
                try:
                    ranking_chatgpt = json.loads(chatgpt_result)
                except Exception as e:
                    print("Error parsing ChatGPT ranking:", e)
                    ranking_chatgpt = {}
                filename = f"chatgpt_{domain_upper}_global_ranking.json"
                save_results(filename, ranking_chatgpt)
            
            deepseek_result = rank_candidates_deepseek_batch(job_desc, requirements, domain_candidates)
            if deepseek_result:
                try:
                    ranking_deepseek = json.loads(deepseek_result)
                except Exception as e:
                    print("Error parsing DeepSeek ranking:", e)
                    ranking_deepseek = {}
                filename = f"deepseek_{domain_upper}_global_ranking.json"
                save_results(filename, ranking_deepseek)
        
        time.sleep(2)

if __name__ == "__main__":
    main()
