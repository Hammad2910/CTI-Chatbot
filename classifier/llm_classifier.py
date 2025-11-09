# classifier/llm_classifier.py

# classifier/llm_classifier.py

import os
from openai import OpenAI

# Initialize client (make sure OPENAI_API_KEY is set in your environment)
client = OpenAI(api_key="")
""
def classify_query(task: str, context: str) -> str:
    """
    LLM-based classifier to detect query type from task + context.
    Classes: memorization, understanding, problem_solving, reasoning_taa, reasoning_ate
    """
    combined_text = f"{task.strip()}\n\n{context.strip()}"

    system_prompt = """You are an expert cyber threat intelligence (CTI) assistant trained to classify CTI benchmark queries.
Your job is to categorize the input query into one of the following categories based on its purpose and reasoning type.

### CTI Task Categories ###
1. **memorization**
   - Pure factual recall or multiple-choice (MCQ-style) tasks.
   - Usually starts with "You are given a multiple-choice question" or contains "options: A, B, C, D".
   - Requires selecting a correct choice without reasoning or analysis.

2. **understanding**
   - Tasks that require conceptual comprehension, mapping, or explanation.
   - Example: Mapping CVEs to CWEs, describing the impact or cause of a vulnerability.
   - The focus is on interpreting or explaining, not computing.

3. **problem_solving**
   - Analytical or computational reasoning tasks that require calculation or metric-based analysis.
   - Example: Calculating CVSS base scores, assessing attack complexity, or scoring vulnerabilities.

4. **reasoning_taa**
   - Threat Actor Attribution reasoning.
   - Involves analyzing a **threat report** or **incident description** to infer the responsible actor.
   - Mentions of campaigns, groups, RATs, or placeholders like [PLACEHOLDER] are strong indicators.

5. **reasoning_ate**
   - Attack Technique Extraction reasoning.
   - Involves identifying MITRE ATT&CK techniques or mapping behaviors to MITRE technique IDs.
   - Often uses terms like “MITRE Enterprise techniques,” “Tactic,” “Technique ID,” “T####,” etc.

### Output Format ###
Return **only one category label** from this exact list:
memorization, understanding, problem_solving, reasoning_taa, reasoning_ate
No extra text, explanations, or formatting.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use gpt-4o-mini or gpt-4-turbo for best cost-speed balance
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_text}
        ],
        temperature=0,
        max_tokens=10
    )

    label = response.choices[0].message.content.strip().lower()

    valid_labels = {
        "memorization",
        "understanding",
        "problem_solving",
        "reasoning_taa",
        "reasoning_ate",
    }

    if label not in valid_labels:
        label = "understanding"  # default fallback

    return label
