import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ---------------------------
# Fixed CVSS prompt (internal)
# ---------------------------
BASE_CTI_PROMPT = """
Analyze the following CVE description and calculate the CVSS v3.1 Base Score. Determine the values for each base metric: AV, AC, PR, UI, S, C, I, and A. Summarize each metric's value and provide the final CVSS v3.1 vector string. Valid options for each metric are as follows: - **Attack Vector (AV)**: Network (N), Adjacent (A), Local (L), Physical (P) - **Attack Complexity (AC)**: Low (L), High (H) - **Privileges Required (PR)**: None (N), Low (L), High (H) - **User Interaction (UI)**: None (N), Required (R) - **Scope (S)**: Unchanged (U), Changed (C) - **Confidentiality (C)**: None (N), Low (L), High (H) - **Integrity (I)**: None (N), Low (L), High (H) - **Availability (A)**: None (N), Low (L), High (H) Summarize each metric's value and provide the final CVSS v3.1 vector string. Ensure the final line of your response contains only the CVSS v3 Vector String in the following format: Example format: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H CVE
"""


class ProblemSolvingPipeline:
    """
    Problem solving pipeline (CVSS v3.1 vector prediction) using a RAG pattern:
      - Retrieve similar CVEs from FAISS vector DB
      - Build an enhanced prompt using retrieved examples + fixed base prompt
      - Call LLM (OpenAI) to infer CVSS metrics and produce a vector string
    """

    def __init__(self):
        # Vector DB location (explicit path per your setup)
        self.vdb_path = Path("/home/ubuntu/CTI-Chatbot/vector_dbs/problem_solving_vdb")

        # Files expected inside the vector DB
        self.index_path = self.vdb_path / "vsp_faiss_index.faiss"
        self.metadata_path = self.vdb_path / "vsp_metadata.pickle"
        self.chunks_path = self.vdb_path / "vsp_chunks.pickle"

        # Load resources
        self._ensure_files_exist()
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        print("Loading metadata and chunks...")
        with open(self.metadata_path, "rb") as f:
            self.metadatas = pickle.load(f)
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        # Embedding model
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)

        print("ProblemSolvingPipeline initialized successfully.")

    def _ensure_files_exist(self):
        """Ensure vector DB files exist to fail early with helpful error."""
        missing = []
        for p in (self.index_path, self.metadata_path, self.chunks_path):
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                "The following required files were not found in problem_solving_vdb:\n"
                + "\n".join(missing)
            )

    # -----------------------------
    # Retrieval
    # -----------------------------
    def _retrieve_similar(self, description: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top_k similar CVE chunks and metadata from FAISS.
        Returns list of dicts with keys: metadata, chunk, distance, similarity_score
        """
        emb = self.embedder.encode([description], convert_to_numpy=True)
        # Ensure float32 for faiss
        distances, indices = self.index.search(emb.astype("float32"), top_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append(
                {
                    "metadata": self.metadatas[idx],
                    "chunk": self.chunks[idx],
                    "distance": float(dist),
                    "similarity_score": float(1.0 / (1.0 + float(dist))),
                }
            )
        return results

    # -----------------------------
    # Prompt builder
    # -----------------------------
    def _build_prompt(self, query_description: str, retrieved: List[Dict[str, Any]]) -> str:
        """Build the final prompt that will be sent to the LLM."""
        context_lines = ["Below are similar historical CVEs for reference:\n"]
        for i, ex in enumerate(retrieved):
            meta = ex.get("metadata", {})
            context_lines.append(f"Example {i+1}:")
            context_lines.append(f"Description: {ex.get('chunk','')}")
            context_lines.append(f"CVSS Vector: {meta.get('cvss_vector', 'N/A')}")
            context_lines.append(f"CWE: {meta.get('cwe_id', 'N/A')}")
            context_lines.append(f"Similarity Score: {ex.get('similarity_score', 0):.3f}\n")

        context = "\n".join(context_lines)

        final_prompt = (
            f"{context}\n"
            f"{BASE_CTI_PROMPT}\n\n"
            f"CVE Description:\n{query_description}\n\n"
            f"Provide the summary for each metric and ensure the final line ONLY contains the CVSS v3.1 vector string."
        )
        return final_prompt

    # -----------------------------
    # LLM caller
    # -----------------------------
    def _call_openai(self, prompt: str, model: str = "gpt-4-turbo") -> str:
        """
        Call OpenAI chat completion and return text content.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
        )
        # defensive checks
        if not response.choices or not getattr(response.choices[0], "message", None):
            return ""
        return response.choices[0].message.content.strip()

    # -----------------------------
    # CVSS vector extractor
    # -----------------------------
    def _extract_cvss_vector(self, text: str) -> str:
        """
        Extract CVSS vector string from the LLM output.
        Expected format: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
        Returns the matched string or None.
        """
        if not text:
            return None
        match = re.search(r"CVSS:3\.1(?:/[A-Z]+:[A-Z]){8}", text.replace(" ", ""))
        if match:
            return match.group(0)
        # fallback: more permissive pattern (if slashes/spaces vary)
        match2 = re.search(r"CVSS:3\.1[^\n\r]*", text)
        return match2.group(0).strip() if match2 else None

    # -----------------------------
    # Public run() — only description argument
    # -----------------------------
    def run(self, description: str) -> Dict[str, Any]:
        """
        Execute the full RAG -> LLM pipeline for a given CVE description.

        Returns a dictionary with:
            - query_description
            - similar_examples (list)
            - enhanced_prompt
            - llm_response
            - extracted_vector (or None)
            - model_used
        """
        if not description or not description.strip():
            raise ValueError("description must be a non-empty string")

        # 1) Retrieve
        retrieved = self._retrieve_similar(description, top_k=3)

        # 2) Build prompt
        enhanced_prompt = self._build_prompt(description, retrieved)

        # 3) Call LLM
        try:
            llm_response = self._call_openai(enhanced_prompt, model="gpt-4-turbo")
        except Exception as e:
            llm_response = f"LLM call failed: {str(e)}"

        # 4) Extract CVSS vector
        extracted_vector = self._extract_cvss_vector(llm_response)
        return extracted_vector



# Singleton instance to mirror your other pipelines' usage pattern
_pipeline_instance = ProblemSolvingPipeline()


def run(description: str) -> Dict[str, Any]:
    """
    Module-level convenience wrapper: call run(description) directly.
    """
    return _pipeline_instance.run(description)
