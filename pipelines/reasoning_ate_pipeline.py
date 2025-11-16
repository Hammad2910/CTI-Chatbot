# pipelines/understanding_pipeline.py
# pipelines/reasoning_ate_pipeline.py
import os
import json
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()  # loads variables from .env

api_key = os.getenv("OPENAI_API_KEY")




class ReasoningATEPipeline:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vector_db_path = os.path.join(base_dir, "../vector_dbs/memorization_vdb")
        self.chunks_path = os.path.join(self.vector_db_path, "chunks.json")

        # Load vector DB and chunks
        self.vector_db = self._load_vector_db()
        self.chunks_data = self._load_chunks()

        # Initialize OpenAI client (reads key from OPENAI_API_KEY env var)
        self.client = OpenAI(api_key=api_key)

    def _load_vector_db(self):
        """Load FAISS database"""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(
            self.vector_db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    def _load_chunks(self):
        """Load text chunks metadata"""
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _retrieve_context(self, description: str, k: int = 5) -> str:
        """Retrieve the most relevant context for the description"""
        results = self.vector_db.similarity_search(description, k=k)
        context = "\n\n".join([
            f"Source: {res.metadata.get('url', 'Unknown')}\nContent: {res.page_content}"
            for res in results
        ])
        return context

    def _build_prompt(self, similar_context: str, description: str) -> str:
        """Build the exact prompt you provided, filling placeholders."""
        prompt = f"""Extract all MITRE attack patterns from the following text and map them to their corresponding MITRE technique IDs. Provide reasoning for each identification. Ensure the final line contains only the IDs for the main techniques, separated by commas, excluding any subtechnique IDs.

Relevant context:
{similar_context}

Text to analyze:
{description}

Provide your response in the following format:
reasoning: [your reasoning here]
answer: [comma-separated technique IDs]"""
        return prompt

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI and return the assistant output (raw)."""
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()

    def run(self, description: str) -> str:
        """
        
        Main entry:
        - description: the user-provided text to analyze
        Returns the model's output (reasoning + answer lines).
        """
        similar_context = self._retrieve_context(description, k=5)
        prompt = self._build_prompt(similar_context=similar_context, description=description)
        return self._call_openai(prompt)


# module-level instance + convenience function (same call pattern as memorization pipeline)
_pipeline_instance = ReasoningATEPipeline()

def run(description: str) -> str:
    return _pipeline_instance.run(description)

