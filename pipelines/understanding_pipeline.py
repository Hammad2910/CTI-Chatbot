import os
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
import faiss
import pickle
import numpy as np

load_dotenv()

class CTIUnderstandingPipeline:
    def __init__(self):
        """
        Load FAISS vector DBs and chunks using relative paths.
        Expected structure:
        ../vector_dbs/understanding_vdbs/faiss_kb
        ../vector_dbs/understanding_vdbs/faiss_cwe
        """
        base_dir = Path(__file__).parent.parent / "vector_dbs" / "understanding_vdbs"

        # KB vector DB
        self.kb_path = base_dir / "faiss_kb"
        self.kb_chunks_path = self.kb_path / "chunks_kb.json"

        # CWE vector DB
        self.cwe_path = base_dir / "faiss_cwe"
        self.cwe_chunks_path = self.cwe_path / "chunks_cwe.json"

        # Load FAISS vector DBs
        self.kb_vector_db = self._load_vector_db(self.kb_path)
        self.cwe_vector_db = self._load_vector_db(self.cwe_path)

        # Load chunks
        self.kb_chunks = self._load_chunks(self.kb_chunks_path)
        self.cwe_chunks = self._load_chunks(self.cwe_chunks_path)

        # OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing in environment variables")
        self.client = OpenAIClient(api_key=api_key)

    def _load_vector_db(self, path: Path):
        """
        Load FAISS DB by reconstructing from index and chunks.
        """
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_core.documents import Document
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        try:
            # Load the FAISS index
            index_file = path / "index.faiss"
            if index_file.exists():
                # Load using FAISS native method
                index = faiss.read_index(str(index_file))
            else:
                # Fallback to pickle if .faiss file doesn't exist
                pkl_file = path / "index.pkl"
                with open(pkl_file, "rb") as f:
                    index = pickle.load(f)
            
            # Determine chunks file based on directory name
            if "faiss_kb" in str(path):
                chunks_file = path / "chunks_kb.json"
                is_kb = True
            elif "faiss_cwe" in str(path):
                chunks_file = path / "chunks_cwe.json"
                is_kb = False
            else:
                raise ValueError(f"Unknown FAISS directory: {path}")
            
            # Load the chunks JSON to reconstruct docstore
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # Reconstruct docstore and index mapping
            docstore = InMemoryDocstore({})
            index_to_docstore_id = {}
            
            for i, chunk in enumerate(chunks):
                doc_id = str(i)
                
                # Handle different chunk formats
                if is_kb:
                    # KB chunks are plain strings
                    if isinstance(chunk, str):
                        page_content = chunk
                        metadata = {}
                    else:
                        # Fallback if format changes
                        page_content = str(chunk)
                        metadata = {}
                else:
                    # CWE chunks are objects with cwe_id, name, text, etc.
                    page_content = chunk.get("text", chunk.get("description", ""))
                    metadata = {
                        "cwe_id": chunk.get("cwe_id", ""),
                        "name": chunk.get("name", "")
                    }
                
                # Create Document object
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                docstore._dict[doc_id] = doc
                index_to_docstore_id[i] = doc_id
            
            # Create FAISS object with proper embedding function
            return FAISS(
                embedding_function=embeddings.embed_query,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
        except Exception as e:
            print(f"Error loading FAISS DB at {path}: {e}")
            raise e

    def _load_chunks(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve_kb_context(self, query: str, k: int = 3) -> str:
        results = self.kb_vector_db.similarity_search(query, k=k)
        return "\n\n".join(
            f"Source: {r.metadata.get('url', 'N/A')}\nContent: {r.page_content}" for r in results
        )

    def retrieve_cwe_context(self, query: str, k: int = 3) -> str:
        results = self.cwe_vector_db.similarity_search(query, k=k)
        return "\n\n".join(
            f"CWE: {r.metadata.get('cwe_id', 'N/A')}\nDescription: {r.page_content}" for r in results
        )

    def generate_understanding_answer(
        self, description: str, kb_context: str, cwe_context: str,
        model: str = "gpt-4-turbo"
    ) -> dict:
        """
        Generate structured CWE prediction using OpenAI
        """
        structured_prompt = f"""
You are a cybersecurity analyst specializing in vulnerability classification.

CONTEXT (Knowledge Base):
{kb_context}

CONTEXT (CWE Descriptions):
{cwe_context}

DESCRIPTION:
{description}

TASK:
1. Identify the most relevant CWE category.
2. Return CWE ID and the CWE name.
3. Provide a short justification using the retrieved context.
4. If uncertain → return "CWE-UNKNOWN".

FORMAT (JSON):
{{
    "predicted_cwe": "<CWE-ID>",
    "cwe_name": "<CWE Name>",
    "justification": "<reason>"
}}
"""
        return self._call_openai(structured_prompt, model)

    def _call_openai(self, prompt: str, model: str) -> dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"predicted_cwe": "CWE-UNKNOWN", "cwe_name": "", "justification": "LLM returned invalid JSON"}


# -----------------------------
# App-ready entrypoint
# -----------------------------
_pipeline_instance = None

def get_pipeline_instance():
    """Lazy initialization of pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = CTIUnderstandingPipeline()
    return _pipeline_instance

def run(description: str) -> dict:
    """
    Entrypoint for Streamlit or other apps
    """
    pipeline = get_pipeline_instance()
    kb_ctx = pipeline.retrieve_kb_context(description)
    cwe_ctx = pipeline.retrieve_cwe_context(description)
    return pipeline.generate_understanding_answer(description, kb_ctx, cwe_ctx)