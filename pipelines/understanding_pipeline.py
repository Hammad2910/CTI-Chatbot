import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnderstandingPipeline:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Paths for knowledge base (using same as memorization)
        self.kb_vector_db_path = os.path.join(base_dir, "../vector_dbs/memorization_vdb")
        self.kb_chunks_path = os.path.join(self.kb_vector_db_path, "chunks.json")
        
        # Paths for CWE database
        self.cwe_vector_db_path = os.path.join(base_dir, "../vector_dbs/understanding_vdbs/faiss_cwe")
        self.cwe_chunks_path = os.path.join(self.cwe_vector_db_path, "chunks_cwe.json")

        # Initialize OpenAI client first
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)

        # Load vector DBs with error handling
        try:
            self.kb_vector_db = self._load_vector_db(self.kb_vector_db_path)
            logger.info("Knowledge base vector DB loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load knowledge base vector DB: {e}")
            self.kb_vector_db = None

        try:
            self.cwe_vector_db = self._load_vector_db(self.cwe_vector_db_path)
            logger.info("CWE vector DB loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CWE vector DB: {e}")
            self.cwe_vector_db = None
        
        # Load chunks metadata
        try:
            self.kb_chunks_data = self._load_chunks(self.kb_chunks_path)
        except Exception as e:
            logger.error(f"Failed to load KB chunks: {e}")
            self.kb_chunks_data = []

        try:
            self.cwe_chunks_data = self._load_chunks(self.cwe_chunks_path)
        except Exception as e:
            logger.error(f"Failed to load CWE chunks: {e}")
            self.cwe_chunks_data = []

    def _load_vector_db(self, db_path: str):
        """Load FAISS database with enhanced error handling"""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Vector DB path does not exist: {db_path}")
        
        # Check for required FAISS files
        required_files = ['index.faiss', 'index.pkl']
        for file in required_files:
            file_path = os.path.join(db_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required FAISS file missing: {file_path}")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        try:
            vector_db = FAISS.load_local(
                db_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_db
        except Exception as e:
            logger.error(f"Error loading FAISS database from {db_path}: {e}")
            raise

    def _load_chunks(self, chunks_path: str):
        """Load text chunks metadata"""
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file does not exist: {chunks_path}")
            
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _retrieve_context(self, query: str, k: int = 3) -> tuple[str, str]:
        """Retrieve context from both KB and CWE databases"""
        kb_context = "No knowledge base context available."
        cwe_context = "No CWE context available."

        # Retrieve from knowledge base (RCM dataset)
        if self.kb_vector_db:
            try:
                kb_results = self.kb_vector_db.similarity_search(query, k=k)
                kb_context = "\n\n".join([
                    f"Source: {res.metadata.get('url', 'Unknown')}\nContent: {res.page_content}"
                    for res in kb_results
                ])
            except Exception as e:
                logger.error(f"Error retrieving KB context: {e}")
                kb_context = "Error retrieving knowledge base context."

        # Retrieve from CWE database
        if self.cwe_vector_db:
            try:
                cwe_results = self.cwe_vector_db.similarity_search(query, k=k)
                cwe_context = "\n\n".join([
                    f"CWE-ID: {res.metadata.get('cwe_id', 'Unknown')}\n"
                    f"Name: {res.metadata.get('name', 'Unknown')}\n"
                    f"Description: {res.page_content}"
                    for res in cwe_results
                ])
            except Exception as e:
                logger.error(f"Error retrieving CWE context: {e}")
                cwe_context = "Error retrieving CWE context."

        return kb_context, cwe_context

    def _generate_response(self, query: str, kb_context: str, cwe_context: str) -> str:
        """Generate CWE mapping response using OpenAI"""
        prompt = f"""
You are a cybersecurity analyst specializing in vulnerability classification and CWE mapping.

CONTEXT FROM RCM KNOWLEDGE BASE:
{kb_context}

CONTEXT FROM CWE DATABASE:
{cwe_context}

VULNERABILITY DESCRIPTION:
{query}

TASK:
Analyze the vulnerability description and map it to the most relevant CWE category based on the provided contexts.

RESPONSE REQUIREMENTS:
1. Provide the exact CWE-ID (e.g., "CWE-284") from the CWE context that best matches the vulnerability
2. Provide the exact CWE name from the CWE context
3. Give a concise justification explaining how the vulnerability aligns with the CWE description
4. Reference specific details from both RCM knowledge base and CWE contexts
5. If no clear match exists in the provided CWE contexts, respond with "CWE-UNKNOWN"

RESPONSE FORMAT (JSON):
{{
    "predicted_cwe": "CWE-XXX",
    "cwe_name": "Exact Name from CWE Context", 
    "justification": "Brief technical explanation referencing specific context details..."
}}

IMPORTANT:
- Only use CWE IDs that appear in the provided CWE context
- Be precise in matching vulnerability characteristics to CWE descriptions
- Focus on access control, hardware security, or configuration issues as relevant
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return json.dumps({
                "predicted_cwe": "ERROR",
                "cwe_name": "API Error",
                "justification": f"Failed to generate response: {str(e)}"
            })

    def run(self, query: str) -> str:
        """Main function: retrieve context and generate CWE mapping"""
        if not self.kb_vector_db and not self.cwe_vector_db:
            return json.dumps({
                "predicted_cwe": "ERROR",
                "cwe_name": "Database Error",
                "justification": "No vector databases are available. Please check the initialization."
            })
            
        kb_context, cwe_context = self._retrieve_context(query)
        answer = self._generate_response(query, kb_context, cwe_context)
        return answer


# Singleton instance for module-level usage with error handling
try:
    _pipeline_instance = UnderstandingPipeline()
    logger.info("UnderstandingPipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize UnderstandingPipeline: {e}")
    _pipeline_instance = None

def run(query: str) -> str:
    """Allow calling run() directly as module function"""
    if _pipeline_instance is None:
        return json.dumps({
            "predicted_cwe": "ERROR", 
            "cwe_name": "Initialization Error",
            "justification": "Pipeline failed to initialize. Check logs for details."
        })
    return _pipeline_instance.run(query)