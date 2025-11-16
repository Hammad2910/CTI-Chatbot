import os
import json
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()  # loads variables from .env

api_key = os.getenv("OPENAI_API_KEY")


class MemorizationPipeline:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vector_db_path = os.path.join(base_dir, "../vector_dbs/memorization_vdb")
        self.chunks_path = os.path.join(self.vector_db_path, "chunks.json")


        # Load vector DB and chunks
        self.vector_db = self._load_vector_db()
        self.chunks_data = self._load_chunks()

        # Initialize OpenAI client
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

    def _retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve the most relevant context"""
        results = self.vector_db.similarity_search(query, k=k)
        context = "\n\n".join([
            f"Source: {res.metadata.get('url', 'Unknown')}\nContent: {res.page_content}"
            for res in results
        ])
        return context

    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using the CTI-specific reasoning prompt"""
        prompt = f"""
    You are an advanced Cyber Threat Intelligence (CTI) assistant trained on CTI benchmark data.
    The user query is in *multiple-choice (MCQ)* format. Your job is to select the correct option
    STRICTLY based on the provided context.

    TASK:
    - Parse the MCQ question and its answer options from the QUERY.
    - Use the CONTEXT to determine which option (A, B, C, or D) is factually supported.
    - If the CONTEXT does not support any option, respond that the answer cannot be derived.

    CONTEXT:
    {context}

    QUERY:
    {query}

    RESPONSE REQUIREMENTS:
    - Provide SHORT structured reasoning based strictly on the context.
    - Clearly cite which parts of the context support or contradict each option.
    - Avoid assumptions or outside knowledge.
    - End with: "Final Answer: <option letter>"

    If no option is supported, end with:
    "Final Answer: Cannot be determined from context"

    Final Answer:
    """
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )

        return response.choices[0].message.content.strip()

    def run(self, query: str) -> str:
        """Main function: retrieve context and generate answer"""
        context = self._retrieve_context(query)
        answer = self._generate_response(query, context)
        return answer


_pipeline_instance = MemorizationPipeline()

def run(query: str) -> str:
    """Allow calling run() directly as module function"""
    return _pipeline_instance.run(query)

