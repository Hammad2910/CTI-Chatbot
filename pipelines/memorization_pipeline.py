import os
import json
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



class MemorizationPipeline:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vector_db_path = os.path.join(base_dir, "../vector_dbs/memorization_vdb")
        self.chunks_path = os.path.join(base_dir, "../vector_dbs/chunks.json")

        # Load vector DB and chunks
        self.vector_db = self._load_vector_db()
        self.chunks_data = self._load_chunks()

        # Initialize OpenAI client
        self.client = OpenAI(api_key="")

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
Analyze the query with a deep technical understanding of vulnerabilities, threat actors, malware, and exploits. 
Use the provided context strictly for factual reasoning.

TASK:
- Interpret the query and infer key entities (CVE, CWE, Threat Actor, TTP, or Malware name).
- Identify relationships or mappings (e.g., CVE → CWE, Threat Actor → Malware, etc.) from the context.
- Base your reasoning on the retrieved information. 
- If context does not support the query, clearly state that the answer cannot be derived.

CONTEXT:
{context}

QUERY:
{query}

RESPONSE REQUIREMENTS:
- Provide a concise, technical answer.
- Avoid assumptions or generic responses.
- Use structured reasoning based only on context.
- End your response with a single clear factual conclusion.

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


# ✅ Standalone test (optional)
if __name__ == "__main__":
    pipeline = MemorizationPipeline()
    test_query = (
        "Analyze this CVE and map it to the appropriate CWE. "
        "CVE Description: In the Linux kernel through 6.7.1, "
        "there is a use-after-free in cec_queue_msg_fh, related to drivers/media/cec/core/cec-adap.c."
    )
    print("\n🔍 Query:", test_query)
    print("\n✅ Final Answer:\n", pipeline.run(test_query))
