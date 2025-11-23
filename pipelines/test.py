import pickle
from pathlib import Path

pkl_file = Path("/home/ubuntu/CTI-Chatbot/vector_dbs/understanding_vdbs/faiss_kb/index.pkl")

with open(pkl_file, "rb") as f:
    content = pickle.load(f)
    
print(f"Type: {type(content)}")
# print(f"Content: {content}")
# if hasattr(content, '__dict__'):
#     print(f"Attributes: {dir(content)}")