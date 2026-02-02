import chromadb
import RR

'''
client = chromadb.PersistentClient(path="./chroma_db")
print([c.name for c in client.list_collections()])  # list collections

coll = client.get_collection("my_rag")              # open your collection
print("count:", coll.count())
print(coll.peek())                                  # small sample
print(coll.get(include=["documents","metadatas"], limit=3))
print(coll.get(include=["embeddings"], limit=1))    # raw vector(s)
'''
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from RR import build_chain   # safe: RR won't auto-run now

# example: embed one string
emb = OllamaEmbeddings(model="mxbai-embed-large")
vec = emb.embed_query("LangChain is a framework for LLM apps.")
print("dim:", len(vec), vec)

# or use the chain
'''
chain = build_chain()
print(chain.invoke("Suslin proved what?"))
'''