from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever

def build_chain(pdf_path: str = "RAG MATH.pdf"):
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vs = Chroma(collection_name="my_rag", embedding_function=embeddings, persist_directory="./chroma_db")
    vs.add_documents(chunks)

    # Base retriever (MMR) — used for each expanded query
    base_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)

# LLM that will GENERATE the alternative queries (small & cheap is fine)
    expander_llm = ChatOllama(model="llama3.1:latest", temperature=0)

# Multi-query wrapper: generate ~3 paraphrases, retrieve for each, take unique union
    retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=expander_llm,
    include_original=True,   # also use the user’s exact wording
)

    prompt = ChatPromptTemplate.from_template(
        "Answer using only the CONTEXT. Cite page numbers.\n\nCONTEXT:\n{context}\n\nQ: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(f"[p.{d.metadata.get('page')}] {d.page_content}" for d in docs)

    chain = (
        {"context": (RunnablePassthrough() | retriever | format_docs),
         "question": RunnablePassthrough()}
        | prompt
        | ChatOllama(model="llama3.1:latest")
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    chain = build_chain()
    for chunk in chain.stream("Suslin proved what ?"):
        print(chunk, end="")
    print()
