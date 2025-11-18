import argparse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


SPEECH_FILE = "speech.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore():
    loader = TextLoader(SPEECH_FILE, encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    return vectordb


def make_ollama_llm():
    return OllamaLLM(model="mistral", temperature=0.1)


def build_chain(vectordb):
    retriever = vectordb.as_retriever()

    template = """Use the context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=make_ollama_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


def main():
    vectordb = build_vectorstore()
    qa = build_chain(vectordb)

    print("\nReady! Ask your question.\n(type 'exit' to quit)")

    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() == "exit":
            break

        res = qa.invoke({"query": q})
        print("\nAnswer:\n", res["result"])
        print("\nSources:")
        for i, d in enumerate(res["source_documents"], 1):
            print(f"[{i}] {d.page_content[:200]}...")


if __name__ == "__main__":
    main()
