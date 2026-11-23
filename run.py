# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)

from app.rag.vectorstore import rerank_documents

documents = rerank_documents(query="Nova core company?", top_n=3)
page_content = [doc.page_content for doc in documents]
for i in range(len(page_content)):
    print("\n" + "--" * 10 + f"Document-{i+1}" + "--" * 10)
    print(page_content[i])
