from RAG import *

# Step 1: Load, embed, and store documents

print("File name should be .txt or .pdf or .docx")
document = ("Enter file name: ")
vectorstore = initialize_faiss_store()
embed_and_store_document(document, vectorstore=vectorstore)

while True:
    query = input("Enter query: ")
    answer = generate_answer(query)
    print("Response: ", answer)
