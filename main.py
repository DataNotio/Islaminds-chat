from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain import ConversationalRetrievalChain, ChatOpenAI, DocArrayInMemorySearch, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain.schema import Document
from PyPDF2 import PdfReader

app = FastAPI()
templates = Jinja2Templates(directory="templates")


def load_pdf_content(file_path, chain_type="stuff", k=4):

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
    

    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa


qa = load_pdf_content("./The-Complete-Hadith.pdf")


chat_history = []

@app.get("/", response_class=HTMLResponse)
async def get_chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request, "chat_history": chat_history})

@app.post("/chat")
async def chat(question: str = Form(...)):
    global chat_history
    if question:
        # Query the chatbot with the user question
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        answer = result["answer"]
    else:
        answer = "Please enter a question."

    return {"question": question, "answer": answer, "chat_history": chat_history}

@app.post("/clear_history")
async def clear_chat_history():
    global chat_history
    chat_history = []
    return {"status": "Chat history cleared"}
