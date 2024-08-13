from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


llm = Ollama(model="llama3.1")
question_user = input("Ingresa la duda que tengas: ")


#Dividimos el documento que es de un gran tamaño en chunks para usarlos más fácil 
def load_pdf_in_chunks(pdf_path, chunk_size=10):
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load_and_split()

    for i in range(0, len(pages), chunk_size):
        yield pages[i:i + chunk_size]


pdf_path = "books/Python_Crash_Course.pdf"

for chunk in load_pdf_in_chunks(pdf_path):
    #Procesar cada chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    docs = text_splitter.split_documents(chunk)
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents = docs,
        embedding = embed_model,
        persist_directory = "chroma_db_dir",
        collection_name = "python_books_data"
    )

print("Almacenando los datos en la base de datos vectorial COMPLETADO")

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

custom_prompt_template = """
Eres un profesor enfocado en enseñar programación

Usa la siguiente información para responder a la pregunta del usuario. Basate en la información proporcionada en el contexto. Aprovecha toda la información proporcionada a continuación:

Contexto: {context}
Pregunta: {question}

Respoden toda la información en español.

Solo devuelve información util que debe proporcionar:

Enunciado de explicación
Ejemplos con código para explicarle a los estudiantes
5 Ejercicios que puedan realizar los estudiantes para poner en práctica lo aprendido con el tema solicitado

Respuesta útil:

"""

prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])

def retriveval_rag(prompt):

    qa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt":prompt})


    response = qa.invoke({"query": question_user})

    return response['result']

