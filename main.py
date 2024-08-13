from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType, create_react_agent, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Optional
from pydantic import BaseModel

import os
import re

#Clase encargada de conectar con la API de open router
class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: Optional[str]
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)


#Inicializar el modelo de lenguaje con el que se va a trabajar
#model = "meta-llama/llama-3.1-8b-instruct:free"
#model = "google/gemma-2-9b-it:free"
#llm = ChatOpenRouter(model_name=model)

llm = Ollama(model="llama3.1")


def setup_rag_agent():
    pass

def setup_dialog_agent(question_user):
    custom_prompt_template ='''
    Eres un asistente que busca tener una tratato cordial y amable con el usuario, requiero que respondas de manera clara y concisa las busquedas que te proporcione.

    Usa el siguiente formato:

    Responde a la siguiente pregunta {question_user}

    Respuesta organizado en formato markdown 

    Debes siempre despedirte del usuario y validar si tiene alguna otra pregunta.

    Ten en cuenta que todas las respuestas deben ser en español. No inventes nada en caso de no sabe la respuesta puedes mencionarlo.

    '''

    #Problema resuelto
    query = PromptTemplate(template=custom_prompt_template, input_variables=[question_user])

    #Llamamos al LLM le entregamos el prompt y lo formateamos para que sea recibido como un string
    answer = llm.invoke(query.format(question_user=question_user))

    return answer


question_user = input("Ingresa la duda que tengas: ")

#answer = setup_dialog_agent(question_user)

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

#Se crea una herramienta de busqueda básada en el RAG construido
def rag_search(query):
    print("Activando la herramienta de busqueda")
    custom_prompt_template = """

    Contexto: {contexto}
    Pregunta: {question}
    Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español
    Respuesta útil:
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    response = qa.invoke({"query": query})
    print("Busqueda completada")

    return response['result']

search_tool = Tool(
    name="Busqueda en PDF",
    func=rag_search,
    description="Útil para buscar en mis libros de Python información relevante para clases y contenido"
)


### Desde aqui hay que empezar a módifica para hacer funcionar el agente RAG

#Lista de herramientas disponibles
tools = [search_tool]

# Nombres de las herramientas
tool_names = [tool.name for tool in tools]

# Prompt de forma de pensamiento que debe tener el RAG
rag_think_prompt = """
    Responde la siguiente pregunta usando las herramientas disponibles

    Pregunta: {input}

    Pensamiento: Analicemos paso a paso cómo responder esta pregunta.
    Acción: La acción que tomaré (debe ser una de las herramientas disponibles)
    Entrada de Acción: La entrada para la acción
    Observación: El resultado de la acción
    ... (este proceso de Pensamiento/Acción/Entrada de Acción/Observación puede repetirse)
    Pensamiento: Ahora puedo dar una respuesta final
    Respuesta Final: La respuesta final a la pregunta

    Herramientas disponibles: {tools}
    Nombre de las herramientas: {tool_names}
    Scratchpad: {agent_scratchpad}
    """

rag_agent_prompt = PromptTemplate(template=rag_think_prompt, input_variables=['input','tools', 'agent_scratchpad', 'tool_names'])

print("Creamos el agente de RAG")
agentRag = create_react_agent(
    llm=llm, 
    tools=tools, 
    prompt=rag_agent_prompt
)
#agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=[search_tool], verbose=True)

#Uso del agente
# Create an agent executor by passing in the agent and tools
print("Creamos el ejecutor del RAG el agente de RAG")
agent_executor = AgentExecutor(
    agent=agentRag, 
    tools=tools, 
    verbose=True, 
    max_iterations=10, 
    handle_parsing_errors=True
)

#response = agent_executor.run(question_user)
print("Activando el agente de RAG")
rag_response = agent_executor.invoke({
    "input" : question_user
})

print(rag_response["output"])
#Creación del agente de organización

format_prompt_template = """
    Eres un asistente que se encarga de formatear, resumir y organizar la respuesta de otro agente.

    Respuesta original: {response}

    Formatea la respuesta en formato markdown, resúmela y organiza la información de manera clara y concisa.
    Asegúrate de que la respuesta esté en español y sea fácil de entender.

    Herramientas disponibles: {tools}
    Nombre de las herramientas: {tool_names}
    Scratchpad: {agent_scratchpad}

    Respuesta formateada y resumida:
"""

format_prompt = PromptTemplate(template=format_prompt_template, input_variables=['response', 'tools', 'agent_scratchpad', 'tool_names'])

# Función para invocar al segundo agente
def format_response(response):
    formatted_response = llm.invoke(format_prompt.format(response=response))
    return formatted_response


#Definimos la herramienta de formateo 

format_tool = Tool(
    name="Formateo y organización de resultados",
    func=format_response,
    description="Sirve para poder organizar las respuestas ofrecidas por los demas agentes"
)

#Lista de herramientas disponibles
tools2 = [format_tool]

# Nombres de las herramientas
tool_names2 = [tool.name for tool in tools2]

#Creamos el agente de formateo
agentFormat = create_react_agent(
    llm=llm, 
    tools=tools2, 
    prompt=format_prompt
)




#Se crea el ejecutos del agente de formateo
agent_format_executor = AgentExecutor(
    agent=agentFormat,
    tools=tools2,
    verbose=True,
    max_iterations=1,
    handle_parsing_errors=True
)


#Activamos el agente de formateo
print("Activando el agente de formateo")

"""
    formatted_response = agent_executor.invoke({
    "response": rag_response
})
"""


# Pasar la respuesta del agente RAG al segundo agente para formatear, resumir y organizar
#formatted_response = format_response(rag_response)

#print(formatted_response)
