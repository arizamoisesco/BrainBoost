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
model = "meta-llama/llama-3.1-8b-instruct:free"
llm = ChatOpenRouter(model_name=model)


def setup_rag_agent():
    pass

def setup_dialog_agent():
    pass


question_user = input("Ingresa la duda que tengas: ")

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

#answer = llm.invoke(question_user)
#Llamamos al LLM le entregamos el prompt y lo formateamos para que sea recibido como un string
answer = llm.invoke(query.format(question_user=question_user))

#Crear el agente RAG
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

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

#Se crea una herramienta de busqueda básada en el RAG construido
def rag_search(query):
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
    return response['result']

search_tool = Tool(
    name="Busqueda en PDF",
    func=rag_search,
    description="Útil para buscar en mis libros de Python información relevante para clases y contenido"
)

#Se define el prompt que va a manejar el agente
class AgentPromptTemplate(PromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        print("Debug - kwargs:", kwargs)
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservación: {observation}\nPensamiento: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class AgentOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Respuesta Final:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Respuesta Final:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Acción: (.*?)[\n]*Entrada de Acción: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
# Crear el agente
agent_prompt = AgentPromptTemplate(
    template="""
    Responde la siguiente pregunta usando las herramientas disponibles:

    {tools}

    Pregunta: {input}
    {agent_scratchpad}

    Pensamiento: Analicemos paso a paso cómo responder esta pregunta.
    Acción: La acción que tomaré (debe ser una de [{tool_names}])
    Entrada de Acción: La entrada para la acción
    Observación: El resultado de la acción
    ... (este proceso de Pensamiento/Acción/Entrada de Acción/Observación puede repetirse)
    Pensamiento: Ahora puedo dar una respuesta final
    Respuesta Final: La respuesta final a la pregunta""",
        input_variables=["input", "intermediate_steps"],
        tools=[search_tool]
)

#llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
llm_chain = agent_prompt | llm

#create_react_agent Revisar como se crea el agente para ver si ahí esta el fallo
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=AgentOutputParser(),
    stop=["\nObservación:"],
    allowed_tools=[search_tool.name]
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=[search_tool], verbose=True)

#Uso del agente
response = agent_executor.run(question_user)

#print(answer.content)
print(response)