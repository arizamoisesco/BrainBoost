from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

llm = Ollama(model="llama3.1")

#Creación del prompt a manejar

format_prompt_template = """
    Eres un asistente que se encarga de formatear, resumir y organizar la respuesta dada por el llm.

    Respuesta original: {response}

    Formatea la respuesta en formato markdown, resúmela y organiza la información de manera clara y concisa.
    Asegúrate de que la respuesta esté en español y sea fácil de entender.

    Respuesta formateada y resumida:
"""

format_prompt = PromptTemplate(template=format_prompt_template, input_variables=['response'])

#Crear la herramienta de formateo
# 1. Función que se usará como herramienta

def format_query(result):
    format_answer = llm.invoke(format_prompt.format(response=result))
    return format_answer

# 2. Definimos la función anterior como una herramienta

format_tool = Tool(
    name="Formateador de resultado",
    func=format_query,
    description="Herramienta para formatear los resultados conseguidos por el LLM"
)


#3 Inicializamos el agente

"""
    agent_executor = initialize_agent(
    [format_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
"""

agent_executor = initialize_agent(
    [format_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
"""
    agent_executor = AgentExecutor(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools = format_tool,
    verbose=True,
    max_iterations=1,
    handle_parsing_errors=True
)

"""


query_user = input("Ingrese su consulta: ")

#4. Ejecutamos el agente
query_result = agent_executor.run(query_user)
print("Resultado del agente:")
print(query_result)



#Creación del agente de organización