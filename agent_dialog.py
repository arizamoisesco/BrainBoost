from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

import zmq

# Configurar ZMQ
context = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.connect("tcp://localhost:5556")

sender_rag = context.socket(zmq.PUSH)
sender_rag.bind("tcp://*:5557")

llm = Ollama(model="llama3.1")

question_user = input("Ingresa la pregunta que deseas")

#Definición del prompt de bienvenida
custom_prompt_template ='''
    Eres un asistente que busca tener una tratato cordial y amable con el usuario, requiero que respondas de manera clara y concisa las busquedas que te proporcione.

    Usa el siguiente formato:

    Envia la pregunta que nos de el usuario {question_user}

    Debes siempre despedirte del usuario y validar si tiene alguna otra pregunta.

    Ten en cuenta que todas las respuestas deben ser en español. No inventes nada en caso de no sabe la respuesta puedes mencionarlo.

    '''
def message_dialog(question_user):

    #Problema resuelto
    query = PromptTemplate(template=custom_prompt_template, input_variables=[question_user])

    #Llamamos al LLM le entregamos el prompt y lo formateamos para que sea recibido como un string
    answer = llm.invoke(query.format(question_user=question_user))

    return answer

dialog_tool = Tool(
    name="Herramienta para dar el mensaje de bienvenida",
    func=message_dialog,
    description="Con esta herramienta le doy el mensaje que empieza a buscar en el llm"
)

agent_executor = initialize_agent(
    [dialog_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

#4. Ejecutamos el agente
query_result = agent_executor.run(question_user)

sender_rag.send_string(query_result)

#print(query_result)
