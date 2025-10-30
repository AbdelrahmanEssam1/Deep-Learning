#Main goal create a chatbot with a form of memory for our agent
import os
from typing import TypedDict , List , Union
from langchain_core.messages import HumanMessage ,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START , END , StateGraph

class AgentState(TypedDict):
    messages: List[Union[HumanMessage , AIMessage]]

llm = ChatGoogleGenerativeAI(
    model= "models/gemini-2.5-pro",
    google_api_key = "AIzaSyD6-54qxPYbx7Lu-a0YaZRzztNhQGV0WYY"
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI response: {response.content}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process" , END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter your message: ")

while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke(
        {
            "messages": conversation_history
        }
    )
    conversation_history = result['messages']

    user_input = input("Enter: ")

with open("logging.txt","w") as file:
    file.write("Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    print(f"Conversation ended")

print("Conversation saved to logging.txt")