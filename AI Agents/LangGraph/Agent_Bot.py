from typing import TypedDict , List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START , END , StateGraph

# GOOGLE_API_KEY= "AIzaSyD6-54qxPYbx7Lu-a0YaZRzztNhQGV0WYY"

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGoogleGenerativeAI(
    model= "models/gemini-2.5-pro",
    google_api_key= "AIzaSyD6-54qxPYbx7Lu-a0YaZRzztNhQGV0WYY",
    temperature= 0.7
    )

def process(state:AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI response: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process" , process)

graph.add_edge(START , "process")
graph.add_edge("process" , END)

agent = graph.compile()

user_input = input("Enter your message: ")

while user_input.lower() != "exit":
    agent.invoke(
        {
            "messages": [HumanMessage(content=user_input)]
        }
    )
    user_input = input("Enter: ")