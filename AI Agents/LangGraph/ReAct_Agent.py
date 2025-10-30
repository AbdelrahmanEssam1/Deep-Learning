from typing import Annotated , Sequence , TypedDict
from langchain_core.messages import BaseMessage #The foundational class for all message types in LangChain.
from langchain_core.messages import ToolMessage #Passes data back to llm after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage #System-level instructions for the AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , END , START
from langgraph.prebuilt import ToolNode


#Anotated -> Provides additional context without affecting the type itself
# email = Annotated[str, "This has to be a valid email address"]

# print(email.__metadata__)

#Sequence -> To atuomatically handle the state updates for sequences such by adding a new messages to a chat history

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int , b: int):
    """Add integer numbers and return the result."""
    return a + b

@tool
def subtract(a: int , b: int):
    """Subtract integer numbers and return the result."""
    return a - b

@tool
def multiply(a: int , b: int):
    """Multiply integer numbers and return the result."""
    return a * b

tools = [add , subtract , multiply]

model = ChatGoogleGenerativeAI(
    model = "models/gemini-2.5-pro",
    google_api_key = "AIzaSyD6-54qxPYbx7Lu-a0YaZRzztNhQGV0WYY"
).bind_tools(tools)

def model_call(state : AgentState) -> AgentState:
    system_prompt = SystemMessage(content= "You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state['messages'])
    return {"messages": [response]}

def should_continue(state: AgentState) -> AgentState:
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent" , model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools" , tool_node)
graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools" , "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message , tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 then multiply the result by 4, then tell me a joke")]}
print_stream(app.stream(inputs , stream_mode = "values"))