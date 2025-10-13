from flask import Flask, request, jsonify, render_template
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent import update, save
import agent
import os


app = Flask(__name__, static_folder="static", template_folder="templates")


# In-memory chat history per single-user demo
conversation_messages: List[BaseMessage] = []

# Local model instance bound to the imported tools (leaves agent.py untouched)
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    google_api_key="AIzaSyD6-54qxPYbx7Lu-a0YaZRzztNhQGV0WYY"
).bind_tools([update, save])

def build_system_prompt() -> SystemMessage:
    return SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.

    The current document content is:{agent.document_content}
    """)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/message", methods=["POST"])
def api_message():
    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    if not user_text:
        return jsonify({"error": "text is required"}), 400

    global conversation_messages
    # 1) Append user's message
    conversation_messages.append(HumanMessage(content=user_text))
    # 2) Call model with system + history
    response = model.invoke([build_system_prompt()] + conversation_messages)
    conversation_messages.append(AIMessage(content=response.content))

    tool_results: List[str] = []
    # 3) Execute any tool calls returned by the model
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args", {})
            if name == "update":
                result = update.invoke(args)
                tool_results.append(result)
                conversation_messages.append(ToolMessage(content=result, tool_call_id=call.get("id", "update")))
            elif name == "save":
                result = save.invoke(args)
                tool_results.append(result)
                conversation_messages.append(ToolMessage(content=result, tool_call_id=call.get("id", "save")))

    ai_contents = [m.content for m in conversation_messages if isinstance(m, AIMessage)]

    return jsonify({
        "ai_messages": ai_contents[-3:],
        "tool_results": tool_results[-3:],
        "document": agent.document_content,
    })


@app.route("/api/state", methods=["GET"])
def api_state():
    global conversation_messages
    ai_contents = [m.content for m in conversation_messages if isinstance(m, AIMessage)]
    tool_results = [m.content for m in conversation_messages if isinstance(m, ToolMessage)]
    return jsonify({
        "ai_messages": ai_contents[-5:],
        "tool_results": tool_results[-5:],
        "document": agent.document_content,
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True)
    filename = data.get("filename", "document").strip() or "document"
    # Reuse the agent save tool implicitly by writing file here to keep API simple
    try:
        safe = filename if filename.endswith(".txt") else f"{filename}.txt"
        with open(safe, "w", encoding="utf-8") as f:
            f.write(agent.document_content)
        return jsonify({"ok": True, "path": safe})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


