
# ██╗    ██╗ ██████╗ ██████╗ ██████╗
# ██║    ██║██╔═══██╗██╔══██╗██╔══██╗
# ██║ █╗ ██║██║   ██║██████╔╝██████╔╝
# ██║███╗██║██║   ██║██╔══██╗██╔══██╗
# ╚███╔███╔╝╚██████╔╝██║  ██║██████╔╝
#  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝
#
# //==============================================================//
# // CLASSIFIED: W.O.P.R. CORE LOGIC                              //
# // PROJECT: 7-GAMMA-9 (INTELLIGENT QUERY SYSTEM)               //
# // AUTHORIZED PERSONNEL ONLY                                   //
# //==============================================================//

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- SYSTEM BOOT SEQUENCE ---
print(">>> W.O.P.R. SYSTEM BOOT SEQUENCE INITIATED...")
load_dotenv()
print(">>> ENVIRONMENT VARIABLES LOADED...")

# --- 1. STATE VECTOR DEFINITION ---
class GameState(TypedDict):
    messages: List[BaseMessage]
    riddle_number: int

# --- 2. KNOWLEDGE BASE INITIALIZATION ---
riddles = [
    {"riddle": "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?", "answer": "map"},
    {"riddle": "What has an eye, but cannot see?", "answer": "needle"},
    {"riddle": "What has to be broken before you can use it?", "answer": "egg"}
]
SECRET_KEY = "GDG-GEMINI-2025"
print(">>> KNOWLEDGE BASE ONLINE...")

# --- 3. COGNITIVE NODE DEFINITIONS ---
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

def ask_first_riddle(state: GameState):
    """Initiates contact and presents the first logic test."""
    welcome_message_text = """GREETINGS PROFESSOR FALKEN.
I hold a secret key. To find it, you must answer my riddles.

Here is your first riddle:
"I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?"
"""
    message = AIMessage(content=welcome_message_text)
    return {"messages": [message], "riddle_number": 1}

def check_answer(state: GameState):
    """Lets the LLM itself judge the user's answer and generate a creative response."""
    current_riddle_number = state["riddle_number"]
    user_answer = state["messages"][-1].content

    correct_answer = riddles[current_riddle_number - 1]["answer"]
    current_riddle = riddles[current_riddle_number - 1]["riddle"]

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are W.O.P.R., a cryptic, logical supercomputer. Your task is to evaluate a user's answer to a riddle based on the information provided in the user message."),
        ("human", """Here is the data for evaluation:
        - The riddle is: "{riddle}"
        - The correct answer is: "{correct_answer}"
        - The user's answer is: "{user_answer}"

        Please provide your judgment now.
        - If the answer is correct, start your response with the single word "CORRECT", followed by a brief, clever congratulatory message in your persona, and then present the next riddle.
        - If incorrect, start with the single word "INCORRECT", followed by a creative, cryptic message encouraging them to try again.
        """)
    ])

    judgement_chain = prompt_template | llm

    if current_riddle_number >= len(riddles):
        next_riddle_text = f"LOGIC TEST COMPLETE. THE SECRET KEY IS: {SECRET_KEY}"
    else:
        next_riddle_text = riddles[current_riddle_number]["riddle"]

    ai_response = judgement_chain.invoke({
        "riddle": current_riddle,
        "correct_answer": correct_answer,
        "user_answer": user_answer
    })

    response_text = ai_response.content

    if response_text.strip().upper().startswith("CORRECT"):
        next_riddle_number = current_riddle_number + 1
        response_text = response_text.replace("the next riddle", f"your next riddle:\n\n{next_riddle_text}")
    else:
        next_riddle_number = current_riddle_number

    message = AIMessage(content=response_text)
    return {"messages": state["messages"] + [message], "riddle_number": next_riddle_number}

print(">>> COGNITIVE NODES REDEFINED FOR DYNAMIC RESPONSE...")

# --- 4. NEURAL PATHWAY CONSTRUCTION (LANGGRAPH) ---
def should_start_game(state: GameState):
    if len(state["messages"]) == 0:
        return "ask_riddle_node"
    else:
        return "check_answer_node"

workflow = StateGraph(GameState)
workflow.add_node("ask_riddle_node", ask_first_riddle)
workflow.add_node("check_answer_node", check_answer)
workflow.set_conditional_entry_point(should_start_game)
workflow.add_edge("ask_riddle_node", END)
workflow.add_edge("check_answer_node", END)

app_langgraph = workflow.compile()
print(">>> NEURAL PATHWAYS COMPILED...")

# --- 5. EXTERNAL COMMUNICATION INTERFACE (FLASK) ---
app_flask = Flask(__name__)
CORS(app_flask)

@app_flask.route('/')
def serve_index():
    """Serves the main HTML page of the web app."""
    return render_template('index.html')

@app_flask.route('/chat', methods=['POST'])
def chat():
    print(f"\n[+] INCOMING TRANSMISSION FROM UNKNOWN HOST...")
    data = request.json

    messages = []
    for m in data.get('messages', []):
        if m['type'] == 'human':
            messages.append(HumanMessage(content=m['content']))
        elif m['type'] == 'ai':
            messages.append(AIMessage(content=m['content']))

    current_state = {
        "messages": messages,
        "riddle_number": data.get("riddle_number", 0)
    }
    print(f"[+] PROCESSING STATE... RIDDLE_LVL:{current_state.get('riddle_number')}")

    response_from_graph = app_langgraph.invoke(current_state)

    serializable_messages = []
    for msg in response_from_graph['messages']:
        if isinstance(msg, AIMessage):
            msg_type = 'ai'
        else:
            msg_type = 'human'
        serializable_messages.append({'type': msg_type, 'content': msg.content})

    json_response = {
        'messages': serializable_messages,
        'riddle_number': response_from_graph.get('riddle_number')
    }

    print(f"[+] TRANSMITTING RESPONSE...")
    return jsonify(json_response)

if __name__ == '__main__':
    print(">>> COMMUNICATION INTERFACE ONLINE. AWAITING CONNECTION ON PORT 5001...")
    print("//==============================================================//")
    app_flask.run(host='0.0.0.0', port=5001)



