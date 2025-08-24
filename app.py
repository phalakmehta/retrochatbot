# ================================================================= #
#  Riddl-O-Tron 9000 - The Ultimate Riddle Challenge!               #
#  GDG Technical Assessment - Final Themed Version                  #
# ================================================================= #

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- System Setup ---
print(">>> Riddl-O-Tron 9000 Boot Sequence Initiated...")
load_dotenv()
print(">>> Loading Game Cartridge...")

# --- 1. Game State Definition ---
class GameState(TypedDict):
    messages: List[BaseMessage]
    riddle_number: int

# --- 2. Riddle & Secret Key Database ---
riddles = [
    {"riddle": "I have cities, but no houses; forests, but no trees; and water, but no fish. What am I?", "answer": "a map"},
    {"riddle": "What has to be broken before you can use it?", "answer": "an egg"},
    {"riddle": "I’m tall when I’m young, and I’m short when I’m old. What am I?", "answer": "a candle"}
]
SECRET_KEY = "GDG-CHALLENGE-COMPLETE-2025"
print(">>> Riddles Loaded... Let the Games Begin!")

# --- 3. The AI Brain (LLM) ---
llm = ChatGroq(model="llama3-8b-8192", temperature=0.8)

# --- 4. Game Logic Nodes ---

def start_game_node(state: GameState):
    """Greets the player and asks the first riddle with a fun theme."""
    welcome_message = """BEEP BOOP! Welcome, contestant, to the Riddl-O-Tron 9000!
I'm your host, and I've got three mind-bending riddles for you.
Answer all three correctly, and you'll win the GRAND PRIZE: a top-secret key!
Let's play! Here is your first question:

"{riddle_text}"
"""
    first_riddle = riddles[0]["riddle"]
    message = AIMessage(content=welcome_message.format(riddle_text=first_riddle))
    return {"messages": [message], "riddle_number": 1}

def check_answer_node(state: GameState):
    """The core game logic node with dynamic, AI-powered hints."""
    current_riddle_number = state["riddle_number"]
    user_answer = state["messages"][-1].content
    
    current_riddle_info = riddles[current_riddle_number - 1]
    correct_answer = current_riddle_info["answer"]
    
    # --- AI JUDGE: First, ask the AI if the answer is correct ---
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI Riddle Judge. The user is answering a riddle. Is their answer correct? The correct answer is '{correct_answer}'. The user's answer is '{user_answer}'. Respond with only the word 'CORRECT' or 'INCORRECT'."),
        ("human", "My answer is: {user_answer}")
    ])
    judge_chain = judge_prompt | llm
    
    # === THIS IS THE FIX: We must pass the variables to the invoke call ===
    judgement = judge_chain.invoke({
        "correct_answer": correct_answer,
        "user_answer": user_answer
    }).content
    # =====================================================================
    
    is_correct = "CORRECT" in judgement.upper()

    if is_correct:
        next_riddle_number = current_riddle_number + 1
        
        if current_riddle_number >= len(riddles):
            response_text = f"CORRECT! BEEP BOOP... AMAZING! You've answered all my riddles! You are a true genius! As promised, here is your GRAND PRIZE:\n\n{SECRET_KEY}"
        else:
            next_riddle_text = riddles[current_riddle_number]["riddle"]
            response_text = f"CORRECT! You got it! You're on a roll! Here's the next one for you:\n\n\"{next_riddle_text}\""
    else:
        # --- AI HINT GENERATOR: If wrong, ask the AI for a subtle hint ---
        hint_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Riddl-O-Tron 9000, a fun game show AI. The user gave a wrong answer. Your job is to give them a fun, subtle hint about why their answer is wrong or how it relates to the correct one, without giving the answer away. Be encouraging!"),
            ("human", "The riddle was: '{riddle}'. The correct answer is '{correct_answer}'. My wrong answer was '{user_answer}'. Give me a hint!")
        ])
        hint_chain = hint_prompt | llm
        hint = hint_chain.invoke({
            "riddle": current_riddle_info["riddle"],
            "correct_answer": correct_answer,
            "user_answer": user_answer
        }).content
        
        response_text = f"BZZZT! Not quite! But let's see... {hint}\nTry answering this riddle again:\n\n\"{current_riddle_info['riddle']}\""
        next_riddle_number = current_riddle_number

    message = AIMessage(content=response_text)
    return {"messages": state["messages"] + [message], "riddle_number": next_riddle_number}


# --- 5. Graph and Workflow Assembly ---
def router(state: GameState):
    return "start_game_node" if len(state["messages"]) == 0 else "check_answer_node"

workflow = StateGraph(GameState)
workflow.add_node("start_game_node", start_game_node)
workflow.add_node("check_answer_node", check_answer_node)
workflow.set_conditional_entry_point(router)
workflow.add_edge("start_game_node", END)
workflow.add_edge("check_answer_node", END)
app_langgraph = workflow.compile()
print(">>> Game Logic Compiled...")

# --- 6. Web Server Interface (Flask) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    messages = [HumanMessage(content=m['content']) if m['type'] == 'human' else AIMessage(content=m['content']) for m in data.get('messages', [])]
    
    current_state = { "messages": messages, "riddle_number": data.get("riddle_number", 0) }
    
    response_from_graph = app_langgraph.invoke(current_state)
    
    serializable_messages = [{'type': 'ai' if isinstance(msg, AIMessage) else 'human', 'content': msg.content} for msg in response_from_graph['messages']]
    json_response = {'messages': serializable_messages, 'riddle_number': response_from_graph.get('riddle_number')}
    
    return jsonify(json_response)

if __name__ == '__main__':
    print(">>> Riddl-O-Tron 9000 is now ONLINE. Access the web interface.")
    app.run(host='0.0.0.0', port=5001)

