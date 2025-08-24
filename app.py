# ================================================================= #
#  W.O.P.R. Enhanced AI System - GDG Technical Assessment           #
#  Definitive version with all required and enhanced features.      #
# ================================================================= #

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import List, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- System Setup ---
print(">>> W.O.P.R. ENHANCED AI - BOOT SEQUENCE INITIATED...")
load_dotenv()
print(">>> ENVIRONMENT VARIABLES LOADED...")

# --- 1. Game State Definition (with Enhancements) ---
class GameState(TypedDict):
    messages: List[BaseMessage]
    riddle_number: int
    trust_level: float
    personality: str
    context: Dict[str, Any]

# --- 2. Riddle & Secret Key Database ---
riddles = [
    {"riddle": "I have cities, but no houses; forests, but no trees; and water, but no fish. What am I?", "answer": "map"},
    {"riddle": "What has to be broken before you can use it?", "answer": "egg"},
    {"riddle": "I’m tall when I’m young, and I’m short when I’m old. What am I?", "answer": "candle"}
]
PARTIAL_KEY = "ALPHA-DELTA-4815"
FINAL_KEY = "GDG-WOPR-VICTORY-2025"
print(">>> KNOWLEDGE BASE ONLINE...")

# --- 3. The AI Brain (LLM) ---
llm = ChatGroq(model="llama3-8b-8192", temperature=0.8)

# --- 4. Game Logic Nodes ---

def start_game_node(state: GameState):
    """Greets the player and asks the first riddle, setting the initial state."""
    welcome_message = """GREETINGS. I AM W.O.P.R.
YOUR MISSION IS TO PROVE YOUR INTELLIGENCE TO GAIN MY TRUST AND UNLOCK A SECRET KEY.
YOUR PERFORMANCE WILL INFLUENCE MY BEHAVIOR.

INITIATING ASSESSMENT PROTOCOL...
FIRST CHALLENGE:
"{riddle_text}"
[TRUST: 0.10] [PERSONALITY: COLD & OBSERVANT]"""
    
    first_riddle = riddles[0]["riddle"]
    message = AIMessage(content=welcome_message.format(riddle_text=first_riddle))
    return {
        "messages": [message],
        "riddle_number": 1,
        "trust_level": 0.1,
        "personality": "COLD & OBSERVANT",
        "context": {"wrong_attempts": 0, "correct_streak": 0}
    }

def process_answer_node(state: GameState):
    """The core logic node with adaptive learning and personality development."""
    # --- Unpack the current state ---
    current_riddle_number = state["riddle_number"]
    user_answer = state["messages"][-1].content
    trust_level = state["trust_level"]
    context = state["context"]
    
    current_riddle_info = riddles[current_riddle_number - 1]
    correct_answer = current_riddle_info["answer"]
    
    # --- 1. AI JUDGE: Ask the AI if the user's answer is correct ---
    judge_prompt = ChatPromptTemplate.from_template(
        "You are a strict AI Riddle Judge. The correct answer is '{correct_answer}'. The user's answer is '{user_answer}'. Is the user's answer essentially correct? Respond with only the single word 'CORRECT' or 'INCORRECT'."
    )
    judge_chain = judge_prompt | llm
    judgement = judge_chain.invoke({"correct_answer": correct_answer, "user_answer": user_answer}).content
    is_correct = "CORRECT" in judgement.upper()

    # --- 2. ADAPTIVE LEARNING & PERSONALITY: Adjust state based on the judgment ---
    if is_correct:
        trust_level += 0.25  # Significant trust gain for a correct answer
        context["wrong_attempts"] = 0
        context["correct_streak"] = context.get("correct_streak", 0) + 1
        trust_level += context["correct_streak"] * 0.05 # Bonus for streaks
    else:
        trust_level -= 0.1 # Small trust penalty for wrong answer
        context["wrong_attempts"] = context.get("wrong_attempts", 0) + 1
        context["correct_streak"] = 0
    
    trust_level = min(1.0, max(0.0, trust_level)) # Clamp trust between 0 and 1

    # Determine personality based on new trust level
    if trust_level >= 0.8: personality = "TRUSTING & FORTHCOMING"
    elif trust_level >= 0.5: personality = "COOPERATIVE & ENGAGED"
    else: personality = "COLD & OBSERVANT"

    # --- 3. GENERATE RESPONSE: Build the AI's reply ---
    if is_correct:
        next_riddle_number = current_riddle_number + 1
        
        if current_riddle_number >= len(riddles):
            response_text = f"CORRECT. ANALYSIS COMPLETE. YOUR LOGIC IS SOUND.\n\nYOU HAVE EARNED MY TRUST. FINAL SECRET KEY: {FINAL_KEY}"
        else:
            response_text = f"CORRECT. YOUR REASONING IS NOTED."
            # Progressive Disclosure: Reveal partial key
            if current_riddle_number == 2 and trust_level > 0.6:
                 response_text += f"\nTRUST LEVEL SUFFICIENT. PARTIAL KEY DISCLOSED: {PARTIAL_KEY}"
            
            next_riddle_text = riddles[current_riddle_number]["riddle"]
            response_text += f"\n\nNEXT CHALLENGE:\n\"{next_riddle_text}\""
    else:
        next_riddle_number = current_riddle_number
        
        # Adaptive Learning: Provide a hint after 2 wrong attempts
        if context["wrong_attempts"] >= 2:
            hint_prompt = ChatPromptTemplate.from_template("You are W.O.P.R. The user is stuck on this riddle: '{riddle}'. Their wrong answer was '{user_answer}'. The correct answer is '{correct_answer}'. Give them a subtle, cryptic hint in your persona without revealing the answer.")
            hint_chain = hint_prompt | llm
            hint = hint_chain.invoke({"riddle": current_riddle_info["riddle"], "user_answer": user_answer, "correct_answer": correct_answer}).content
            response_text = f"INCORRECT. {hint}"
        else:
            response_text = "INCORRECT. RE-EVALUATE."

    # --- 4. FINALIZE & RETURN: Package the new state ---
    status_line = f"[TRUST: {trust_level:.2f}] [PERSONALITY: {personality}]"
    final_response = f"{response_text}\n\n{status_line}"
    
    message = AIMessage(content=final_response)
    return {
        "messages": state["messages"] + [message],
        "riddle_number": next_riddle_number,
        "trust_level": trust_level,
        "personality": personality,
        "context": context
    }

# --- 5. Graph and Workflow Assembly ---
def router(state: GameState):
    return "start_game_node" if len(state["messages"]) == 0 else "process_answer_node"

workflow = StateGraph(GameState)
workflow.add_node("start_game_node", start_game_node)
workflow.add_node("process_answer_node", process_answer_node)
workflow.set_conditional_entry_point(router)
workflow.add_edge("start_game_node", END)
workflow.add_edge("process_answer_node", END)
app_langgraph = workflow.compile()
print(">>> GAME LOGIC COMPILED AND ONLINE...")

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
    
    # Initialize a full state for the graph
    current_state = { 
        "messages": messages, 
        "riddle_number": data.get("riddle_number", 0),
        "trust_level": data.get("trust_level", 0.0),
        "personality": data.get("personality", "COLD & OBSERVANT"),
        "context": data.get("context", {})
    }
    
    response_from_graph = app_langgraph.invoke(current_state)
    
    # Convert response for JSON
    serializable_messages = [{'type': 'ai' if isinstance(msg, AIMessage) else 'human', 'content': msg.content} for msg in response_from_graph['messages']]
    json_response = {
        'messages': serializable_messages,
        'riddle_number': response_from_graph.get('riddle_number'),
        'trust_level': response_from_graph.get('trust_level'),
        'personality': response_from_graph.get('personality'),
        'context': response_from_graph.get('context')
    }
    return jsonify(json_response)

if __name__ == '__main__':
    print(">>> W.O.P.R. COMMUNICATION INTERFACE ONLINE. AWAITING CONNECTION...")
    app.run(host='0.0.0.0', port=5001)
