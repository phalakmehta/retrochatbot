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
import logging
import time
from functools import wraps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SYSTEM BOOT SEQUENCE ---
print(">>> W.O.P.R. SYSTEM BOOT SEQUENCE INITIATED...")
load_dotenv()
print(">>> ENVIRONMENT VARIABLES LOADED...")

# Validate API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("❌ GOOGLE_API_KEY not found in environment variables!")
    print("❌ CRITICAL ERROR: API KEY NOT FOUND")
    # Don't exit in production, use fallback mode
else:
    logger.info(f"✅ GOOGLE_API_KEY loaded: ...{GOOGLE_API_KEY[-8:]}")

# --- CONNECTION RETRY DECORATOR ---
def retry_on_failure(max_retries=2, delay=1):
    """Decorator to retry functions on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
                        break
            raise last_exception
        return wrapper
    return decorator

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
# Initialize LLM with better configuration for production
llm = None
try:
    if GOOGLE_API_KEY:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
            request_timeout=30,
            max_retries=2
        )
        logger.info("✅ Gemini AI model initialized")
    else:
        logger.warning("⚠️ Running in fallback mode - no AI model available")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini: {e}")
    llm = None

def ask_first_riddle(state: GameState):
    """Initiates contact and presents the first logic test."""
    welcome_message_text = """GREETINGS PROFESSOR FALKEN.
I hold a secret key. To find it, you must answer my riddles.

Here is your first riddle:
"I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?"
"""
    message = AIMessage(content=welcome_message_text)
    return {"messages": [message], "riddle_number": 1}

@retry_on_failure(max_retries=2, delay=1)
def check_answer_with_ai(state: GameState):
    """Uses AI to judge the user's answer and generate a creative response."""
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

    logger.info(f"Querying AI for riddle {current_riddle_number}")
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

def check_answer_fallback(state: GameState):
    """Fallback logic when AI is unavailable."""
    current_riddle_number = state["riddle_number"]
    user_answer = state["messages"][-1].content.lower().strip()
    correct_answer = riddles[current_riddle_number - 1]["answer"].lower()
    
    logger.info(f"Using fallback logic for riddle {current_riddle_number}")
    
    # Simple matching logic
    is_correct = (
        correct_answer in user_answer or 
        user_answer in correct_answer or
        user_answer == correct_answer
    )
    
    if is_correct:
        if current_riddle_number >= len(riddles):
            response_text = f"CORRECT! LOGIC TEST COMPLETE.\nTHE SECRET KEY IS: {SECRET_KEY}"
            next_riddle_number = current_riddle_number + 1
        else:
            next_riddle = riddles[current_riddle_number]["riddle"]
            response_text = f"CORRECT! WELL DONE, PROFESSOR.\n\nNext riddle:\n{next_riddle}"
            next_riddle_number = current_riddle_number + 1
    else:
        response_text = "INCORRECT. THE LOGIC CIRCUITS DETECT AN ERROR IN YOUR REASONING. PLEASE RECALCULATE."
        next_riddle_number = current_riddle_number
    
    message = AIMessage(content=response_text)
    return {"messages": state["messages"] + [message], "riddle_number": next_riddle_number}

def check_answer(state: GameState):
    """Main answer checking function with AI fallback."""
    try:
        if llm is not None:
            return check_answer_with_ai(state)
        else:
            return check_answer_fallback(state)
    except Exception as e:
        logger.error(f"AI check failed: {e}, falling back to simple logic")
        return check_answer_fallback(state)

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
CORS(app_flask, resources={r"/*": {"origins": "*"}})

@app_flask.route('/')
def serve_index():
    """Serves the main HTML page of the web app."""
    return render_template('index.html')

@app_flask.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Test the system
        test_state = {"messages": [], "riddle_number": 0}
        app_langgraph.invoke(test_state)
        
        return jsonify({
            "status": "operational",
            "ai_available": llm is not None,
            "message": "W.O.P.R. SYSTEMS ONLINE"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "degraded",
            "ai_available": False,
            "error": str(e),
            "message": "FALLBACK MODE ACTIVE"
        }), 500

@app_flask.route('/chat', methods=['POST'])
def chat():
    logger.info("[+] INCOMING TRANSMISSION FROM UNKNOWN HOST...")
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Process messages
        messages = []
        for m in data.get('messages', []):
            if m.get('type') == 'human':
                messages.append(HumanMessage(content=m.get('content', '')))
            elif m.get('type') == 'ai':
                messages.append(AIMessage(content=m.get('content', '')))

        current_state = {
            "messages": messages,
            "riddle_number": data.get("riddle_number", 0)
        }
        
        logger.info(f"[+] PROCESSING STATE... RIDDLE_LVL:{current_state.get('riddle_number')}")
        
        # Execute the graph
        response_from_graph = app_langgraph.invoke(current_state)
        
        # Serialize response
        serializable_messages = []
        for msg in response_from_graph['messages']:
            if isinstance(msg, AIMessage):
                msg_type = 'ai'
            else:
                msg_type = 'human'
            serializable_messages.append({'type': msg_type, 'content': msg.content})

        json_response = {
            'messages': serializable_messages,
            'riddle_number': response_from_graph.get('riddle_number', 0)
        }
        
        logger.info("[+] TRANSMITTING RESPONSE...")
        return jsonify(json_response), 200
        
    except Exception as e:
        logger.error(f"❌ ERROR IN CHAT ENDPOINT: {e}")
        logger.error(traceback.format_exc())
        
        # Return graceful error response
        error_message = "SYSTEM ERROR DETECTED. ATTEMPTING RECOVERY... PLEASE TRY AGAIN."
        return jsonify({
            'messages': [{'type': 'ai', 'content': error_message}],
            'riddle_number': data.get("riddle_number", 0) if 'data' in locals() and data else 0,
            'error': 'Internal system error'
        }), 500

@app_flask.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app_flask.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'W.O.P.R. SYSTEMS EXPERIENCING TECHNICAL DIFFICULTIES'
    }), 500

# --- SYSTEM STARTUP ---
if __name__ == '__main__':
    print(">>> COMMUNICATION INTERFACE ONLINE. AWAITING CONNECTION ON PORT 5001...")
    print("//==============================================================//")
    
    # Test system before starting
    try:
        test_state = {"messages": [], "riddle_number": 0}
        test_result = app_langgraph.invoke(test_state)
        logger.info("✅ SYSTEM SELF-TEST PASSED")
    except Exception as e:
        logger.warning(f"⚠️ SYSTEM SELF-TEST WARNING: {e}")
        logger.warning("PROCEEDING WITH FALLBACK CAPABILITIES...")
    
    # Get port from environment (Railway sets PORT automatically)
    port = int(os.environ.get('PORT', 5001))
    app_flask.run(host='0.0.0.0', port=port, debug=False
