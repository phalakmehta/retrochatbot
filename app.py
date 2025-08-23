# ██╗    ██╗ ██████╗ ██████╗ ██████╗
# ██║    ██║██╔═══██╗██╔══██╗██╔══██╗
# ██║ █╗ ██║██║   ██║██████╔╝██████╔╝
# ██║███╗██║██║   ██║██╔══██╗██╔══██╗
# ╚███╔███╔╝╚██████╔╝██║  ██║██████╔╝
#  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝
#
# //==============================================================//
# // CLASSIFIED: W.O.P.R. CORE LOGIC - FIXED VERSION             //
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
    
    # Bounds checking
    if current_riddle_number < 1 or current_riddle_number > len(riddles):
        logger.error(f"Invalid riddle number: {current_riddle_number}")
        return handle_invalid_state(state)
    
    correct_answer = riddles[current_riddle_number - 1]["answer"]
    current_riddle = riddles[current_riddle_number - 1]["riddle"]

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are W.O.P.R., a cryptic, logical supercomputer. Your task is to evaluate a user's answer to a riddle based on the information provided."),
        ("human", """Here is the data for evaluation:
        - The riddle is: "{riddle}"
        - The correct answer is: "{correct_answer}"
        - The user's answer is: "{user_answer}"
        - This is riddle {riddle_number} of {total_riddles}

        Please provide your judgment:
        - If the answer is correct, start your response with "CORRECT"
        - If incorrect, start with "INCORRECT"
        - Keep responses brief and in character as W.O.P.R.
        """)
    ])
    
    judgement_chain = prompt_template | llm

    logger.info(f"Querying AI for riddle {current_riddle_number}")
    ai_response = judgement_chain.invoke({
        "riddle": current_riddle,
        "correct_answer": correct_answer,
        "user_answer": user_answer,
        "riddle_number": current_riddle_number,
        "total_riddles": len(riddles)
    })
    
    response_text = ai_response.content

    # Process the response and determine next state
    if response_text.strip().upper().startswith("CORRECT"):
        next_riddle_number = current_riddle_number + 1
        
        if next_riddle_number > len(riddles):
            # Game complete
            response_text = f"CORRECT! LOGIC TEST COMPLETE.\n\nTHE SECRET KEY IS: {SECRET_KEY}"
        else:
            # Add next riddle
            next_riddle_text = riddles[next_riddle_number - 1]["riddle"]
            response_text += f"\n\nNext riddle:\n\"{next_riddle_text}\""
    else:
        # Wrong answer, stay on same riddle
        next_riddle_number = current_riddle_number

    message = AIMessage(content=response_text)
    # Return complete message history plus new message
    return {
        "messages": state["messages"] + [message], 
        "riddle_number": next_riddle_number
    }

def check_answer_fallback(state: GameState):
    """Fallback logic when AI is unavailable."""
    current_riddle_number = state["riddle_number"]
    
    # Bounds checking
    if current_riddle_number < 1 or current_riddle_number > len(riddles):
        logger.error(f"Invalid riddle number: {current_riddle_number}")
        return handle_invalid_state(state)
    
    user_answer = state["messages"][-1].content.lower().strip()
    correct_answer = riddles[current_riddle_number - 1]["answer"].lower()
    
    logger.info(f"Using fallback logic for riddle {current_riddle_number}")
    
    # Enhanced matching logic
    is_correct = (
        correct_answer in user_answer or 
        user_answer in correct_answer or
        user_answer == correct_answer or
        # Handle common variations
        (correct_answer == "map" and "map" in user_answer) or
        (correct_answer == "needle" and "needle" in user_answer) or
        (correct_answer == "egg" and "egg" in user_answer)
    )
    
    if is_correct:
        next_riddle_number = current_riddle_number + 1
        if next_riddle_number > len(riddles):
            response_text = f"CORRECT! LOGIC TEST COMPLETE.\n\nTHE SECRET KEY IS: {SECRET_KEY}"
        else:
            next_riddle = riddles[next_riddle_number - 1]["riddle"]
            response_text = f"CORRECT! WELL DONE, PROFESSOR.\n\nNext riddle:\n\"{next_riddle}\""
    else:
        response_text = "INCORRECT. THE LOGIC CIRCUITS DETECT AN ERROR IN YOUR REASONING. PLEASE RECALCULATE."
        next_riddle_number = current_riddle_number
    
    message = AIMessage(content=response_text)
    return {
        "messages": state["messages"] + [message], 
        "riddle_number": next_riddle_number
    }

def handle_invalid_state(state: GameState):
    """Handle invalid game states gracefully."""
    logger.warning("Handling invalid state, resetting to first riddle")
    error_message = AIMessage(content="ERROR DETECTED. RESETTING TO FIRST RIDDLE.\n\n\"I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?\"")
    return {
        "messages": state["messages"] + [error_message],
        "riddle_number": 1
    }

def check_answer(state: GameState):
    """Main answer checking function with AI fallback."""
    try:
        # Validate state
        if not state.get("messages"):
            logger.error("No messages in state")
            return handle_invalid_state(state)
        
        if not isinstance(state.get("riddle_number"), int):
            logger.error(f"Invalid riddle_number: {state.get('riddle_number')}")
            return handle_invalid_state(state)
        
        if llm is not None:
            return check_answer_with_ai(state)
        else:
            return check_answer_fallback(state)
    except Exception as e:
        logger.error(f"AI check failed: {e}, falling back to simple logic")
        traceback.print_exc()
        try:
            return check_answer_fallback(state)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return handle_invalid_state(state)

print(">>> COGNITIVE NODES REDEFINED FOR DYNAMIC RESPONSE...")

# --- 4. NEURAL PATHWAY CONSTRUCTION (LANGGRAPH) ---
def should_start_game(state: GameState):
    """Determine the entry point based on current state."""
    messages = state.get("messages", [])
    riddle_number = state.get("riddle_number", 0)
    
    logger.info(f"Entry decision: {len(messages)} messages, riddle_number: {riddle_number}")
    
    # If no messages, start the game
    if len(messages) == 0:
        return "ask_riddle_node"
    
    # If we have messages, check the answer
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
        # Test the system with a minimal state
        test_state = {"messages": [], "riddle_number": 0}
        result = app_langgraph.invoke(test_state)
        
        return jsonify({
            "status": "operational",
            "ai_available": llm is not None,
            "message": "W.O.P.R. SYSTEMS ONLINE",
            "riddles_total": len(riddles)
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
            logger.error("Request is not JSON")
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if data is None:
            logger.error("No JSON data received")
            return jsonify({"error": "Invalid JSON data"}), 400
        
        logger.info(f"Received data: {data}")
        
        # Process messages - ensure they're properly formatted
        messages = []
        for m in data.get('messages', []):
            msg_type = m.get('type', '')
            content = m.get('content', '')
            
            if msg_type == 'human':
                messages.append(HumanMessage(content=content))
            elif msg_type == 'ai':
                messages.append(AIMessage(content=content))
            else:
                logger.warning(f"Unknown message type: {msg_type}")

        # Get riddle number, default to 0 for new games
        riddle_number = data.get("riddle_number", 0)
        
        # Ensure riddle_number is valid
        if not isinstance(riddle_number, int) or riddle_number < 0:
            logger.warning(f"Invalid riddle_number {riddle_number}, defaulting to 0")
            riddle_number = 0

        current_state = {
            "messages": messages,
            "riddle_number": riddle_number
        }
        
        logger.info(f"[+] PROCESSING STATE... RIDDLE_LVL:{riddle_number}, MESSAGES:{len(messages)}")
        
        # Execute the graph
        response_from_graph = app_langgraph.invoke(current_state)
        
        logger.info(f"Graph response: riddle_number={response_from_graph.get('riddle_number')}, messages={len(response_from_graph.get('messages', []))}")
        
        # Serialize response
        serializable_messages = []
        for msg in response_from_graph['messages']:
            if isinstance(msg, AIMessage):
                msg_type = 'ai'
            elif isinstance(msg, HumanMessage):
                msg_type = 'human'
            else:
                msg_type = 'unknown'
            serializable_messages.append({'type': msg_type, 'content': msg.content})

        json_response = {
            'messages': serializable_messages,
            'riddle_number': response_from_graph.get('riddle_number', 0)
        }
        
        logger.info(f"[+] TRANSMITTING RESPONSE... RIDDLE_LVL:{json_response['riddle_number']}")
        return jsonify(json_response), 200
        
    except Exception as e:
        logger.error(f"❌ ERROR IN CHAT ENDPOINT: {e}")
        logger.error(traceback.format_exc())
        
        # Return graceful error response
        error_message = "SYSTEM ERROR DETECTED. ATTEMPTING RECOVERY... PLEASE TRY AGAIN."
        fallback_riddle_number = 0
        
        # Try to preserve the riddle number if possible
        try:
            if 'data' in locals() and data:
                fallback_riddle_number = data.get("riddle_number", 0)
        except:
            pass
        
        return jsonify({
            'messages': [{'type': 'ai', 'content': error_message}],
            'riddle_number': fallback_riddle_number,
            'error': 'Internal system error'
        }), 500

@app_flask.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app_flask.errorhandler(500)
def internal_error(error):
    logger.error(f"Flask 500 error: {error}")
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
        logger.info(f"✅ SYSTEM SELF-TEST PASSED: {test_result}")
    except Exception as e:
        logger.warning(f"⚠️ SYSTEM SELF-TEST WARNING: {e}")
        logger.warning("PROCEEDING WITH FALLBACK CAPABILITIES...")
    
    # Get port from environment (Railway sets PORT automatically)
    port = int(os.environ.get('PORT', 5001))
    app_flask.run(host='0.0.0.0', port=port, debug=False)


