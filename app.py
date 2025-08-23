# ██╗    ██╗ ██████╗ ██████╗ ██████╗
# ██║    ██║██╔═══██╗██╔══██╗██╔══██╗
# ██║ █╗ ██║██║   ██║██████╔╝██████╔╝
# ██║███╗██║██║   ██║██╔══██╗██╔══██╗
# ╚███╔███╔╝╚██████╔╝██║  ██║██████╔╝
#  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝
#
# //==============================================================//
# // CLASSIFIED: W.O.P.R. CORE LOGIC - CONNECTION FIXED VERSION  //
# // PROJECT: 7-GAMMA-9 (INTELLIGENT QUERY SYSTEM)               //
# // AUTHORIZED PERSONNEL ONLY                                   //
# //==============================================================//

import os
import logging
import time
from functools import wraps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import traceback
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# Global thread executor for AI calls
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="wopr-ai")

def cleanup_executor():
    """Clean shutdown of thread executor"""
    global executor
    if executor:
        logger.info("Shutting down AI executor...")
        executor.shutdown(wait=True)

# Register cleanup
signal.signal(signal.SIGTERM, lambda s, f: cleanup_executor())
signal.signal(signal.SIGINT, lambda s, f: cleanup_executor())

# --- ENHANCED CONNECTION RETRY DECORATOR ---
def retry_on_failure(max_retries=3, delay=1, backoff_factor=2):
    """Enhanced decorator with exponential backoff and timeout handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    # Run with timeout to prevent hanging
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=25)  # 25 second timeout
                    
                except (FuturesTimeoutError, TimeoutError) as e:
                    last_exception = e
                    logger.warning(f"Function timeout on attempt {attempt + 1}")
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)[:100]}")
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        break
            
            # Return graceful fallback instead of raising
            logger.error(f"Function failed after all retries: {last_exception}")
            return None
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

# --- 3. ENHANCED LLM INITIALIZATION ---
llm = None
llm_lock = threading.Lock()

def initialize_llm():
    """Thread-safe LLM initialization with connection pooling."""
    global llm
    with llm_lock:
        if llm is None and GOOGLE_API_KEY:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.7,
                    google_api_key=GOOGLE_API_KEY,
                    request_timeout=20,  # Reduced timeout
                    max_retries=1,       # Fewer retries per request
                    max_output_tokens=500,  # Limit response size
                    top_p=0.8,
                    top_k=40
                )
                # Test the connection
                test_response = llm.invoke("Test connection")
                logger.info("✅ Gemini AI model initialized and tested")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {e}")
                llm = None
                return False
    return llm is not None

# Initialize on startup
initialize_llm()

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
    """Uses AI to judge the user's answer with enhanced error handling."""
    try:
        current_riddle_number = state["riddle_number"]
        user_answer = state["messages"][-1].content
        
        # Validate inputs
        if not (1 <= current_riddle_number <= len(riddles)):
            logger.error(f"Invalid riddle number: {current_riddle_number}")
            return None
            
        if not user_answer or len(user_answer.strip()) == 0:
            logger.error("Empty user answer")
            return None
        
        correct_answer = riddles[current_riddle_number - 1]["answer"]
        current_riddle = riddles[current_riddle_number - 1]["riddle"]

        # Simplified prompt to reduce processing time
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are W.O.P.R. Judge if the user's answer matches the riddle answer. Start with CORRECT or INCORRECT. Be brief."),
            ("human", """Riddle: "{riddle}"
Correct answer: "{correct_answer}"
User answer: "{user_answer}"
Riddle {riddle_number} of {total_riddles}""")
        ])
        
        # Ensure LLM is available
        if not llm:
            logger.warning("LLM not available, reinitializing...")
            if not initialize_llm():
                return None
        
        judgement_chain = prompt_template | llm
        
        logger.info(f"Querying AI for riddle {current_riddle_number}")
        ai_response = judgement_chain.invoke({
            "riddle": current_riddle[:100],  # Truncate long riddles
            "correct_answer": correct_answer,
            "user_answer": user_answer[:50],  # Truncate long answers
            "riddle_number": current_riddle_number,
            "total_riddles": len(riddles)
        })
        
        if not ai_response or not hasattr(ai_response, 'content'):
            logger.error("Invalid AI response")
            return None
            
        response_text = ai_response.content.strip()
        
        # Process the response
        if response_text.upper().startswith("CORRECT"):
            next_riddle_number = current_riddle_number + 1
            
            if next_riddle_number > len(riddles):
                response_text = f"CORRECT! LOGIC TEST COMPLETE.\n\nTHE SECRET KEY IS: {SECRET_KEY}"
            else:
                next_riddle_text = riddles[next_riddle_number - 1]["riddle"]
                response_text += f"\n\nNext riddle:\n\"{next_riddle_text}\""
        else:
            next_riddle_number = current_riddle_number

        message = AIMessage(content=response_text)
        return {
            "messages": state["messages"] + [message], 
            "riddle_number": next_riddle_number
        }
        
    except Exception as e:
        logger.error(f"AI processing error: {e}")
        return None

def check_answer_fallback(state: GameState):
    """Enhanced fallback logic when AI is unavailable."""
    try:
        current_riddle_number = state.get("riddle_number", 1)
        
        # Bounds checking
        if not (1 <= current_riddle_number <= len(riddles)):
            logger.error(f"Invalid riddle number: {current_riddle_number}")
            return handle_invalid_state(state)
        
        user_answer = state["messages"][-1].content.lower().strip()
        correct_answer = riddles[current_riddle_number - 1]["answer"].lower()
        
        logger.info(f"Using fallback logic for riddle {current_riddle_number}")
        
        # Enhanced matching with fuzzy logic
        answer_words = user_answer.split()
        is_correct = (
            correct_answer in user_answer or 
            user_answer == correct_answer or
            correct_answer in answer_words or
            any(correct_answer in word for word in answer_words) or
            # Handle specific cases
            (correct_answer == "map" and any(w in user_answer for w in ["map", "atlas", "chart"])) or
            (correct_answer == "needle" and any(w in user_answer for w in ["needle", "pin", "sewing"])) or
            (correct_answer == "egg" and any(w in user_answer for w in ["egg", "shell", "yolk"]))
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
    except Exception as e:
        logger.error(f"Fallback logic error: {e}")
        return handle_invalid_state(state)

def handle_invalid_state(state: GameState):
    """Handle invalid game states gracefully."""
    logger.warning("Handling invalid state, resetting to first riddle")
    try:
        error_message = AIMessage(content="ERROR DETECTED. RESETTING TO FIRST RIDDLE.\n\n\"I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?\"")
        return {
            "messages": state.get("messages", []) + [error_message],
            "riddle_number": 1
        }
    except Exception as e:
        logger.error(f"Error in invalid state handler: {e}")
        # Last resort fallback
        return {
            "messages": [AIMessage(content="SYSTEM RESET. FIRST RIDDLE: What am I if I have cities but no houses?")],
            "riddle_number": 1
        }

def check_answer(state: GameState):
    """Main answer checking function with robust error handling."""
    try:
        # Validate state thoroughly
        if not isinstance(state, dict):
            logger.error("State is not a dictionary")
            return handle_invalid_state({"messages": [], "riddle_number": 1})
            
        messages = state.get("messages", [])
        if not messages or len(messages) == 0:
            logger.error("No messages in state")
            return handle_invalid_state(state)
        
        riddle_number = state.get("riddle_number", 1)
        if not isinstance(riddle_number, int) or riddle_number < 1:
            logger.error(f"Invalid riddle_number: {riddle_number}")
            return handle_invalid_state(state)
        
        # Try AI first, then fallback
        if llm is not None:
            ai_result = check_answer_with_ai(state)
            if ai_result is not None:
                return ai_result
            else:
                logger.warning("AI check failed, using fallback")
        
        # Use fallback
        return check_answer_fallback(state)
        
    except Exception as e:
        logger.error(f"Critical error in check_answer: {e}")
        logger.error(traceback.format_exc())
        return handle_invalid_state(state)

print(">>> COGNITIVE NODES ENHANCED FOR PRODUCTION...")

# --- 4. NEURAL PATHWAY CONSTRUCTION (LANGGRAPH) ---
def should_start_game(state: GameState):
    """Enhanced entry point decision with validation."""
    try:
        messages = state.get("messages", [])
        riddle_number = state.get("riddle_number", 0)
        
        logger.info(f"Entry decision: {len(messages)} messages, riddle_number: {riddle_number}")
        
        # If no messages or riddle_number is 0, start the game
        if len(messages) == 0 or riddle_number == 0:
            return "ask_riddle_node"
        
        # If we have messages, check the answer
        return "check_answer_node"
    except Exception as e:
        logger.error(f"Error in entry point decision: {e}")
        return "ask_riddle_node"  # Default to starting

# Build the workflow with error handling
try:
    workflow = StateGraph(GameState)
    workflow.add_node("ask_riddle_node", ask_first_riddle)
    workflow.add_node("check_answer_node", check_answer)
    workflow.set_conditional_entry_point(should_start_game)
    workflow.add_edge("ask_riddle_node", END)
    workflow.add_edge("check_answer_node", END)
    app_langgraph = workflow.compile()
    print(">>> NEURAL PATHWAYS COMPILED...")
except Exception as e:
    logger.error(f"Failed to compile workflow: {e}")
    sys.exit(1)

# --- 5. ENHANCED FLASK APPLICATION ---
app_flask = Flask(__name__)
CORS(app_flask, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure Flask for production
app_flask.config['JSON_SORT_KEYS'] = False
app_flask.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app_flask.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app_flask.after_request
def log_response(response):
    logger.info(f"Response: {response.status_code} for {request.path}")
    return response

@app_flask.route('/')
def serve_index():
    """Serves the main HTML page of the web app."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return "W.O.P.R. TERMINAL OFFLINE", 500

@app_flask.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with system diagnostics."""
    try:
        start_time = time.time()
        
        # Test basic system
        test_state = {"messages": [], "riddle_number": 0}
        result = app_langgraph.invoke(test_state)
        
        processing_time = time.time() - start_time
        
        health_data = {
            "status": "operational",
            "ai_available": llm is not None,
            "message": "W.O.P.R. SYSTEMS ONLINE",
            "riddles_total": len(riddles),
            "processing_time_ms": round(processing_time * 1000, 2),
            "system_load": "normal" if processing_time < 2.0 else "high"
        }
        
        # Test AI if available
        if llm is not None:
            try:
                # Quick AI test with timeout
                future = executor.submit(lambda: llm.invoke("ping"))
                future.result(timeout=5)
                health_data["ai_status"] = "responsive"
            except Exception:
                health_data["ai_status"] = "slow"
                health_data["ai_available"] = False
        
        return jsonify(health_data), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "degraded",
            "ai_available": False,
            "error": str(e)[:100],
            "message": "FALLBACK MODE ACTIVE"
        }), 500

@app_flask.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with comprehensive error handling."""
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] INCOMING TRANSMISSION...")
    
    start_time = time.time()
    
    try:
        # Validate request format
        if not request.is_json:
            logger.error(f"[{request_id}] Request is not JSON")
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if data is None:
            logger.error(f"[{request_id}] No JSON data received")
            return jsonify({"error": "Invalid JSON data"}), 400
        
        logger.info(f"[{request_id}] Processing state...")
        
        # Process and validate messages
        messages = []
        raw_messages = data.get('messages', [])
        
        if not isinstance(raw_messages, list):
            logger.error(f"[{request_id}] Messages is not a list")
            return jsonify({"error": "Messages must be a list"}), 400
        
        for i, m in enumerate(raw_messages):
            if not isinstance(m, dict):
                logger.warning(f"[{request_id}] Invalid message format at index {i}")
                continue
                
            msg_type = m.get('type', '').lower()
            content = str(m.get('content', '')).strip()
            
            if not content:
                logger.warning(f"[{request_id}] Empty message content at index {i}")
                continue
            
            # Limit message length
            if len(content) > 1000:
                content = content[:1000] + "..."
                logger.warning(f"[{request_id}] Truncated long message")
            
            if msg_type == 'human':
                messages.append(HumanMessage(content=content))
            elif msg_type == 'ai':
                messages.append(AIMessage(content=content))
            else:
                logger.warning(f"[{request_id}] Unknown message type: {msg_type}")

        # Validate riddle number
        riddle_number = data.get("riddle_number", 0)
        if not isinstance(riddle_number, int):
            try:
                riddle_number = int(riddle_number)
            except (ValueError, TypeError):
                logger.warning(f"[{request_id}] Invalid riddle_number, defaulting to 0")
                riddle_number = 0
        
        if riddle_number < 0 or riddle_number > len(riddles):
            logger.warning(f"[{request_id}] Riddle number {riddle_number} out of range")
            riddle_number = max(0, min(riddle_number, len(riddles)))

        current_state = {
            "messages": messages,
            "riddle_number": riddle_number
        }
        
        logger.info(f"[{request_id}] Executing graph: riddle={riddle_number}, msgs={len(messages)}")
        
        # Execute with timeout
        try:
            future = executor.submit(app_langgraph.invoke, current_state)
            response_from_graph = future.result(timeout=30)
        except FuturesTimeoutError:
            logger.error(f"[{request_id}] Graph execution timeout")
            return jsonify({
                'error': 'Request timeout',
                'message': 'SYSTEM OVERLOAD - PLEASE TRY AGAIN'
            }), 408
        
        if not response_from_graph:
            logger.error(f"[{request_id}] Empty graph response")
            return jsonify({
                'error': 'Empty response',
                'message': 'SYSTEM ERROR - PLEASE RETRY'
            }), 500
        
        # Serialize response safely
        serializable_messages = []
        for msg in response_from_graph.get('messages', []):
            try:
                if isinstance(msg, AIMessage):
                    msg_type = 'ai'
                elif isinstance(msg, HumanMessage):
                    msg_type = 'human'
                else:
                    msg_type = 'system'
                
                content = str(msg.content) if hasattr(msg, 'content') else str(msg)
                serializable_messages.append({
                    'type': msg_type, 
                    'content': content[:2000]  # Limit response size
                })
            except Exception as e:
                logger.warning(f"[{request_id}] Error serializing message: {e}")
                continue

        final_riddle_number = response_from_graph.get('riddle_number', riddle_number)
        
        json_response = {
            'messages': serializable_messages,
            'riddle_number': final_riddle_number,
            'processing_time': round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(f"[{request_id}] Success: riddle={final_riddle_number}, time={json_response['processing_time']}ms")
        return jsonify(json_response), 200
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Critical error after {processing_time:.1f}ms: {e}")
        logger.error(traceback.format_exc())
        
        # Return graceful error response
        fallback_riddle_number = 0
        try:
            if 'data' in locals() and data:
                fallback_riddle_number = max(0, int(data.get("riddle_number", 0)))
        except:
            pass
        
        return jsonify({
            'messages': [{
                'type': 'ai', 
                'content': 'SYSTEM ERROR DETECTED. W.O.P.R. ATTEMPTING RECOVERY... PLEASE TRY AGAIN.'
            }],
            'riddle_number': fallback_riddle_number,
            'error': 'Internal system error',
            'processing_time': round(processing_time, 2)
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

@app_flask.errorhandler(408)
def timeout_error(error):
    return jsonify({
        'error': 'Request timeout',
        'message': 'W.O.P.R. PROCESSING TIMEOUT - PLEASE RETRY'
    }), 408

# --- SYSTEM STARTUP ---
if __name__ == '__main__':
    print(">>> RUNNING COMPREHENSIVE SYSTEM DIAGNOSTICS...")
    
    # Comprehensive system test
    try:
        test_state = {"messages": [], "riddle_number": 0}
        test_result = app_langgraph.invoke(test_state)
        logger.info(f"✅ SYSTEM SELF-TEST PASSED")
        
        # Test AI if available
        if llm:
            try:
                test_ai = llm.invoke("System test")
                logger.info("✅ AI SUBSYSTEM OPERATIONAL")
            except Exception as e:
                logger.warning(f"⚠️ AI SUBSYSTEM WARNING: {e}")
                
    except Exception as e:
        logger.warning(f"⚠️ SYSTEM SELF-TEST WARNING: {e}")
        logger.warning("PROCEEDING WITH FALLBACK CAPABILITIES...")
    
    print(">>> COMMUNICATION INTERFACE ONLINE...")
    print("//==============================================================//")
    
    # Get port from environment
    port = int(os.environ.get('PORT', 5001))
    
    try:
        app_flask.run(
            host='0.0.0.0', 
            port=port, 
            debug=False,
            threaded=True,
            use_reloader=False
        )
    finally:
        cleanup_executor()


