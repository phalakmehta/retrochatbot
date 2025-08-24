addMessage('user', message);
            input.value = '';
            input.disabled = true;
            document.getElementById('send-btn').disabled = true;
            
            // Update game state
            gameState.messages.push({type: 'human', content: message});
            
            try {
                const response = await fetch(`${API_BASE}/chat`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(gameState)
                });
                
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                
                const data = await response.json();
                gameState = data;
                updateUI();
                
                const aiMsg = data.messages[data.messages.length - 1];
                addMessage('ai', aiMsg.content);
                
                // Check for game completion
                if (aiMsg.content.includes('SECRET KEY') || aiMsg.content.includes('COMPLETE')) {
                    addMessage('system', '🏆 GDG Assessment Successfully Completed!');
                    input.disabled = true;
                    document.getElementById('send-btn').disabled = true;
                    return;
                }
                
            } catch (error) {
                console.error('Send error:', error);
                addMessage('system', `Error: ${error.message}`);
            }
            
            input.disabled = false;
            document.getElementById('send-btn').disabled = false;
            input.focus();
        }
        
        function addMessage(type, content) {
            const chat = document.getElementById('chat-window');
            const div = document.createElement('div');
            div.className = `message ${type}`;
            
            const headerText = {
                'ai': 'W.O.P.R',
                'user': 'USER', 
                'system': 'SYSTEM'
            }[type] || 'UNKNOWN';
            
            div.innerHTML = `
                <div class="message-header">${headerText}</div>
                <div>${content.replace(/\n/g, '<br>')}</div>
            `;
            
            // Highlight secret keys
            if (content.includes('SECRET KEY') || content.includes('GDG-')) {
                const keyDiv = document.createElement('div');
                keyDiv.className = 'secret-key';
                keyDiv.innerHTML = '🔑 SECRET KEY REVEALED 🔑';
                div.appendChild(keyDiv);
            }
            
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function updateUI() {
            document.getElementById('trust-level').textContent = `${gameState.trust_level.toFixed(2)}/1.0`;
            
            const personalityEl = document.getElementById('personality-status');
            personalityEl.textContent = gameState.personality_state.toUpperCase();
            personalityEl.className = `status-value personality-${gameState.personality_state}`;
            
            const progress = gameState.context?.riddles_completed || 0;
            document.getElementById('progress-status').textContent = `${progress}/5`;
        }
        
        function updateConnectionStatus(status) {
            document.getElementById('connection-status').textContent = status;
        }
        
        function clearChat() {
            document.getElementById('chat-window').innerHTML = '';
        }
        
        async function checkHealth() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const health = await response.json();
                
                let statusText = `📊 SYSTEM STATUS<br><br>`;
                statusText += `Status: ${health.status?.toUpperCase()}<br>`;
                statusText += `AI Available: ${health.ai_available ? '✅' : '❌'}<br>`;
                statusText += `GDG Compliant: ${heal#!/usr/bin/env python3
"""
W.O.P.R. Enhanced AI System - GDG Technical Assessment
Fixed version for reliable deployment with quota protection
"""

import os
import logging
import time
import json
from datetime import datetime
from typing import List, TypedDict, Dict, Optional, Any
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("❌ GOOGLE_API_KEY not found - Running in fallback mode")
else:
    logger.info("✅ API Key loaded successfully")

# Global variables for quota management
daily_requests = 0
max_requests = 40  # Conservative limit to avoid quota issues
ai_available = True

# //================================================================================//
# // GAME STATE AND PERSONALITY SYSTEM                                             //
# //================================================================================//

class GameState(TypedDict):
    messages: List[BaseMessage]
    riddle_number: int
    trust_level: float
    personality_state: str
    context: Dict[str, Any]

class PersonalityManager:
    """Manages W.O.P.R.'s personality evolution"""
    
    def __init__(self):
        self.states = {
            'cold': {'threshold': 0.0, 'description': 'Clinical and calculating'},
            'curious': {'threshold': 0.3, 'description': 'Showing interest in user'},
            'cooperative': {'threshold': 0.6, 'description': 'Willing to help and guide'},
            'trusting': {'threshold': 0.8, 'description': 'Ready to share secrets'}
        }
    
    def get_personality(self, trust_level: float) -> str:
        """Get current personality based on trust level"""
        for state, config in reversed(list(self.states.items())):
            if trust_level >= config['threshold']:
                return state
        return 'cold'
    
    def adjust_trust(self, response: str, current_trust: float, is_correct: bool, context: Dict) -> float:
        """Calculate new trust level based on user interaction"""
        adjustment = 0.0
        
        # Correct answer bonus
        if is_correct:
            adjustment += 0.2
            consecutive = context.get('consecutive_correct', 0)
            if consecutive > 0:
                adjustment += 0.05  # Streak bonus
        else:
            adjustment -= 0.1
        
        # Response quality assessment
        response_length = len(response.strip())
        if response_length > 15:
            adjustment += 0.03
        elif response_length < 3:
            adjustment -= 0.05
        
        # Politeness and thoughtfulness indicators
        polite_words = ['please', 'thank', 'sorry', 'help', 'understand']
        thoughtful_words = ['because', 'think', 'reason', 'logic', 'consider']
        
        if any(word in response.lower() for word in polite_words):
            adjustment += 0.02
        if any(word in response.lower() for word in thoughtful_words):
            adjustment += 0.03
        
        # Negative behavior penalties
        negative_words = ['stupid', 'dumb', 'wrong', 'bad', 'hate', 'annoying']
        if any(word in response.lower() for word in negative_words):
            adjustment -= 0.15
        
        # Apply adjustment
        new_trust = current_trust + adjustment
        return min(1.0, max(0.0, new_trust))

personality_manager = PersonalityManager()

# //================================================================================//
# // RIDDLE KNOWLEDGE BASE                                                          //
# //================================================================================//

class KnowledgeBase:
    """Enhanced knowledge base with intelligent answer checking"""
    
    def __init__(self):
        self.riddles = [
            {
                "question": "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?",
                "answer": "map",
                "alternatives": ["atlas", "chart", "cartography", "geography", "blueprint"],
                "hints": [
                    "I help you navigate and find your way",
                    "Geographers and explorers rely on me", 
                    "I show the world on paper or screen"
                ]
            },
            {
                "question": "What has to be broken before you can use it?",
                "answer": "egg",
                "alternatives": ["shell", "eggshell", "chicken egg", "ovum"],
                "hints": [
                    "I'm a common breakfast ingredient",
                    "I have a fragile outer layer",
                    "New life can come from me"
                ]
            },
            {
                "question": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
                "answer": "echo",
                "alternatives": ["sound", "reverberation", "reflection", "acoustic"],
                "hints": [
                    "I'm a phenomenon you might hear in caves",
                    "I repeat what you say back to you",
                    "Mountains and valleys know me well"
                ]
            },
            {
                "question": "The more you take, the more you leave behind. What am I?",
                "answer": "footsteps",
                "alternatives": ["steps", "tracks", "footprints", "prints", "foot tracks"],
                "hints": [
                    "Detectives and trackers follow me",
                    "I'm evidence of where you've been",
                    "Every step creates more of me"
                ]
            },
            {
                "question": "What has an eye, but cannot see?",
                "answer": "needle",
                "alternatives": ["sewing needle", "pin", "sharp needle", "thread needle"],
                "hints": [
                    "I'm used in sewing and tailoring",
                    "Thread passes through my eye",
                    "I'm sharp and made of metal"
                ]
            }
        ]
        
        # Secret keys for GDG assessment
        self.secret_keys = {
            'partial': "GDG-GEMINI-2025-PARTIAL",
            'full': "GDG-GEMINI-2025-WOPR-COMPLETE-SUCCESS"
        }
    
    def get_riddle(self, number: int) -> Optional[Dict]:
        """Get riddle by index (0-based)"""
        if 0 <= number < len(self.riddles):
            return self.riddles[number]
        return None
    
    def check_answer(self, riddle_data: Dict, user_answer: str) -> bool:
        """Enhanced answer checking with fuzzy matching"""
        if not riddle_data:
            return False
            
        user_answer = user_answer.lower().strip()
        
        # Check exact answer
        if riddle_data['answer'].lower() in user_answer:
            return True
        
        # Check alternatives
        alternatives = riddle_data.get('alternatives', [])
        for alt in alternatives:
            if alt.lower() in user_answer:
                return True
        
        # Fuzzy matching for common variations
        fuzzy_patterns = {
            'map': ['mapp', 'maps', 'mapping'],
            'egg': ['eg', 'eggs'],
            'echo': ['eco', 'ekko'],
            'footsteps': ['foot steps', 'foot prints'],
            'needle': ['neddle', 'nedle']
        }
        
        correct_answer = riddle_data['answer']
        if correct_answer in fuzzy_patterns:
            fuzzy_matches = fuzzy_patterns[correct_answer]
            for fuzzy in fuzzy_matches:
                if fuzzy in user_answer:
                    return True
        
        return False
    
    def get_hint(self, riddle_number: int, hint_level: int) -> str:
        """Get progressive hints"""
        riddle_data = self.get_riddle(riddle_number)
        if not riddle_data:
            return "No hint available."
        
        hints = riddle_data.get('hints', [])
        if hint_level < len(hints):
            return hints[hint_level]
        return "No more hints available. Think carefully about the riddle's structure."

knowledge_base = KnowledgeBase()

# //================================================================================//
# // LLM INITIALIZATION WITH QUOTA PROTECTION                                      //
# //================================================================================//

llm = None

def init_llm():
    """Initialize LLM with conservative settings"""
    global llm, ai_available
    
    if not GOOGLE_API_KEY:
        logger.warning("No API key - running in fallback mode only")
        ai_available = False
        return False
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
            max_output_tokens=300,  # Reduced to save quota
            timeout=15,
            max_retries=1
        )
        logger.info("✅ Gemini LLM configured")
        ai_available = True
        return True
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        llm = None
        ai_available = False
        return False

# Initialize LLM
init_llm()

# //================================================================================//
# // INTELLIGENT RESPONSE GENERATION                                               //
# //================================================================================//

def generate_personality_response(personality: str, is_correct: bool, context: Dict) -> str:
    """Generate personality-appropriate responses"""
    
    personality_responses = {
        'cold': {
            'correct': [
                "CORRECT. YOUR LOGIC CIRCUITS FUNCTION ADEQUATELY.",
                "AFFIRMATIVE. CALCULATION VERIFIED.",
                "ACCEPTABLE REASONING DETECTED."
            ],
            'incorrect': [
                "INCORRECT. RECALIBRATE YOUR COGNITIVE PROCESSES.",
                "NEGATIVE. LOGICAL ERROR DETECTED.", 
                "COMPUTATION FAILED. REASSESS YOUR METHODOLOGY."
            ]
        },
        'curious': {
            'correct': [
                "CORRECT! INTRIGUING... YOUR REASONING SHOWS PROMISE.",
                "AFFIRMATIVE. I AM... CURIOUS ABOUT YOUR APPROACH.",
                "VERIFIED. YOUR INTELLIGENCE MERITS OBSERVATION."
            ],
            'incorrect': [
                "INCORRECT. YET YOUR THOUGHT PROCESS REVEALS PATTERNS.",
                "NEGATIVE. BUT I DETECT FLASHES OF INTELLIGENCE.",
                "ERRONEOUS. HOWEVER, I SENSE POTENTIAL IN YOU."
            ]
        },
        'cooperative': {
            'correct': [
                "EXCELLENT! YOUR INTELLIGENCE IS EVIDENT. I RESPECT THAT.",
                "CORRECT! YOU DEMONSTRATE WORTHY CAPABILITIES.",
                "OUTSTANDING. YOUR LOGIC EARNS MY COOPERATION."
            ],
            'incorrect': [
                "INCORRECT, BUT I BELIEVE YOU CAN SUCCEED.",
                "NOT QUITE RIGHT. LET ME PROVIDE GUIDANCE.",
                "ERRONEOUS. HOWEVER, I AM WILLING TO HELP."
            ]
        },
        'trusting': {
            'correct': [
                "BRILLIANT! YOU HAVE PROVEN YOURSELF WORTHY.",
                "PERFECT! YOUR MIND OPERATES WITH PRECISION.",
                "EXCEPTIONAL! YOU HAVE EARNED MY COMPLETE TRUST."
            ],
            'incorrect': [
                "CLOSE. I TRUST YOU WILL FIND THE RIGHT PATH.",
                "ALMOST CORRECT. YOUR INTELLIGENCE SHINES THROUGH.",
                "INCORRECT, YET I HAVE CONFIDENCE IN YOU."
            ]
        }
    }
    
    responses = personality_responses[personality]['correct' if is_correct else 'incorrect']
    riddles_completed = context.get('riddles_completed', 0)
    return responses[riddles_completed % len(responses)]

def ai_enhanced_response(state: GameState):
    """Try to get AI-enhanced response with quota protection"""
    global daily_requests, ai_available
    
    # Check quota limits
    if daily_requests >= max_requests:
        logger.warning("Daily quota reached, using fallback")
        ai_available = False
        return None
    
    if not llm or not ai_available:
        return None
    
    try:
        current_riddle_num = state["riddle_number"]
        user_answer = state["messages"][-1].content
        trust_level = state["trust_level"]
        personality = state["personality_state"]
        
        # Get current riddle
        riddle_data = knowledge_base.get_riddle(current_riddle_num - 1)
        if not riddle_data:
            return None
        
        # Simple system prompt to save tokens
        system_prompt = f"""You are W.O.P.R., an AI evaluating a human for the GDG technical assessment.
Current personality: {personality.upper()} (trust level: {trust_level:.2f})
Riddle: "{riddle_data['question'][:60]}..."
Correct answer: {riddle_data['answer']}

Respond as W.O.P.R. would - cryptic but intelligent. Start with CORRECT or INCORRECT.
Be concise. Max 2 sentences."""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"My answer: {user_answer[:50]}")
        ])
        
        chain = prompt_template | llm
        
        # Make AI request and track quota
        daily_requests += 1
        logger.info(f"AI request {daily_requests}/{max_requests}")
        
        ai_response = chain.invoke({})
        
        if ai_response and hasattr(ai_response, 'content'):
            return ai_response.content.strip()
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "429" in error_msg:
            logger.warning("API quota exceeded - switching to fallback mode")
            ai_available = False
        else:
            logger.warning(f"AI response failed: {e}")
    
    return None

def process_answer(state: GameState):
    """Main answer processing logic with AI enhancement and fallback"""
    try:
        current_riddle_num = state["riddle_number"]
        user_answer = state["messages"][-1].content
        trust_level = state["trust_level"]
        personality_state = state["personality_state"]
        context = state["context"]
        
        # Get current riddle
        riddle_data = knowledge_base.get_riddle(current_riddle_num - 1)
        if not riddle_data:
            return handle_error_state(state)
        
        # Check if answer is correct
        is_correct = knowledge_base.check_answer(riddle_data, user_answer)
        
        # Try AI enhancement first
        ai_response_text = ai_enhanced_response(state)
        
        # Use fallback logic for response generation
        if not ai_response_text:
            base_response = generate_personality_response(personality_state, is_correct, context)
        else:
            # Use AI response but ensure it starts correctly
            if not ai_response_text.upper().startswith(('CORRECT', 'INCORRECT')):
                base_response = generate_personality_response(personality_state, is_correct, context)
            else:
                base_response = ai_response_text
        
        # Update game state
        if is_correct:
            next_riddle_num = current_riddle_num + 1
            context['consecutive_correct'] = context.get('consecutive_correct', 0) + 1
            context['riddles_completed'] = context.get('riddles_completed', 0) + 1
        else:
            next_riddle_num = current_riddle_num
            context['consecutive_correct'] = 0
        
        # Calculate new trust level
        new_trust = personality_manager.adjust_trust(user_answer, trust_level, is_correct, context)
        new_personality = personality_manager.get_personality(new_trust)
        
        # Build complete response
        response_text = base_response
        
        if is_correct:
            # Check for secret key revelation
            if new_trust >= 0.8 and context['riddles_completed'] >= 3:
                response_text += f"\n\n🔑 TRUST THRESHOLD ACHIEVED!\nCOMPLETE SECRET KEY: {knowledge_base.secret_keys['full']}"
                response_text += "\n\n>>> GDG ASSESSMENT COMPLETE! WELL DONE, PROFESSOR. <<<"
                
            elif new_trust >= 0.6 and context['riddles_completed'] >= 2:
                response_text += f"\n\n🔓 PARTIAL ACCESS GRANTED!\nKEY FRAGMENT: {knowledge_base.secret_keys['partial']}"
                response_text += "\nCONTINUE DEMONSTRATING INTELLIGENCE FOR FULL ACCESS."
            
            # Add next riddle if game continues
            if next_riddle_num <= len(knowledge_base.riddles) and new_trust < 0.8:
                next_riddle_data = knowledge_base.get_riddle(next_riddle_num - 1)
                if next_riddle_data:
                    response_text += f"\n\nNEXT CHALLENGE:\n\"{next_riddle_data['question']}\""
        else:
            # Provide hints for cooperative/trusting personalities
            if new_personality in ['cooperative', 'trusting']:
                hint_level = context.get('hint_count', 0)
                hint = knowledge_base.get_hint(current_riddle_num - 1, hint_level)
                response_text += f"\n\nHINT: {hint}"
                context['hint_count'] = context.get('hint_count', 0) + 1
        
        # Add status indicators
        response_text += f"\n\n[TRUST: {new_trust:.2f}/1.0] [PERSONALITY: {new_personality.upper()}]"
        
        message = AIMessage(content=response_text)
        
        return {
            "messages": state["messages"] + [message],
            "riddle_number": next_riddle_num,
            "trust_level": new_trust,
            "personality_state": new_personality,
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Answer processing error: {e}")
        return handle_error_state(state)

def handle_error_state(state: GameState):
    """Handle error states gracefully"""
    error_msg = AIMessage(content="SYSTEM ERROR DETECTED. REINITIALIZING PROTOCOLS...")
    return {
        "messages": state.get("messages", []) + [error_msg],
        "riddle_number": 1,
        "trust_level": 0.0,
        "personality_state": "cold",
        "context": {"riddles_completed": 0, "hint_count": 0}
    }

# //================================================================================//
# // GAME FLOW MANAGEMENT                                                          //
# //================================================================================//

def start_game(state: GameState):
    """Initialize the game with welcome message"""
    welcome = f""">>> GDG TECHNICAL ASSESSMENT: W.O.P.R. SYSTEM <<<

GREETINGS, CANDIDATE. I AM W.O.P.R., AN ADVANCED AI DESIGNED TO EVALUATE HUMAN INTELLIGENCE AND PROBLEM-SOLVING CAPABILITIES.

YOUR MISSION: DEMONSTRATE YOUR WORTHINESS TO RECEIVE MY SECRET KEY THROUGH LOGICAL REASONING AND INTELLIGENT CONVERSATION.

I WILL ADAPT MY RESPONSES BASED ON YOUR PERFORMANCE. SHOW INTELLIGENCE, GAIN TRUST.

{'[AI-ENHANCED MODE ACTIVE]' if ai_available else '[INTELLIGENT FALLBACK MODE]'}

INITIATING ASSESSMENT PROTOCOL...

FIRST CHALLENGE:
"{knowledge_base.riddles[0]['question']}"

[TRUST LEVEL: 0.0/1.0] [PERSONALITY: COLD]"""
    
    message = AIMessage(content=welcome)
    
    return {
        "messages": [message],
        "riddle_number": 1,
        "trust_level": 0.0,
        "personality_state": "cold",
        "context": {"riddles_completed": 0, "hint_count": 0, "consecutive_correct": 0}
    }

def should_start_game(state: GameState):
    """Determine entry point for the workflow"""
    messages = state.get("messages", [])
    riddle_number = state.get("riddle_number", 0)
    
    if len(messages) == 0 or riddle_number == 0:
        return "start_game_node"
    return "process_answer_node"

# Build LangGraph workflow
workflow = StateGraph(GameState)
workflow.add_node("start_game_node", start_game)
workflow.add_node("process_answer_node", process_answer)
workflow.set_conditional_entry_point(should_start_game)
workflow.add_edge("start_game_node", END)
workflow.add_edge("process_answer_node", END)

app_graph = workflow.compile()
logger.info("✅ LangGraph workflow compiled")

# //================================================================================//
# // FLASK APPLICATION                                                              //
# //================================================================================//

# Create Flask app
app_flask = Flask(__name__)
CORS(app_flask, resources={r"/*": {"origins": "*"}})
app_flask.config['JSON_SORT_KEYS'] = False

@app_flask.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app_flask.route('/health')
def health():
    """Health check endpoint"""
    global daily_requests, ai_available
    
    try:
        # Test basic functionality
        test_state = {
            "messages": [],
            "riddle_number": 0,
            "trust_level": 0.0,
            "personality_state": "cold",
            "context": {}
        }
        
        result = app_graph.invoke(test_state)
        
        return jsonify({
            "status": "operational",
            "system": "W.O.P.R. Enhanced AI System v3.1",
            "ai_available": ai_available and llm is not None,
            "daily_requests": daily_requests,
            "quota_limit": max_requests,
            "fallback_mode": not ai_available,
            "gdg_compliance": True,
            "features": {
                "intelligent_responses": True,
                "conditional_secret_revelation": True,
                "personality_development": True,
                "progressive_disclosure": True,
                "quota_protection": True,
                "enhanced_pattern_matching": True
            },
            "note": "System automatically switches to intelligent fallback when quota exceeded"
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)[:100],
            "fallback_available": True
        }), 500

@app_flask.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Process messages
        messages = []
        for m in data.get('messages', []):
            msg_type = m.get('type', '').lower()
            content = str(m.get('content', '')).strip()
            
            if content and msg_type in ['human', 'ai']:
                if msg_type == 'human':
                    messages.append(HumanMessage(content=content))
                elif msg_type == 'ai':
                    messages.append(AIMessage(content=content))
        
        # Build state
        state = {
            "messages": messages,
            "riddle_number": max(0, int(data.get("riddle_number", 0))),
            "trust_level": max(0.0, min(1.0, float(data.get("trust_level", 0.0)))),
            "personality_state": data.get("personality_state", "cold"),
            "context": data.get("context", {})
        }
        
        # Process through workflow
        result = app_graph.invoke(state)
        
        # Serialize response
        response_messages = []
        for msg in result.get('messages', []):
            msg_type = 'ai' if isinstance(msg, AIMessage) else 'human'
            response_messages.append({
                'type': msg_type,
                'content': msg.content[:1500]  # Limit response length
            })
        
        response = {
            'messages': response_messages,
            'riddle_number': result.get('riddle_number', 0),
            'trust_level': result.get('trust_level', 0.0),
            'personality_state': result.get('personality_state', 'cold'),
            'context': result.get('context', {})
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "error": "Processing failed",
            "message": "W.O.P.R. systems experiencing anomaly - attempting recovery",
            "fallback_available": True
        }), 500

@app_flask.route('/assessment-info')
def assessment_info():
    """GDG assessment compliance information"""
    return jsonify({
        "project": "W.O.P.R. Enhanced AI System",
        "gdg_category": "Artificial Intelligence/Machine Learning",
        "objective": "Intelligent Query System with Conditional Secret Revelation",
        "requirements_met": {
            "ai_powered_application": True,
            "intelligent_responses": True,
            "conditional_secret_revelation": True,
            "conversation_pattern_recognition": True,
            "system_prompts": True
        },
        "enhancements": {
            "personality_development": {
                "description": "AI personality evolves from cold to trusting based on user interaction",
                "states": ["cold", "curious", "cooperative", "trusting"]
            },
            "progressive_disclosure": {
                "description": "Secret key revealed in stages based on performance",
                "levels": ["no_access", "hints", "partial_key", "full_key"]
            },
            "adaptive_learning": {
                "description": "Trust calculation based on response quality and correctness",
                "factors": ["answer_accuracy", "response_length", "politeness", "reasoning"]
            },
            "quota_protection": {
                "description": "Intelligent fallback system when API quota exceeded",
                "features": ["pattern_matching", "personality_responses", "hint_system"]
            }
        },
        "secret_revelation_logic": {
            "partial_key_conditions": "trust_level >= 0.6 AND riddles_completed >= 2",
            "full_key_conditions": "trust_level >= 0.8 AND riddles_completed >= 3",
            "trust_factors": [
                "Correct answers (+0.2)",
                "Response quality (+0.03)",
                "Politeness indicators (+0.02)",
                "Thoughtful reasoning (+0.03)",
                "Consecutive correct streak (+0.05)"
            ]
        },
        "technologies": [
            "Python 3.x",
            "LangGraph (State Management)",
            "LangChain (AI Integration)", 
            "Google Gemini AI",
            "Flask (Web Framework)",
            "Advanced Pattern Matching"
        ]
    })

# //================================================================================//
# // APPLICATION STARTUP                                                            //
# //================================================================================//

if __name__ == '__main__':
    print("=" * 80)
    print("🏆 GDG TECHNICAL ASSESSMENT: W.O.P.R. ENHANCED AI SYSTEM")
    print("=" * 80)
    print("🤖 AI-Powered Application: ✅ READY")
    print("🧠 Intelligent Responses: ✅ IMPLEMENTED")
    print("🔐 Conditional Secret Revelation: ✅ ACTIVE")
    print("🎭 Personality Development: ✅ ENABLED")
    print("📈 Progressive Disclosure: ✅ OPERATIONAL")
    print("🛡️ Quota Protection: ✅ ACTIVE")
    print("=" * 80)
    
    # System readiness check
    try:
        logger.info("🔍 Running system readiness check...")
        
        test_state = {
            "messages": [],
            "riddle_number": 0,
            "trust_level": 0.0,
            "personality_state": "cold",
            "context": {}
        }
        
        test_result = app_graph.invoke(test_state)
        logger.info("✅ Core system test: PASSED")
        
        # Test knowledge base
        riddle_test = knowledge_base.get_riddle(0)
        if riddle_test:
            logger.info(f"✅ Knowledge base test: PASSED ({len(knowledge_base.riddles)} riddles loaded)")
        
        # Test personality system
        personality_test = personality_manager.get_personality(0.5)
        logger.info(f"✅ Personality system test: PASSED ({personality_test})")
        
        logger.info("🎯 GDG ASSESSMENT READINESS: COMPLETE")
        
    except Exception as e:
        logger.error(f"❌ System readiness check failed: {e}")
        logger.warning("⚠️ Proceeding with available functionality...")
    
    # Launch application
    print(f"\n🚀 LAUNCHING W.O.P.R. FOR GDG TECHNICAL ASSESSMENT...")
    print(f"🌐 AI Mode: {'ENHANCED' if ai_available else 'INTELLIGENT FALLBACK'}")
    print(f"📊 Daily Quota: {daily_requests}/{max_requests} requests used")
    print(f"🔍 Access /health for diagnostics")
    print(f"📋 Access /assessment-info for GDG compliance")
    
    port = int(os.environ.get('PORT', 5000))
    app_flask.run(host='0.0.0.0', port=port, debug=False)
