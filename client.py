#!/usr/bin/env python3
"""
W.O.P.R. Terminal Client - GDG Technical Assessment
Simple, reliable client for testing the AI system
"""

import requests
import json
import time
from typing import Dict, Any

class WOPRClient:
    def __init__(self, base_url="https://retrochatbot-production.up.railway.app"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        self.game_state = {
            "messages": [],
            "riddle_number": 0,
            "trust_level": 0.0,
            "personality_state": "cold",
            "context": {}
        }
    
    def check_health(self):
        """Check if W.O.P.R. system is operational"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ System Status: {health.get('status', 'Unknown').upper()}")
                print(f"   AI Available: {'✅' if health.get('ai_available') else '❌'}")
                print(f"   GDG Compliant: {'✅' if health.get('gdg_compliance') else '❌'}")
                return True
            else:
                print(f"⚠️ Health check returned: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to W.O.P.R. with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                print(f"⏰ Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
            except requests.exceptions.ConnectionError:
                print(f"🔌 Connection failed (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
            except requests.exceptions.HTTPError as e:
                print(f"❌ HTTP Error: {e}")
                if e.response.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        return None
    
    def start_game(self):
        """Initialize the game"""
        print("🚀 Initializing connection to W.O.P.R...")
        
        response = self.send_request(self.game_state)
        if response:
            self.game_state = response
            ai_message = self.get_last_ai_message()
            if ai_message:
                print(f"\n🤖 W.O.P.R: {ai_message}\n")
                self.display_status()
                return True
        
        print("❌ Failed to establish connection with W.O.P.R.")
        return False
    
    def send_answer(self, user_input: str):
        """Send user answer and get response"""
        # Add user message to state
        self.game_state["messages"].append({
            "type": "human",
            "content": user_input
        })
        
        response = self.send_request(self.game_state)
        if response:
            self.game_state = response
            ai_message = self.get_last_ai_message()
            if ai_message:
                print(f"\n🤖 W.O.P.R: {ai_message}")
                self.display_status()
                
                # Check for game completion
                if "SECRET KEY" in ai_message or "COMPLETE" in ai_message:
                    print("\n🏆 GDG ASSESSMENT COMPLETED!")
                    return False  # Game over
                
                return True  # Continue game
        
        print("❌ Communication with W.O.P.R. failed")
        return False
    
    def get_last_ai_message(self):
        """Get the last AI message from state"""
        messages = self.game_state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("type") == "ai":
                return msg.get("content", "")
        return None
    
    def display_status(self):
        """Display current game status"""
        trust = self.game_state.get("trust_level", 0.0)
        personality = self.game_state.get("personality_state", "cold").upper()
        riddles_completed = self.game_state.get("context", {}).get("riddles_completed", 0)
        
        # Create trust bar
        trust_bar = "█" * int(trust * 10) + "░" * (10 - int(trust * 10))
        
        print(f"\n📊 Status: Trust [{trust_bar}] {trust:.2f}/1.0 | Personality: {personality} | Progress: {riddles_completed}/5")
    
    def show_assessment_info(self):
        """Show GDG assessment information"""
        try:
            response = self.session.get(f"{self.base_url}/assessment-info", timeout=10)
            if response.ok:
                info = response.json()
                print("\n📋 GDG TECHNICAL ASSESSMENT INFO:")
                print(f"   Project: {info.get('project')}")
                print(f"   Category: {info.get('gdg_category')}")
                print(f"   Objective: {info.get('objective')}")
                print("\n✅ Requirements Met:")
                for req, met in info.get('requirements_met', {}).items():
                    status = "✅" if met else "❌"
                    print(f"   {status} {req.replace('_', ' ').title()}")
                print()
        except Exception as e:
            print(f"❌ Could not load assessment info: {e}")

def main():
    """Main game loop"""
    print("=" * 70)
    print("🎮 W.O.P.R. TERMINAL CLIENT")
    print("🏆 GDG Technical Assessment - AI Challenge")
    print("=" * 70)
    
    # Allow custom URL for testing
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://retrochatbot-production.up.railway.app"
    
    client = WOPRClient(base_url)
    
    # System checks
    print("🔍 Checking W.O.P.R. system status...")
    if not client.check_health():
        print("⚠️ System may have issues, but attempting to connect...")
    
    print()
    
    # Start game
    if not client.start_game():
        print("\n💥 CRITICAL ERROR: Could not establish connection")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the server URL is correct")
        print("3. Try again in a few minutes")
        return
    
    print("\n💡 Commands:")
    print("   • Type your answers to riddles")
    print("   • 'status' - Show current progress")
    print("   • 'info' - Show GDG assessment details")
    print("   • 'health' - Check system status") 
    print("   • 'quit' - Exit the program")
    print("-" * 50)
    
    # Main game loop
    while True:
        try:
            user_input = input("\n👤 Your Response: ").strip()
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Disconnecting from W.O.P.R...")
                break
            
            if user_input.lower() == 'status':
                client.display_status()
                continue
            
            if user_input.lower() == 'info':
                client.show_assessment_info()
                continue
            
            if user_input.lower() == 'health':
                client.check_health()
                continue
            
            if user_input.lower() == 'help':
                print("\n📖 Help:")
                print("   Answer each riddle thoughtfully to build trust with W.O.P.R.")
                print("   The AI's personality will evolve based on your responses.")
                print("   Reach high trust levels to unlock secret key fragments.")
                continue
            
            if not user_input:
                print("⚠️ Please enter a response")
                continue
            
            # Send answer
            if not client.send_answer(user_input):
                print("\n🎯 Session completed!")
                break
                
        except KeyboardInterrupt:
            print("\n\n⚡ Emergency shutdown")
            print("👋 Connection to W.O.P.R. terminated")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            break
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    print(f"   Trust Level: {client.game_state.get('trust_level', 0.0):.2f}/1.0")
    print(f"   Personality: {client.game_state.get('personality_state', 'unknown').upper()}")
    print(f"   Riddles Completed: {client.game_state.get('context', {}).get('riddles_completed', 0)}")
    print("\n🎓 Thank you for participating in the GDG Technical Assessment!")

if __name__ == '__main__':
    main()

