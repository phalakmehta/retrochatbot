import requests
import json
import time
from typing import Optional, Dict, Any

# Configuration
BASE_URL = "https://retrochatbot-production.up.railway.app"
CHAT_URL = f"{BASE_URL}/chat"
HEALTH_URL = f"{BASE_URL}/health"

class WOPRClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
        self.game_state = {"messages": [], "riddle_number": 0}
        self.max_retries = 3
        
    def check_system_health(self) -> bool:
        """Check if the W.O.P.R. system is operational."""
        try:
            response = self.session.get(HEALTH_URL, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ System Status: {health_data.get('message', 'ONLINE')}")
                if not health_data.get('ai_available', False):
                    print("⚠️ AI Mode: FALLBACK (Limited functionality)")
                return True
            else:
                print(f"⚠️ System Status: HTTP {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"❌ Health Check Failed: {e}")
            return False

    def send_request(self, payload: Dict[Any, Any], retry_count: int = 0) -> Optional[Dict]:
        """Send request to W.O.P.R. with retry logic."""
        try:
            response = self.session.post(
                CHAT_URL, 
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            print(f"⏰ Connection timeout (attempt {retry_count + 1})")
            if retry_count < self.max_retries:
                print(f"Retrying in {2 ** retry_count} seconds...")
                time.sleep(2 ** retry_count)
                return self.send_request(payload, retry_count + 1)
            return None
            
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection failed (attempt {retry_count + 1})")
            if retry_count < self.max_retries:
                print(f"Retrying in {2 ** retry_count} seconds...")
                time.sleep(2 ** retry_count)
                return self.send_request(payload, retry_count + 1)
            return None
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            if e.response.status_code == 500:
                try:
                    error_data = e.response.json()
                    print(f"Server Error: {error_data.get('message', 'Unknown error')}")
                except:
                    pass
            return None
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def start_game(self) -> bool:
        """Initialize the game and get the first riddle."""
        print("🚀 Initializing W.O.P.R. connection...")
        
        initial_state = {"messages": [], "riddle_number": 0}
        response = self.send_request(initial_state)
        
        if response:
            self.game_state = response
            ai_message = self.game_state.get('messages', [{}])[-1].get('content', "Error: No message received.")
            print(f"\nW.O.P.R: {ai_message}\n")
            return True
        else:
            print("\n[!] FAILED TO ESTABLISH CONNECTION WITH W.O.P.R.")
            return False

    def send_answer(self, user_input: str) -> bool:
        """Send user's answer and get response."""
        # Add user message to state
        messages = self.game_state.get('messages', [])
        riddle_number = self.game_state.get('riddle_number', 0)
        
        messages.append({'type': 'human', 'content': user_input})
        
        payload = {
            "messages": messages,
            "riddle_number": riddle_number
        }
        
        response = self.send_request(payload)
        
        if response:
            self.game_state = response
            ai_message = self.game_state.get('messages', [{}])[-1].get('content', "Error: No message received.")
            print(f"\nW.O.P.R: {ai_message}\n")
            
            # Check if game is complete
            if "SECRET KEY" in ai_message:
                return False  # Game over
            return True  # Continue game
        else:
            print("\n[!] CONNECTION LOST TO W.O.P.R. SYSTEM")
            return False

def main():
    """Main game loop for the W.O.P.R. chatbot."""
    print("=" * 60)
    print("🎮 W.O.P.R. TERMINAL CLIENT v2.0")
    print("=" * 60)
    
    client = WOPRClient()
    
    # Check system health first
    print("🔍 Checking W.O.P.R. system status...")
    if not client.check_system_health():
        print("\n⚠️ System may be experiencing issues, but attempting connection...")
    
    # Start the game
    if not client.start_game():
        print("\n💥 FATAL ERROR: Could not establish connection")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Verify the server is running on Railway")
        print("3. Try again in a few minutes")
        return
    
    print("🎯 Game Commands:")
    print("   - Type your answer to each riddle")
    print("   - Type 'quit' or 'exit' to terminate")
    print("   - Type 'health' to check system status")
    print("-" * 40)
    
    # Main game loop
    while True:
        try:
            user_input = input("Your Answer: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 TERMINATING CONNECTION TO W.O.P.R...")
                break
            
            if user_input.lower() == "health":
                client.check_system_health()
                continue
            
            if not user_input:
                print("⚠️ Please enter an answer")
                continue
            
            # Send answer and continue if game is still active
            if not client.send_answer(user_input):
                print("🏁 CONNECTION TERMINATED BY W.O.P.R. SYSTEM")
                break
                
        except KeyboardInterrupt:
            print("\n\n⚡ EMERGENCY SHUTDOWN INITIATED")
            print("👋 CONNECTION TO W.O.P.R. TERMINATED")
            break
        except Exception as e:
            print(f"\n❌ Unexpected client error: {e}")
            break
    
    print("\n" + "=" * 60)
    print("📋 SESSION COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()

