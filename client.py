import requests

# Local Flask server endpoint
URL = "http://localhost:5001/chat"


def main():
    """Runs the main game loop for the W.O.P.R. chatbot."""
    print("--- Connecting to W.O.P.R. ---")

    # --- Start the game ---
    try:
        initial_state = {"messages": [], "riddle_number": 0}
        response = requests.post(URL, json=initial_state, timeout=10)
        response.raise_for_status()
        game_state = response.json()
    except requests.exceptions.RequestException as e:
        print("\n[!] CONNECTION FAILED.")
        print("Make sure your Flask app (server) is running on localhost:5001")
        print(f"Error details: {e}")
        return

    # Print the AI's first message
    ai_message = game_state.get('messages', [{}])[-1].get('content', "Error: No message received.")
    print(f"\nW.O.P.R: {ai_message}\n")

    # --- Game Loop ---
    while True:
        try:
            user_input = input("Your Answer: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n--- TERMINATING CONNECTION ---")
            break

        if user_input.lower() in ["quit", "exit"]:
            print("--- TERMINATING CONNECTION ---")
            break

        # Extract the relevant parts of the state
        messages = game_state.get('messages', [])
        riddle_number = game_state.get('riddle_number', 0)

        # Append user input
        messages.append({'type': 'human', 'content': user_input})

        payload = {
            "messages": messages,
            "riddle_number": riddle_number
        }

        try:
            response = requests.post(URL, json=payload, timeout=10)
            response.raise_for_status()
            game_state = response.json()

            ai_message = game_state.get('messages', [{}])[-1].get('content', "Error: No message received.")
            print(f"\nW.O.P.R: {ai_message}\n")

            # Check if secret is revealed → terminate
            if "SECRET KEY" in ai_message:
                print("--- CONNECTION TERMINATED BY HOST ---")
                break
        except requests.exceptions.RequestException as e:
            print(f"\n[!] CONNECTION LOST. Error: {e}")
            break


if __name__ == '__main__':
    main()

