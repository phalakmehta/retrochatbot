import requests
import json

# The URL where your Flask app is running
URL = "http://127.0.0.1:5001/chat"


def main():
    """Runs the main game loop for the W.O.P.R. chatbot."""
    print("--- Connecting to W.O.P.R. ---")

    # --- Start the game ---
    try:
        initial_state = {"messages": [], "riddle_number": 0}
        response = requests.post(URL, json=initial_state)
        response.raise_for_status()  # Raise an exception for bad status codes
        game_state = response.json()
    except requests.exceptions.RequestException as e:
        print("\n[!] CONNECTION FAILED.")
        print("Please ensure the 'app.py' server is running in a separate terminal.")
        print(f"Error details: {e}")
        return

    # Print the AI's first message
    ai_message = game_state.get('messages', [{}])[-1].get('content', "Error: No message received.")
    print(f"\nW.O.P.R: {ai_message}\n")

    # --- Game Loop ---
    while True:
        user_input = input("Your Answer: ")

        if user_input.lower() in ["quit", "exit"]:
            print("--- TERMINATING CONNECTION ---")
            break

        # Extract the relevant parts of the state to send back
        messages = game_state.get('messages', [])
        riddle_number = game_state.get('riddle_number', 0)

        # Add our new answer
        messages.append({'type': 'human', 'content': user_input})

        # Create the payload for the next request
        payload = {
            "messages": messages,
            "riddle_number": riddle_number
        }

        # Send the request and get the new state
        try:
            response = requests.post(URL, json=payload)
            response.raise_for_status()
            game_state = response.json()

            ai_message = game_state.get('messages', [{}])[-1].get('content', "Error: No message received.")
            print(f"\nW.O.P.R: {ai_message}\n")

            # Check if the game is over
            if "SECRET KEY" in ai_message:
                print("--- CONNECTION TERMINATED BY HOST ---")
                break
        except requests.exceptions.RequestException as e:
            print(f"\n[!] CONNECTION LOST. Error: {e}")
            break


if __name__ == '__main__':
    main()
