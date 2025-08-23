# GDG Technical Assessment: Project W.O.P.R.

## Description

Project W.O.P.R. (Well-Ordered Operant Puzzle Responder) is an intelligent query system developed for the Google Developer Group technical assessment. This project is a stateful conversational agent with a "retro-cyber" theme, inspired by the cryptic, Cold War-era AI from the movie *WarGames*.

The application challenges the user to a game of wits. The AI, W.O.P.R., holds a secret key that is only revealed upon the successful completion of a sequence of riddles.

## Features

* **Stateful Conversation:** Built with LangGraph to remember the user's progress and manage a multi-step dialogue.
* **Conditional Logic:** The AI's path changes based on whether the user's answers are correct or incorrect.
* **Thematic Persona:** The AI embodies the cryptic, logical W.O.P.R. character for an immersive retro-cyber experience.
* **Gated Secret Key:** The key is not hardcoded but is the final reward for completing the logic puzzle.

## Technology Stack

* Language: Python
* Core Libraries: LangChain, LangGraph
* AI Model: Google Gemini (`gemini-1.5-flash`)
* Environment: Google Colab / Jupyter Notebook

## Setup and Installation

1.  **Get API Key:** Obtain a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  **Set Up Notebook:** Open the `chatbot.ipynb` notebook in Google Colab.
3.  **Store API Key:** Add your API key to the Colab Secrets manager (🔑 icon) with the name `GOOGLE_API_KEY`.
4.  **Install Dependencies:** Run the first code cell to install all required libraries.
5.  **Run All Cells:** Execute the cells in the notebook from top to bottom.

## How to Play

1.  Run the "Start the Game" cell to receive the initial greeting and the first riddle from W.O.P.R.
2.  In the final cell, update the `my_answer` variable with your solution.
3.  Run the final cell to submit your answer. The AI will respond.
4.  Repeat step 2 and 3 for each riddle until the game is complete.

## Secret Key Discovery Methodology

The secret key is revealed by the AI itself as the final reward. To trigger the reveal, the user must engage the W.O.P.R. in its logic game and successfully answer all three of its riddles in the correct sequence.
