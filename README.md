# Chat with Lex RAG

This project implements a chatbot capable of using Retrieval-Augmented Generation (RAG) to answer questions about specific episodes of the Lex Fridman podcast.

## Features

- Interactive chat interface to explore podcast episodes
- RAG-based question answering system
- Support for MongoDB and local JSON storage
- Streamlit-based web application


## Prerequisites

- Ollama installed locally with some models loaded (suggested are gemma2:2b and llama3.1:8b )


## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/CharlieNestor/chat_with_Lex_RAG.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add necessary environment variables (e.g., MongoDB connection string)


## Usage

1. Run the Streamlit app:
   ```
   streamlit run rag_video_streamlit.py
   ```

2. Select a podcast episode from the sidebar
3. Click "Load Interview" to prepare the RAG system
4. Start chatting and asking questions about the selected episode


## Notes

This is still a work-in-progress project.