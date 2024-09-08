# Chat with Lex RAG

This project implements a friendly chatbot capable of using Retrieval-Augmented Generation (RAG) to answer questions about specific episodes of the Lex Fridman podcast published on YouTube. The chatbot is implemented using user-friendly Streamlit interface and can also be run locally and free of charge by using Ollama for LLM generation. 

The full dataset for the Lex Fridman podcast is available in the folder 'JSON_storage'. Alternatively, you can obtain the dataset autonomously by checking and using the project [YouTube Channel Video Tracker](https://github.com/CharlieNestor/retrieve_video_info_YouTube_channel) and then following the steps in the files `create_video_db.ipynb` and `create_transcript_db.ipynb`.


## Features

- Local database with more than 400 Lex Fridman podcast episodes
- Interactive chat interface to explore podcast episodes via Streamlit
- RAG-based question answering system
- Support for MongoDB and local JSON storage
- Step-by-step guide on how to prepare the dataset


## Prerequisites

One among the following:
- Ollama installed locally with some models loaded ( suggested models are gemma2:2b and llama3.1:8b )
- OpenAI API key set up
- Anthropic API key set up

Depending on the chosen model provider, you need to specify the desired model for embedding and LLM generation. This is done in the `rag_single_video.py` file. In particular, check out the `define_vector_store` and `define_llm` functions.


## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/CharlieNestor/chat_with_Lex_RAG.git
   ```

2. Create your own virtual environment and install the dependencies:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add necessary environment variables, for example:
   ````
    MONGO_URI=mongodb://admin:password@localhost:27017/
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    ````

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