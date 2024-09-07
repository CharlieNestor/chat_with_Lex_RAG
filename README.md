# Chat with Lex RAG

This project implements a friendly chatbot capable of using Retrieval-Augmented Generation (RAG) to answer questions about specific episodes of the Lex Fridman podcast published on YouTube. The chatbot is implemented using user-friendly Streamlit interface and can also be run locally by using Ollama for LLM generation. 

The full dataset is available in the folder 'JSON_storage'. Alternatively, you can obtain the dataset autonomously by checking and using the project at [https://github.com/CharlieNestor/retrieve_video_info_YouTube_channel](https://github.com/CharlieNestor/retrieve_video_info_YouTube_channel) and then following the steps in the files `create_video_db.ipynb` and `create_transcript_db.ipynb`.


## Features

- Local database with some 400 podcast episodes
- Interactive chat interface to explore podcast episodes via Streamlit
- RAG-based question answering system
- Support for MongoDB and local JSON storage
- Step-by-step guide on how to prepare the dataset


## Prerequisites

- Ollama installed locally with some models loaded (suggested models are gemma2:2b and llama3.1:8b )
- OpenAI API key set up


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
   - Add necessary environment variables, for example:
   ````
    MONGO_URI=mongodb://admin:password@localhost:27017/
    OPENAI_API_KEY=your_openai_api_key
    ````

4. Run the Streamlit app:
   ```
   streamlit run rag_video_streamlit.py
   ```

2. Select a podcast episode from the sidebar
3. Click "Load Interview" to prepare the RAG system
4. Start chatting and asking questions about the selected episode


## Notes

This is still a work-in-progress project.