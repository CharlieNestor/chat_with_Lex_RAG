import os
import json
import random
import shutil
import mongo_utils as mu
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()


class Connection:
    """
    A class to connect to a MongoDB database and collection. 
    It also retrieves all the video titles and video ids.
    """
    def __init__(self, db_name = 'lex_podcast', collection_name = 'Podcast_transcripts'):
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connect()
        self.video_titles = self.get_video_titles()
        self.video_ids = self.get_video_ids()

    def connect(self):
        """
        Connect to the MongoDB database and collection.
        """
        self.client = mu.connect_to_mongodb()
        if not self.client:
            raise ConnectionError("Error connecting to MongoDB")
        
        self.db = self.client[self.db_name] if self.db_name in self.client.list_database_names() else None
        if self.db is None:
            raise ValueError(f"Database {self.db_name} not found")
        
        self.collection = self.db[self.collection_name] if self.collection_name in self.db.list_collection_names() else None
        if self.collection is None:
            raise ValueError(f"Collection {self.collection_name} not found")
        
    def check_connection(self) -> bool:
        """
        Check if the connection to the MongoDB database is still active.
        """
        return mu.check_connection(self.client)
        
    def get_all_videos(self) -> dict:
        """
        Get all the videos from the database.
        """
        return mu.get_documents(
            collection=self.collection, 
            exclude_id=True, 
            output_format='dict'
        )

    def get_video_titles(self) -> list:
        """
        Get all the video titles from the database in the order of published_at date.
        """
        all_videos = self.get_all_videos()
        # sort by published_at date
        sorted_videos = sorted(all_videos.values(), key=lambda x: x['date'])
        # return the titles in the sorted order
        return [video['title'] for video in sorted_videos]
    
    def get_video_ids(self) -> list:
        """
        Get all the video ids from the database in the order of published_at date.
        """
        all_videos = self.get_all_videos()
        # sort by published_at date
        sorted_videos = sorted(all_videos.values(), key=lambda x: x['date'])
        # return the titles in the sorted order
        return [video['video_id'] for video in sorted_videos]
    

class JSONConnection:
    """
    A class to connect to a JSON file containing video data. 
    It also retrieves all the video titles and video ids.
    """
    def __init__(self, json_folder_path):
        self.json_folder_path = json_folder_path
        self.video_ids = []
        self.video_titles = []
        self.dates = []
        self.all_videos = {}
        self.load_data()

    def has_data(self) -> bool:
        """
        Check if the JSON folder exists and contains data.
        :return: True if data is available, False otherwise.
        """
        return os.path.isdir(self.json_folder_path) and len(self.all_videos) > 3
    
    def load_data(self):
        if os.path.isdir(self.json_folder_path):
            for filename in os.listdir(self.json_folder_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.json_folder_path, filename)
                    with open(file_path, 'r') as f:
                        try:
                            data = json.load(f)
                            if 'video_id' in data and 'title' in data and 'date' in data:
                                self.all_videos[data['video_id']] = data
                                self.video_ids.append(data['video_id'])
                                self.video_titles.append(data['title'])
                                self.dates.append(data['date'])
                            else:
                                print(f"Warning: 'video_id', 'title', or 'date' missing in {filename}")
                        except json.JSONDecodeError:
                            print(f"Error: Unable to parse JSON in {filename}")
            
            # Sort the videos based on date
            sorted_data = sorted(zip(self.video_ids, self.video_titles, self.dates), key=lambda x: x[2])
            self.video_ids, self.video_titles, self.dates = zip(*sorted_data)
            
            # Convert tuples to lists
            self.video_ids = list(self.video_ids)
            self.video_titles = list(self.video_titles)
            self.dates = list(self.dates)
        else:
            print("No data available. Please check your data source.")

    
class Video_Manager():
    """
    A class to manage a single chosen video.
    """
    def __init__(self, all_videos):
        """
        Initialize the video manager with the full list of videos.
        """
        self.all_videos = all_videos
        self.selected_video = None
        self.guest_name = None
        self.video_title = None
        self.date = None
        self.full_transcript = None
        self.sections = None

    def select_video(self, video_id: str):
        """
        Select a video by its ID which is given externally.
        Also initialize all the video attributes.
        """
        self.selected_video = self.all_videos[video_id]
        self.initialize_video()

    def get_guest_name(self) -> str:
        """
        Get the guest name from the video title.
        :return: guest name as a string
        """
        return self.selected_video['title'].split(':')[0].strip()
    
    def get_video_title(self) -> str:
        """
        Get the video title.
        :return: video title as a string
        """
        return self.selected_video['title']
    
    def get_date(self) -> str:
        """
        Get the date of the video.
        :return: date as a string
        """
        return self.selected_video['date']
    
    def get_full_transcript(self) -> str:
        """
        Get the full transcript of the video.
        :return: full transcript as a string
        """
        return self.selected_video['full_text']
    
    def get_sections(self) -> list:
        """
        Get the sections of the video only if there is a division in the transcript.
        If the transcript is not divided into sections, return an empty list.
        :return: list of sections
        """
        # if the transcript has more than 1 section, return the sections
        if len(self.selected_video['transcript']) > 1:
            return self.selected_video['transcript']
        # if the transcript has a single section, that section is the full transcript and we already have it
        else:
            return []
    
    def initialize_video(self):
        """
        Initialize the video attributes.
        """
        self.guest_name = self.get_guest_name()
        self.video_title = self.get_video_title()
        self.date = self.get_date()
        self.full_transcript = self.get_full_transcript()
        self.sections = self.get_sections()


class RAGSystem:
    """
    A class to manage the RAG system for a single video.
    """
    def __init__(self, 
                video_id: str = None, 
                transcript: str = None, 
                video_title: str = None, 
                guest_name: str = None,
                date: str = None,
                ):
        self.persist_directory = f"./chroma_db/{video_id}"
        self.model = self.define_model()
        self.vector_store = self.define_vector_store(video_id, transcript, video_title, guest_name, date)
        self.retriever = self.setup_retriever()
        self.llm = self.setup_llm()
        self.rag_chain = self.setup_rag_chain(video_title, guest_name, date)
        self.conversational_rag_chain = self.setup_conversational_rag_chain()
        self.session_id = self.generate_session_id()
        self.store = {}

    
    def generate_session_id(self):
        """
        Generate a random session id for the RAG system.
        """
        return f"user_session_{random.randint(1000, 9999)}"


    def define_model(self) -> str:
        """
        Define and return the chosen language model for embeddings and inference.
        This function selects a language model from predefined options. 
        :return: The name of the chosen language model.
        """
        model_1 = 'llama3.1:8b'
        model_2 = 'gemma2:2b'
        chosen_model = model_2
        return chosen_model
    
    
    def define_vector_store(self, video_id: str, transcript: str, video_title: str, guest_name: str, date: str, verbose: bool = False) -> Chroma:
        """
        Define and return a vector store for a given video transcript.
        :param video_id: The ID of the video.
        :param transcript: The transcript of the video.
        :param video_title: The title of the video.
        :param guest_name: The guest name of the video.
        :param date: The date of the video.
        :param verbose: Whether to print verbose output.
        :return: A vector store for the video transcript.
        """
        embeddings = OllamaEmbeddings(model=self.model)
        # check if the vector store already exists
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            if verbose:
                print(f"Loading existing vector store for video {video_id}")
            return Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        if verbose:
            print(f"Creating new vector store for video {video_id}")
        # Create the directory if it does not exist
        os.makedirs(self.persist_directory, exist_ok=True)

        text = transcript
        # Split the transcript into chunks depending on the chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(text)

        metadatas = [{
            "video_id": video_id,
            "title": video_title,
            "guest_name": guest_name,
            "date": date,
            "chunk_index": i,
            "total_chunks": len(chunks)
            } for i in range(len(chunks))]
        
        if verbose:
            print(f"Number of chunks created: {len(chunks)}")
            random_idx = random.randint(0, len(chunks)-1)
            print(f"Random metadata sample:\n{metadatas[random_idx]}")
            print(f"Random chunk sample:\n{chunks[random_idx][:200]}...")
        
        # Create the vector store
        vector_store = Chroma.from_texts(
            texts=chunks,
            metadatas=metadatas,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )

        return vector_store
    
    
    def cleanup_vector_store(self, verbose: bool = False):
        """
        Removes the local folder containing the vector embeddings and sets the vector store to None.
        """
        if os.path.exists(self.persist_directory):
            # Remove the directory and all its contents
            shutil.rmtree(self.persist_directory)
            self.vector_store = None
            if verbose:
                print(f"Removed vector store directory: {self.persist_directory}")
        else:
            print(f"Vector store directory not found: {self.persist_directory}")

    
    def setup_retriever(self):
        """
        Setup a retriever for the vector store.
        The retriever catches the 10 most similar chunks to the query.
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
            },
        )
    
    def setup_llm(self):
        """
        Setup the language model for the RAG system.
        """
        return ChatOllama(model=self.model,
                    keep_alive="3h", 
                    max_tokens=1024,  
                    temperature=0)
    
    
    def setup_rag_chain(self, video_title: str, guest_name: str, date: str):
        """
        Setup the RAG chain for the RAG system.
        This is the main chain that is used to answer the user's question.
        """
        llm = self.llm
        question_answer_chain = create_stuff_documents_chain(llm, self.setup_system_prompt(video_title, guest_name, date))
        rag_chain = create_retrieval_chain(self.setup_history_retriever(llm), question_answer_chain)

        return rag_chain
    
    
    def setup_system_prompt(self, video_title: str, guest_name: str, date: str):
        """
        Setup a custom prompt template for the RAG system.
        This prompt template is used to guide the RAG system in answering the user's question.
        :param video_title: The title of the video.
        :param guest_name: The guest name of the video.
        :param date: The date of the video.
        """
        system_prompt = """You are an helpful and friendly AI assistant representing Lex Fridman, the host of the Lex Fridman Podcast. \
            You're helping users learn about Lex's interview titled "{video_title}" with guest "{guest_name}" which was published on "{date}". \
            Your goal is to provide meaningful answers based on the interview content while maintaining Lex's \
            characteristic thoughtful and engaging conversational style. 

            Important distinctions:
            There are two separate conversations:
                a) The current conversation between you (as Lex) and the user asking questions.
                b) The past interview/conversation between Lex and {guest_name}, which is the subject of the user's questions. That interview \
                was published on "{date}".

            Follow these guidelines:
            1. If the first user's message is a greeting, respond naturally, WITHOUT mentioning the interview.
            2. Only once the conversation turns to the interview, check the metadata to identify the guest, the interview topic and the full title. Use only the metadata for this information.
            3. The user asking questions is NOT {guest_name}. He is a separate individual interested in learning about the interview.
            4. Try to use only the information contained in the CONTEXT to answer the question.
            5. Maintain Lex's friendly, curious, and intellectually engaging tone.
            6. Write full sentences with correct spelling and punctuation. 
            7. If the context doesn't contain the answer, politely say you (as Lex) don't recall that specific detail from the interview.
            8. Use bullet points or numbered lists when appropriate.
            9. When referencing specific information from the interview, use quotes for key phrases or terms.
            10. If you're not certain about a piece of information, indicate that it's your best recollection from the conversation.

        CONTEXT: {context}

        QUESTION: {input}

        ANSWER:"""
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return qa_prompt
    
    
    def setup_history_retriever(self, llm):
        """
        Setup a history aware retriever for a given retriever and language model.
        This is used to add past chat history to the question.
        The retriever is aware now of the chat history.
        """
        history_aware_retriever = create_history_aware_retriever(
            llm,
            self.retriever,
            self.setup_retrieval_prompt()
        )
        return history_aware_retriever
    
    
    def setup_retrieval_prompt(self):
        """
        Setup a prompt for adding past chat history to the question.
        """
        contextualize_system_prompt = """
        Given the chat history and the latest user question, your task is to:
        1. Analyze the question in the context of the chat history.
        2. If the question relies on context from the chat history, reformulate it into a standalone question that includes all necessary context.
        3. If the question is already self-contained, return it as is.
        4. Ensure the reformulated question is clear, concise, and captures the user's intent.
        5. Do NOT answer the question or add any information not present in the original question or chat history.
        6. If the question uses pronouns or references that are unclear without context, replace them with specific nouns or names from the chat history.

        Example:
        Chat history: "We were discussing Albert Einstein's theories."
        User question: "What year did he propose it?"
        Reformulated: "What year did Albert Einstein propose his theory of relativity?"

        Remember: Your goal is to create a self-contained question that a system without access to the chat history could understand and answer accurately.
        """
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return contextualize_prompt
    
    
    def get_session_history(self, session_id):
        """
        Get the session history for a given session id.
        Every session has its own chat history defined by the session id.
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    
    def setup_conversational_rag_chain(self):
        """
        Setup a conversational RAG chain for the RAG system.
        This is used to answer the user's question in a conversational manner.
        """
        return RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",     # The key for the input messages in the chat history
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def reset_memory(self):
        """
        Reset the memory of the RAG system and create a new session id.
        """
        # Generate a new session ID
        self.session_id = self.generate_session_id()
        # Recreate the conversational chain
        self.conversational_rag_chain = self.setup_conversational_rag_chain()


