import time
import streamlit as st
from rag_single_video import Connection, Video_Manager, RAGSystem, JSONConnection

# Initialize Connection (should be done only once)
@st.cache_resource
def get_connection():
    try:
        return Connection()
    except Exception as e:
        return None
# Initialize JSON Connection
@st.cache_resource
def get_json_connection():
    json_folder_path = 'JSON_storage'
    return JSONConnection(json_folder_path)

connection = get_connection()

# Check if the connection message has been shown before
if "connection_message_shown" not in st.session_state:
    st.session_state.connection_message_shown = False

if connection is None and not st.session_state.connection_message_shown:
    st.warning("Unable to connect to MongoDB. Using local JSON storage instead.", icon="ℹ️")
    st.session_state.connection_message_shown = True

# Streamlit app
st.title("🎙️ Lex Fridman Podcast Explorer 🧠")

# Sidebar for video selection
with st.sidebar:

    if connection:
        # The connection with MongoDB is successful
        video_titles = connection.video_titles
        selected_title = st.selectbox("Select a podcast episode:", video_titles)
        selected_video_id = connection.video_ids[video_titles.index(selected_title)]
        all_videos = connection.get_all_videos()
    else:
        # Fallback to JSON storage
        json_connection = get_json_connection()
        if json_connection.has_data():  
            video_titles = json_connection.video_titles
            selected_title = st.selectbox("Select a podcast episode:", video_titles)
            selected_video_id = json_connection.video_ids[video_titles.index(selected_title)]
            all_videos = json_connection.all_videos
        else:
            st.error("No data available. Please check your connection or data source.")
            # Stop the app
            st.stop()

    # Initialize Video Manager
    video_manager = Video_Manager(all_videos)
    video_manager.select_video(selected_video_id)

    st.subheader("Episode Info")
    st.write(f"🎙️ **Guest:** {video_manager.guest_name}")
    st.write(f"📅 **Date:** {video_manager.date}")

    # Add "Load Interview" button
    if st.button("Load Interview"):
        with st.spinner("Preparing the interview... This might take a moment! 🧠"):
            st.session_state.rag_system = RAGSystem(
                video_id=selected_video_id,
                transcript=video_manager.full_transcript,
                video_title=video_manager.video_title,
                guest_name=video_manager.guest_name,
                date=video_manager.date
            )
        st.success("Interview loaded successfully! You can now start chatting.")

    # Add "Clear Chat" button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        if st.session_state.get("rag_system") is not None:
            st.session_state.rag_system.reset_memory()
        st.success("Chat cleared. You can start a new conversation!")

# Main chat interface
st.subheader(f"💬 {video_manager.video_title}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat history when video changes
if "current_video_id" not in st.session_state or st.session_state.current_video_id != selected_video_id:
    st.session_state.messages = []
    # Clean up the vector store if RAG system exists
    if st.session_state.get("rag_system") is not None:
        st.session_state.rag_system.cleanup_vector_store(verbose=False)
    st.session_state.current_video_id = selected_video_id
    st.session_state.rag_system = None  # Reset RAG system

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("Ask me anything about this episode! 🤔"):
    if st.session_state.rag_system is None:
        st.warning("Please load the interview first by clicking the 'Load Interview' button in the sidebar.")
    else:

        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.spinner("Thinking... 🤔"):
            response = st.session_state.rag_system.conversational_rag_chain.invoke(
                {
                    "input": prompt,
                    "video_title": video_manager.video_title,
                    "guest_name": video_manager.guest_name,
                    "date": video_manager.date
                },
                config={"configurable": {"session_id": st.session_state.rag_system.session_id}}
            )
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response['answer'])
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})


# Call-to-action
if not st.session_state.messages:
    st.info("👋 Welcome! Select an episode and start chatting to explore its content.")