#!/usr/bin/env python3
"""
PDF RAG System with Azure AI Integration

A comprehensive Retrieval-Augmented Generation (RAG) system for PDF documents
using LangChain, Azure OpenAI, Azure Speech Services, and FAISS vector store.

Features:
- PDF document processing with chunking
- Conversational AI with memory
- Voice input/output with Azure Speech Services
- Multi-language support (Ukrainian, English, Russian)
- Source document references
- Follow-up question generation

Author: AI Assistant
Version: 2.1.0
"""

# Standard library imports
import os
import json
import tempfile
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
import base64

# Third-party imports
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Load environment variables
load_dotenv()

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 4
DEFAULT_MEMORY_K = 5
MAX_TTS_TEXT_LENGTH = 1000
UKRAINIAN_VOICES = ["uk-UA-PolinaNeural", "uk-UA-OstapNeural"]
SUPPORTED_LANGUAGES = ["uk-UA", "en-US"]
FALLBACK_VOICE = "en-US-AriaNeural"


class ConfigurationError(Exception):
    """Raised when there's a configuration issue."""
    pass


class AudioProcessingError(Exception):
    """Raised when there's an audio processing issue."""
    pass

class SpeechService:
    """Handles Azure Speech Services integration."""
    
    def __init__(self):
        """Initialize speech service with Azure credentials."""
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        
        if not self.speech_key or not self.speech_region:
            raise ConfigurationError(
                "Azure Speech credentials not found. Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION."
            )
    
    def speech_to_text(self, audio_data: Union[bytes, Any]) -> Optional[str]:
        """
        Convert speech to text using Azure Speech Services.
        
        Args:
            audio_data: Audio data as bytes or file-like object
            
        Returns:
            str: Transcribed text or None if failed
        """
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Handle different types of audio data
            audio_bytes = audio_data.read() if hasattr(audio_data, 'read') else audio_data
            
            # Create speech config with auto-detection
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            
            auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=SUPPORTED_LANGUAGES
            )
            
            # Use temporary file for better reliability
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            try:
                audio_config = speechsdk.audio.AudioConfig(filename=temp_audio_path)
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config,
                    auto_detect_source_language_config=auto_detect_config,
                    audio_config=audio_config
                )
                
                result = speech_recognizer.recognize_once()
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    # Display detected language if available
                    try:
                        auto_detect_result = speechsdk.AutoDetectSourceLanguageResult(result)
                        st.info(f"🗣️ Detected language: {auto_detect_result.language}")
                    except:
                        pass
                    return result.text
                    
                elif result.reason == speechsdk.ResultReason.NoMatch:
                    st.warning("⚠️ No speech could be recognized. Please speak more clearly.")
                    return None
                    
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = result.cancellation_details
                    error_msg = f"Speech Recognition canceled: {cancellation_details.reason}"
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        error_msg += f" - {cancellation_details.error_details}"
                    st.error(f"🚫 {error_msg}")
                    return None
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            st.error(f"🔧 Speech-to-text error: {str(e)}")
            return None
    
    def text_to_speech(self, text: str) -> Optional[bytes]:
        """
        Convert text to speech using Azure Speech Services.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bytes: Audio data or None if failed
        """
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Limit text length
            text_to_speak = text[:MAX_TTS_TEXT_LENGTH] if len(text) > MAX_TTS_TEXT_LENGTH else text
            
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            
            # Try Ukrainian voice first
            speech_config.speech_synthesis_voice_name = UKRAINIAN_VOICES[0]
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            
            result = synthesizer.speak_text_async(text_to_speak).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                st.success(f"🔊 Generated speech using voice: {UKRAINIAN_VOICES[0]}")
                return result.audio_data
                
            elif result.reason == speechsdk.ResultReason.Canceled:
                # Try fallback to English voice
                st.info("🔄 Trying English voice as fallback...")
                speech_config.speech_synthesis_voice_name = FALLBACK_VOICE
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
                result = synthesizer.speak_text_async(text_to_speak).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    st.info("🔊 Generated speech using English voice")
                    return result.audio_data
                else:
                    st.error("🚫 Speech synthesis failed with fallback voice")
                    return None
                    
        except Exception as e:
            st.error(f"🔧 Text-to-speech error: {str(e)}")
            return None


class PDFRAGSystem:
    """
    A comprehensive PDF RAG system with Azure AI integration.
    
    This class provides functionality for:
    - Loading and processing PDF documents
    - Creating vector embeddings and search
    - Conversational question-answering
    - Follow-up question generation
    """
    
    def __init__(self):
        """Initialize the RAG system with Azure AI services."""
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.conversational_chain: Optional[ConversationalRetrievalChain] = None
        
        # Initialize Azure services
        self._initialize_azure_services()
        
        # Initialize speech service
        try:
            self.speech_service = SpeechService()
        except ConfigurationError as e:
            st.warning(f"Speech service not available: {e}")
            self.speech_service = None
    
    def _initialize_azure_services(self) -> None:
        """Initialize Azure OpenAI services."""
        try:
            # Azure OpenAI Embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION")
            )
            
            # Azure OpenAI Chat Model
            self.llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0.7
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Azure services: {str(e)}")
    def load_documents_from_folder(self, folder_path: str = "documents") -> Tuple[List, List[str]]:
        """
        Load and process PDF files from a local folder.
        
        Args:
            folder_path: Path to the folder containing PDF files
            
        Returns:
            Tuple of (processed documents, list of PDF filenames)
            
        Raises:
            FileNotFoundError: If folder doesn't exist or no PDFs found
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Documents folder '{folder_path}' not found.")
        
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{folder_path}' folder.")
        
        documents = []
        successful_files = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                documents.extend(docs)
                successful_files.append(pdf_file)
            except Exception as e:
                st.error(f"❌ Error loading {pdf_file}: {str(e)}")
                continue
        
        if not documents:
            raise FileNotFoundError("No documents could be loaded successfully.")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len
        )
        
        split_docs = text_splitter.split_documents(documents)
        return split_docs, successful_files
    
    def create_vectorstore(self, documents: List) -> None:
        """
        Create vector store from processed documents.
        
        Args:
            documents: List of processed document chunks
        """
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
    
    def setup_qa_chain(self) -> None:
        """Setup both QA chain and conversational chain with custom prompts."""
        if not self.vectorstore:
            raise ValueError("Vector store must be initialized before setting up QA chains")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": DEFAULT_RETRIEVAL_K}
        )
        
        # Standard QA prompt
        qa_prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        qa_prompt = PromptTemplate(
            template=qa_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Setup standard QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        
        # Setup conversational chain with proper message history
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
    
    def query(self, question: str, use_conversation: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system with or without conversation context.
        
        Args:
            question: The question to ask
            use_conversation: Whether to use conversation context
            
        Returns:
            Dict containing result and source documents
        """
        if not self.qa_chain:
            return {"result": "Please upload and process PDF files first.", "source_documents": []}
        
        try:
            if use_conversation and self.conversational_chain and 'chat_history' in st.session_state:
                # Use invoke instead of __call__ to avoid deprecation warning
                result = self.conversational_chain.invoke({
                    "question": question,
                    "chat_history": st.session_state.get('chat_history', [])
                })
                return {
                    "result": result["answer"],
                    "source_documents": result.get("source_documents", [])
                }
            else:
                result = self.qa_chain.invoke({"query": question})
                return result
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            return {"result": "An error occurred while processing your question.", "source_documents": []}
    
    def generate_follow_up_questions(self, last_answer: str, context_docs: List) -> List[str]:
        """
        Generate follow-up questions based on the last answer and context.
        
        Args:
            last_answer: The AI's last response
            context_docs: Relevant context documents
            
        Returns:
            List of follow-up questions
        """
        if not last_answer or not context_docs:
            return []
        
        try:
            # Extract key topics from the context
            context_text = " ".join([doc.page_content[:200] for doc in context_docs[:2]])
            
            follow_up_prompt = f"""Based on this answer: "{last_answer[:300]}..." 
            and this context: "{context_text[:300]}..."
            
            Generate 3 short, relevant follow-up questions that a user might ask. 
            Make them specific and focused. Return only the questions, one per line."""
            
            response = self.llm.invoke(follow_up_prompt)
            questions = response.content.strip().split('\n')
            return [q.strip('- ').strip() for q in questions if q.strip() and len(q.strip()) > 10][:3]
        except Exception as e:
            st.error(f"❌ Error generating follow-up questions: {str(e)}")
            return []
    
    # Speech service wrapper methods
    def speech_to_text(self, audio_data: Union[bytes, Any]) -> Optional[str]:
        """Convert speech to text using Azure Speech Services."""
        if not self.speech_service:
            st.error("Speech service not available. Please check your Azure Speech configuration.")
            return None
        return self.speech_service.speech_to_text(audio_data)
    
    def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using Azure Speech Services."""
        if not self.speech_service:
            st.error("Speech service not available. Please check your Azure Speech configuration.")
            return None
        return self.speech_service.text_to_speech(text)


def create_audio_player(audio_data: bytes) -> str:
    """
    Create an audio player HTML element for the given audio data.
    
    Args:
        audio_data: Audio data as bytes
        
    Returns:
        str: HTML string for audio player
    """
    if not audio_data:
        return ""
    
    audio_base64 = base64.b64encode(audio_data).decode()
    audio_html = f"""
    <audio controls style="width: 100%;">
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html


def initialize_session_state() -> None:
    """Initialize all session state variables."""
    default_values = {
        'rag_system': None,
        'chat_history': [],
        'conversation_mode': True,
        'follow_up_questions': [],
        'current_question': "",
        'voice_enabled': False,
        'audio_data': None
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def validate_environment() -> None:
    """Validate that all required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            st.error(f"Please set {var} in the .env file")
            st.stop()
    
    # Optional voice-related environment variables
    if not os.getenv("AZURE_SPEECH_KEY"):
        st.warning("AZURE_SPEECH_KEY not set - voice features will not work")
    
    if not os.getenv("AZURE_SPEECH_REGION"):
        st.warning("AZURE_SPEECH_REGION not set - voice features will not work")


def render_sidebar() -> None:
    """Render the sidebar with document management and settings."""
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Check for documents in the folder
        documents_folder = "documents"
        if os.path.exists(documents_folder):
            pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]
            
            if pdf_files:
                st.success(f"Found {len(pdf_files)} PDF file(s):")
                for pdf_file in pdf_files:
                    st.write(f"• {pdf_file}")
                
                # Process documents button
                if st.button("📤 Process Documents", type="primary"):
                    with st.spinner("Processing PDF documents..."):
                        try:
                            documents, processed_files = st.session_state.rag_system.load_documents_from_folder(documents_folder)
                            st.session_state.rag_system.create_vectorstore(documents)
                            st.session_state.rag_system.setup_qa_chain()
                            st.success(f"✅ Successfully processed {len(processed_files)} PDF file(s)!")
                            st.info(f"📊 Total chunks created: {len(documents)}")
                        except Exception as e:
                            st.error(f"❌ Error processing documents: {str(e)}")
                
                # Auto-process on first load
                if st.session_state.rag_system.qa_chain is None and 'auto_processed' not in st.session_state:
                    with st.spinner("Auto-processing documents..."):
                        try:
                            documents, processed_files = st.session_state.rag_system.load_documents_from_folder(documents_folder)
                            st.session_state.rag_system.create_vectorstore(documents)
                            st.session_state.rag_system.setup_qa_chain()
                            st.session_state.auto_processed = True
                            st.success(f"✅ Auto-processed {len(processed_files)} PDF file(s)!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error auto-processing documents: {str(e)}")
            else:
                st.warning(f"⚠️ No PDF files found in '{documents_folder}' folder.")
                st.info("Please add PDF files to the documents folder and refresh.")
        else:
            st.error(f"❌ Documents folder '{documents_folder}' not found.")
            st.info("Please create a 'documents' folder and add PDF files to it.")
        
        st.markdown("---")
        
        # Conversation controls
        st.header("💬 Conversation Settings")
        
        # Conversation mode toggle
        conversation_mode = st.toggle(
            "Conversational Mode",
            value=st.session_state.conversation_mode,
            help="When enabled, the AI remembers previous questions and answers in the conversation."
        )
        st.session_state.conversation_mode = conversation_mode
        
        # Voice chat toggle
        voice_enabled = st.toggle(
            "🎤 Voice Chat",
            value=st.session_state.voice_enabled,
            help="Enable voice input and output for hands-free interaction."
        )
        st.session_state.voice_enabled = voice_enabled
        
        if voice_enabled:
            st.info("🎙️ Voice chat enabled! Use the microphone button to record your questions.")
            st.info("🗣️ **Language Support**: Ukrainian and English auto-detection")
        
        # Clear conversation button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.follow_up_questions = []
            st.success("✅ Chat history cleared!")
            st.rerun()
        
        # Show conversation history count
        if st.session_state.chat_history:
            st.info(f"💭 {len(st.session_state.chat_history)} messages in history")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PDF RAG System with Voice Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 PDF RAG System with Voice Chat")
    st.markdown("---")
    
    # Validate environment and initialize session state
    validate_environment()
    initialize_session_state()
    
    # Initialize RAG system
    if st.session_state.rag_system is None:
        try:
            st.session_state.rag_system = PDFRAGSystem()
        except ConfigurationError as e:
            st.error(f"❌ Configuration Error: {e}")
            st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Check if system is ready
    if st.session_state.rag_system.qa_chain is None:
        st.warning("⚠️ Please process PDF documents first using the sidebar.")
        st.stop()
    
    # Voice input section
    if st.session_state.voice_enabled:
        st.subheader("🎤 Voice Input")
        
        audio_bytes = st.audio_input("Record your question")
        
        if audio_bytes is not None:
            st.audio(audio_bytes)
            
            if st.button("🎙️ Process Voice Input", disabled=not audio_bytes):
                with st.spinner("🎧 Converting speech to text..."):
                    transcribed_text = st.session_state.rag_system.speech_to_text(audio_bytes)
                    if transcribed_text:
                        st.success(f"📝 Transcribed: {transcribed_text}")
                        st.session_state.current_question = transcribed_text
                        st.rerun()
        
        st.markdown("---")
    
    # Text input
    st.subheader("⌨️ Text Input")
    
    question = st.text_input(
        "Ask a question about your documents:",
        value=st.session_state.current_question,
        placeholder="Type your question here or use voice input above..."
    )
    
    # Clear current question after using it
    if st.session_state.current_question:
        st.session_state.current_question = ""
    
    # Query processing buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🔍 Ask Question", type="primary", disabled=not question):
            if question:
                with st.spinner("🤔 Thinking..."):
                    result = st.session_state.rag_system.query(question, use_conversation=True)
                    
                    # Store in chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["result"],
                        "sources": result.get("source_documents", []),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Generate follow-up questions
                    if result.get("source_documents"):
                        follow_ups = st.session_state.rag_system.generate_follow_up_questions(
                            result["result"], 
                            result["source_documents"]
                        )
                        st.session_state.follow_up_questions = follow_ups
                    
                    st.rerun()
    
    with col2:
        # Generate audio for last answer
        if st.button("🔊 Read Answer Aloud", disabled=not st.session_state.chat_history):
            if st.session_state.chat_history:
                latest_answer = st.session_state.chat_history[-1]["answer"]
                with st.spinner("🎵 Generating speech..."):
                    audio_data = st.session_state.rag_system.text_to_speech(latest_answer)
                    if audio_data:
                        audio_html = create_audio_player(audio_data)
                        st.markdown(audio_html, unsafe_allow_html=True)
    
    with col3:
        # Generate follow-up questions
        if st.button("❓ Suggest Follow-ups", disabled=not st.session_state.chat_history):
            if st.session_state.chat_history:
                latest = st.session_state.chat_history[-1]
                follow_ups = st.session_state.rag_system.generate_follow_up_questions(
                    latest["answer"], 
                    latest.get("sources", [])
                )
                st.session_state.follow_up_questions = follow_ups
                if follow_ups:
                    st.rerun()
    
    # Display follow-up suggestions
    if st.session_state.follow_up_questions:
        st.subheader("💡 Follow-up Questions")
        for i, follow_up in enumerate(st.session_state.follow_up_questions):
            if st.button(f"{i+1}. {follow_up}", key=f"followup_{i}"):
                st.session_state.current_question = follow_up
                st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(message["question"])
            
            with st.chat_message("assistant"):
                st.write(message["answer"])
                
                # Show source documents
                if message.get("sources"):
                    with st.expander("📄 Source Documents"):
                        for j, doc in enumerate(message["sources"][:3]):
                            st.write(f"**Source {j+1}:** Page {doc.metadata.get('page', 'N/A')}")
                            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    
    # Example questions for new users
    if not st.session_state.chat_history:
        st.subheader("💡 Example Questions")
        example_questions = [
            "Який склад ципрофлоксацину?",
            "Які побічні ефекти ципрофлоксацину?", 
            "Які протипоказання флуконазолу?",
            "Як правильно приймати ципрофлоксацин?",
            "Які можливі взаємодії з іншими препаратами?"
        ]
        
        cols = st.columns(2)
        for i, question_example in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question_example, key=f"example_{i}"):
                    st.session_state.current_question = question_example
                    st.rerun()


if __name__ == "__main__":
    main()
