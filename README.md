# PDF RAG System with Azure AI Integration

A comprehensive Retrieval-Augmented Generation (RAG) system for PDF documents using LangChain, Azure OpenAI, Azure Speech Services, and FAISS vector store.

## ✨ Features

### Core Functionality
- 📚 **Multi-PDF Processing**: Automatically process multiple PDF documents from a folder
- 🔍 **Semantic Search**: Advanced similarity search using Azure OpenAI embeddings
- 💬 **Conversational AI**: Interactive question-answering with conversation memory
- 🖥️ **Modern Web Interface**: Clean, responsive Streamlit UI with professional design
- � **Source References**: Detailed source document citations with page numbers

### Advanced Features
- 🎤 **Voice Chat**: Speech-to-text and text-to-speech with Azure Speech Services
- 🌍 **Multi-language Support**: Auto-detection for Ukrainian, English, and Russian
- 🤖 **Follow-up Questions**: AI-generated contextual follow-up suggestions
- � **Auto-processing**: Automatic document processing on startup
- 🗣️ **Voice Output**: Professional neural voices for answer playback
- 🎯 **Smart Error Handling**: Comprehensive error management and user feedback

## 📋 Requirements

- Python 3.8 or higher
- Azure OpenAI account with deployed models
- (Optional) Azure Speech Services for voice features

## 🚀 Installation

1. **Clone or download this project**

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Set up your Azure credentials:**
   - Copy `.env.example` to `.env`
   - Fill in your Azure credentials in the `.env` file:

```env
# Required - Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name

# Optional - Azure Speech Services (for voice features)
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_speech_region
```

4. **Prepare your documents:**
   - Create a `documents/` folder in the project directory
   - Add your PDF files to the `documents/` folder

## 🎯 Usage

### Running the Application
```bash
streamlit run rag_app.py
```
or
```bash
python -m streamlit run rag_app.py
```

The application will open in your browser at `http://localhost:8501`

### Key Features:

#### 📁 **Document Management**
- **Auto-discovery**: Automatically finds PDF files in the `documents/` folder
- **Auto-processing**: Documents are processed automatically on first run
- **Manual processing**: Use the "📤 Process Documents" button to reprocess files
- **Status indicators**: Clear feedback on document processing status

#### 💬 **Chat Interface**
- **Text Input**: Type questions directly or use example questions
- **Conversational Mode**: Toggle to enable/disable conversation memory
- **Source Citations**: View relevant document excerpts for each answer
- **Chat History**: Full conversation tracking with timestamps

#### 🎤 **Voice Features** (Optional)
- **Voice Input**: Record questions using the microphone
- **Multi-language**: Auto-detection for Ukrainian, English, and Russian
- **Voice Output**: Listen to answers with neural text-to-speech
- **Language Feedback**: Visual confirmation of detected speech language

#### 💡 **Smart Suggestions**
- **Follow-up Questions**: AI-generated contextual questions
- **Example Questions**: Pre-defined starter questions for new users
- **Quick Actions**: One-click buttons for common operations

## 🔧 How It Works

### Technical Architecture

1. **📄 Document Loading**: PDF documents are loaded using `PyPDFLoader` with error handling
2. **✂️ Text Splitting**: Documents are split into manageable chunks using `RecursiveCharacterTextSplitter`
3. **🧮 Embeddings**: Text chunks are converted to vector embeddings using Azure OpenAI's `text-embedding-ada-002` model
4. **🗃️ Vector Store**: Embeddings are stored in FAISS for efficient similarity search
5. **🔍 Retrieval**: When a question is asked, the system finds the most relevant chunks using semantic similarity
6. **🤖 Generation**: Azure OpenAI's GPT model generates contextual answers based on retrieved content
7. **🎤 Speech Processing**: Azure Speech Services handle voice input/output with multi-language support

### Code Architecture

- **`SpeechService`**: Dedicated class for Azure Speech Services integration
- **`PDFRAGSystem`**: Main RAG system with comprehensive error handling
- **Modular Functions**: Clean separation of concerns with utility functions
- **Type Hints**: Full type annotation for better code maintainability
- **Custom Exceptions**: Specific error classes for different failure modes

## 📁 File Structure

```
```
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── rag_app.py              # Streamlit web application
├── documents/              # PDF documents folder
├── vectorstore/            # Persistent vector store (created automatically)
├── .vscode/                # VS Code debugging configuration
├── AZURE_SETUP.md          # Azure OpenAI setup guide
└── README.md               # This file
```

## ⚙️ Configuration

### Environment Variables
All configuration is handled through environment variables in the `.env` file:

```python
# Core RAG Parameters (defined as constants)
DEFAULT_CHUNK_SIZE = 1000          # Size of each text chunk
DEFAULT_CHUNK_OVERLAP = 200        # Overlap between chunks
DEFAULT_RETRIEVAL_K = 4            # Number of chunks to retrieve
DEFAULT_MEMORY_K = 5               # Conversation memory length
MAX_TTS_TEXT_LENGTH = 1000         # Maximum text length for TTS

# Voice Configuration
UKRAINIAN_VOICES = ["uk-UA-PolinaNeural", "uk-UA-OstapNeural"]
SUPPORTED_LANGUAGES = ["uk-UA", "en-US"]
FALLBACK_VOICE = "en-US-AriaNeural"
```

### Advanced Configuration

#### Chunk Size and Overlap
Modify these parameters in the constants section:
```python
# In rag_app.py - Constants section
DEFAULT_CHUNK_SIZE = 1000        # Size of each text chunk
DEFAULT_CHUNK_OVERLAP = 200      # Overlap between chunks

# Used in RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    length_function=len
)
```

#### Retrieval Parameters
Adjust the number of documents retrieved:
```python
# In PDFRAGSystem.setup_qa_chain()
retriever = self.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": DEFAULT_RETRIEVAL_K}  # Number of chunks to retrieve
)
```

#### Model Configuration
The system uses these Azure OpenAI models:
```python
# Embeddings Model
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  # text-embedding-ada-002
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Chat Model
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),      # GPT-4 or GPT-3.5-turbo
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.7              # Creativity level (0-1)
)
```

## 💡 Example Questions

### Ukrainian (Medical Documents)
- "Який склад ципрофлоксацину?"
- "Які побічні ефекти ципрофлоксацину?"
- "Які протипоказання флуконазолу?"
- "Як правильно приймати ципрофлоксацин?"
- "Які можливі взаємодії з іншими препаратами?"

### English (General)
- "What are the main components of this medication?"
- "What are the side effects mentioned?"
- "How should this medication be administered?"
- "Are there any contraindications?"
- "What drug interactions should I be aware of?"

## 📦 Dependencies

### Core Dependencies
- **`langchain`** (^0.1.0): Main framework for building the RAG system
- **`langchain-openai`**: Azure OpenAI integration for LangChain
- **`langchain-community`**: Community extensions for document loaders and vector stores
- **`streamlit`** (^1.29.0): Modern web interface framework
- **`python-dotenv`**: Environment variable management

### AI & ML Libraries
- **`azure-cognitiveservices-speech`** (^1.45.0): Azure Speech Services for voice features
- **`faiss-cpu`**: Efficient vector database for similarity search
- **`pypdf`**: PDF document processing and text extraction

### Development & Utilities
- **`azure-identity`**: Azure authentication library
- **Optional**: `azure-storage-blob` for cloud document storage
