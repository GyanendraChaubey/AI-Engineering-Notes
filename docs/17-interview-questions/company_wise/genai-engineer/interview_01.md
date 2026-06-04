# GenAI Engineer — Interview 1

**1. User Interaction Layer:**

**Voice Input Processing:**
- User speaks into the microphone
- Google Web Speech API converts speech to text using SpeechRecognition library
- Supports multilingual input (English/Hindi/Hinglish)
- Audio preprocessing with pydub for noise handling

**Text Input Alternative:**
- Users can also type messages directly via the web interface

---

**2. Session Management:**

**Session Initialization:**
- When user starts, FastAPI creates a unique session_id
- Redis stores the session state for fast access
- SQLite/PostgreSQL stores long-term conversation history
- Session tracks: language preference, conversation history, sentiment trends, crisis indicators

**State Tracking:**
- Maintains last 10 messages for context
- Tracks sentiment history for trend analysis
- Stores user preferences and coping strategies
- Monitors session timeout (hard timeout check)

---

**3. Message Processing Pipeline (POST /chat/{session_id}):**

**Step 1: Timeout Check**
- Checks if session has expired
- Returns appropriate message if timeout occurred

**Step 2: Crisis Detection (Layer 1 - Fast Pattern Matching)**
- Immediately scans message for crisis keywords
- Pattern matching using intent_detector
- Categories: immediate_risk, suicidal_ideation, severe_distress
- If crisis detected → LLM verification step
- Immediate risk → blocks normal flow, returns emergency hotlines
- Severe distress → passes to agent with crisis_context flag

**Step 3: User Input Processing**
- Appends user message as HumanMessage to conversation state
- Updates message history in Redis

**Step 4: Omni-Analyzer (Single LLM Call)**
- Classifies multiple aspects in one call:
 - **primary_intent**: consent_yes/no, gratitude_end, new_problem, therapy_request
 - **sentiment_score**: -1.0 to +1.0 scale
 - **is_crisis**: boolean flag
 - **is_new_topic**: topic change detection
- Uses Llama 3.3-70B via Groq API

**Step 5: Sentiment Trend Analysis**
- Analyzes last 10 sentiment scores
- Detects patterns: declining mood, persistent negativity
- Triggers intervention warnings if needed
- Logs to json_logger for monitoring

**Step 6: Topic Change Reset**
- If new topic detected and phase allows reset
- Clears previous context
- Resets conversation flow for fresh start

---

**4. LLM Orchestration with LangGraph:**

**Agent Routing:**
The system uses LangGraph to orchestrate multiple specialized agents:

**A. Normal Therapy Agent:**
- Handles general emotional support
- Empathetic responses using Llama 3.3-70B
- Temperature: 0.7 for warm, human-like responses
- Maintains conversation context

**B. Gita RAG Agent (Spiritual Guidance):**
- When user asks spiritual questions
- Retrieves relevant verses from Bhagavad Gita
- Uses FAISS vector database with multilingual-e5-large embeddings
- Semantic search for contextually relevant wisdom
- Combines retrieved verses with LLM explanation

**C. Therapy Recommendation Agent:**
- Triggers when user needs therapy suggestions
- Calls the Therapy Recommendation System (explained earlier)
- Integrates with TherapyVectorStore (FAISS-based)
- Returns personalized therapy videos/exercises

**D. Crisis Intervention Agent:**
- Activated on crisis detection
- Provides immediate support resources
- Emergency hotline numbers
- Safety planning guidance

---

**5. RAG Systems (Two Separate Knowledge Bases):**

**Gita RAG:**
- FAISS vector database storing Bhagavad Gita verses
- Embeddings: intfloat/multilingual-e5-large
- Retrieves top 3-5 relevant verses
- Combines with LLM for contextual explanation

**Therapy RAG:**
- FAISS-based TherapyVectorStore
- 50+ therapy PDFs embedded
- BioBERT embeddings for medical accuracy
- Retrieves relevant therapy documents
- Feeds into recommendation generation

---

**6. Response Generation:**

**LLM Processing:**
- Groq API with Llama 3.3-70B (main model)
- Fast model for small classification calls
- Structured prompts with:
 - Conversation history
 - User sentiment trends
 - Crisis context (if applicable)
 - Retrieved RAG context
 - Cultural/language preferences

**Response Assembly:**
- Combines LLM output with retrieved knowledge
- Formats response based on intent
- Adds therapy recommendations if requested
- Includes follow-up questions for engagement

---

**7. Voice Output (TTS):**

**Text-to-Speech Conversion:**
- Pluggable TTS service
- Converts text response to WAV audio
- Returns audio/wav response to frontend
- User hears the response in natural voice

---

**8. Persistent Storage:**

**Database Design (SQLAlchemy 2):**

**Tables:**
- **users**: User profiles (email, name, external_user_id)
- **chat_sessions**: Session records (language, summary, primary_concern)
- **chat_messages**: Every message (role, content, metadata_json)
- **therapy_records**: Videos recommended + feedback
- **user_insights**: LLM-extracted patterns/preferences
- **user_memories**: Long-term memory (beliefs, events, coping strategies)
- **outbox_messages**: Transactional outbox for async DB writes

**Outbox Pattern:**
- Messages written to outbox_messages with status=pending
- Background worker processes and writes to chat_messages
- Decouples real-time response from DB latency

---

**9. Key Technical Components:**

**Backend:**
- FastAPI (async, Python 3.11+)
- Handles all API endpoints
- WebSocket support for real-time chat

**LLM Service:**
- Groq API for fast inference
- Llama 3.3-70B for main conversations
- Smaller models for quick classifications

**Vector Databases:**
- FAISS for both Gita and Therapy RAG
- Fast semantic search
- Multilingual embeddings

**Session Management:**
- Redis for fast state access
- Tracks conversation flow
- Manages user context

**Deployment:**
- Docker / Docker Compose
- Containerized services
- Easy scaling and deployment

---

**10. Conversation Flow Example:**

**User**: "I'm feeling very stressed and can't sleep"

1. **Speech → Text**: Google Speech API converts
2. **Crisis Check**: No immediate crisis detected
3. **Omni-Analyzer**: 
 - Intent: new_problem
 - Sentiment: -0.6 (negative)
 - Crisis: false
4. **Agent Selection**: Normal Therapy Agent
5. **Response**: Empathetic acknowledgment + follow-up questions
6. **User**: "Can you suggest something to help?"
7. **Intent**: therapy_request
8. **Agent Switch**: Therapy Recommendation Agent
9. **RAG Retrieval**: Finds stress + sleep therapies
10. **Recommendation**: Yoga Nidra + Pranayama with rationale
11. **TTS**: Converts to voice
12. **Storage**: Saves conversation + recommendation to DB

---

**Key Features:**

✅ **Multilingual Support**: English, Hindi, Hinglish
✅ **Crisis Detection**: Real-time safety monitoring
✅ **Multi-Agent System**: Specialized agents for different needs
✅ **RAG Integration**: Knowledge-grounded responses
✅ **Voice Enabled**: Full STT + TTS pipeline
✅ **Persistent Memory**: Long-term user understanding
✅ **Scalable Architecture**: Docker-based deployment

**Performance Metrics:**
- Sub-second response latency
- 95%+ intent detection accuracy
- Real-time crisis intervention
- Seamless voice interaction

This system combines conversational AI, RAG, agentic workflows, and voice technology to provide comprehensive mental wellness support.

---
