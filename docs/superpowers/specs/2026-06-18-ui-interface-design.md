# UI Interface Design Specification
**Date**: 2026-06-18
**Component**: Gradio Interface for Project Aletheia

## Overview
This document specifies the design for the Gradio-based web interface (ui/app.py) for Project Aletheia, an autonomous AI tracking and reasoning system. The interface provides real-time visualization of the multi-agent LangGraph pipeline and enables user interaction with the system.

## Design Details

### Interface Layout
The interface uses a Gradio Blocks layout with five main sections:

1. **Header**: Project title and description
2. **Chat Interface**: Conversational interface for user queries and system responses
3. **Status Panel**: Real-time pipeline progress indicators
4. **Sources Panel**: Expandable sections showing retrieved context from each memory layer
5. **Metrics Panel**: Cost, token usage, and performance metrics
6. **Configuration Panel** (collapsible): System settings and preferences

### Component Specifications

#### Chatbot Display Area
- **Purpose**: Shows conversation history with user queries and system responses
- **Features**:
  - Displays user queries and system responses in chronological order
  - Each response includes inline citations matching source documents
  - Confidence scores displayed with each response
  - Supports markdown formatting for readable technical content
  - Auto-scrolls to latest message

#### Input Controls
- **Textbox**: For entering AI research questions
  - Placeholder: "Enter your AI research question..."
  - Supports multi-line input for complex queries
  - Enter key submission supported
- **Submit Button**: Triggers query processing
  - Label: "Ask Aletheia"
  - Disabled during processing to prevent duplicate submissions

#### Status Panel (Real-time)
- **Current Pipeline Step**: Shows active LangGraph node:
  * Routing → Retrieval → Evaluation → [Search if needed] → Compression → Generation
- **Step Indicator**: Visual progression through pipeline stages
- **Processing Status**: Idle/Processing/Complete/Error states with visual indicators

#### Sources Panel (Expandable Sections)
Each section shows retrieved content from a specific memory source:

- **📊 Neo4j Knowledge Graph**
  - Displays serialized graph triples when expanded
  - Shows ontology, facts, and passage chunks with provenance
  - Includes confidence scores for retrieved entities

- **🔍 FAISS Vector Store**
  - Shows dense vector retrieved chunks
  - Displays similarity scores for each chunk
  - Presents raw text with highlighted relevant sections

- **📚 BM25 Lexical Index**
  - Shows sparse lexical retrieved chunks
  - Displays BM25 relevance scores
  - Presents raw text with query term highlighting

- **🌐 External APIs** (Conditional, appears when external search is triggered)
  - Shows results from Crossref, OpenAlex, and Semantic Scholar
  - Displays paper titles, authors, abstracts, and publication years
  - Includes links to original sources when available

#### Metrics Panel
- **Total Cost**: Accumulated LLM API usage cost (USD)
- **Total Tokens**: Combined input/output token count
- **Evidence Score**: Current 100-point rubric score (0-100)
- **Avg Response Time**: Rolling average of query processing times
- **External API Calls**: Count of fallback searches performed

#### Configuration Panel (Collapsible by Default)
- **LLM Provider**: Radio buttons for provider selection
  * Local Ollama (default)
  * OpenAI API
  * Anthropic API
- **Local Model**: Dropdown for Ollama model selection
  * qwen2.5:7b (default)
  * llama3.1:8b
  * nemotron-3-super-120b-a12b
  * Custom model input
- **Evidence Threshold**: Slider (0-100) for triggering external search
  * Default: 80 (matches system specification)
  * Tooltips explain impact on performance vs. accuracy
- **Enable External Search**: Checkbox to toggle fallback search
  * Default: Enabled
- **Context Compression Ratio**: Display-only showing current AttentionRAG ratio
  * Typically 4-6.3x compression as per specification

### Technical Implementation

#### Backend Integration
- Imports LangGraph orchestrator from `graph.orchestrator`
- Creates async wrapper functions for LangGraph execution
- Uses Gradio's native async support for non-blocking UI updates
- Implements callback mechanism for real-time state updates

#### State Management
- Utilizes Gradio State component for conversation history
- Maps LangGraph state updates to specific UI component updates
- Implements error boundaries and recovery mechanisms
- Handles session state persistence for ongoing conversations

#### Real-time Updates
- Implements Gradio update mechanisms for panel refresh
- Uses periodic polling for long-running operations (>30s)
- Implements timeout handling (default 5-minute limit)
- Provides manual cancel operation for stuck processes

#### Error Handling
- Graceful degradation when individual components fail
- User-friendly error messages with retry options
- Detailed logging for debugging purposes
- Fallback to simplified interface when advanced features fail

### File Structure
```
ui/
├── __init__.py
└── app.py
```

### Dependencies
- Gradio (for interface)
- Existing Project Aletheia modules (core, graph, memory)
- Standard Python libraries (asyncio, json, etc.)

### Design Rationale
This design fulfills all requirements specified in SystemDesign.md:
- Provides Gradio web interface for interaction
- Features real-time pipeline status indicators
- Implements step-by-step cost tracking
- Offers expandable context sources for transparency
- Enables configuration of core system parameters
- Maintains consistency with existing codebase patterns

The interface balances usability with technical depth, allowing both casual users to get quick answers and researchers to inspect the system's internal workings for verification and debugging purposes.

### Open Issues & Future Enhancements
1. **Voice Input**: Potential future addition of speech-to-text for query input
2. **Response Audio**: Text-to-speech for system responses
3. **Conversation Export**: Ability to save/load conversation histories
4. **Advanced Filtering**: More granular control over source inclusion/exclusion
5. **Custom Pipelines**: Ability to enable/disable specific LangGraph nodes
6. **Visualization Graphs**: Neo4j graph visualization in the sources panel