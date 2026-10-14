# User Testing Guide for Project Aletheia

This guide provides step-by-step instructions for testing the Project Aletheia system, covering installation, configuration, and various test scenarios.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Basic System Tests](#basic-system-tests)
5. [Functional Tests](#functional-tests)
6. [Performance Tests](#performance-tests)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

Before testing Project Aletheia, ensure you have the following installed:

- **Python 3.8+**
- **Git** (for cloning the repository)
- **Neo4j Desktop** or Neo4j Server (version 4.0+)
- **Ollama** (for local LLM inference) - Optional but recommended
- **Git LFS** (for handling large files if applicable)

### Verifying Prerequisites

```bash
# Check Python version
python --version

# Check Git
git --version

# Check if Neo4j is running (should be accessible at bolt://localhost:7687)
# Neo4j Desktop typically runs on this port by default

# Check Ollama (optional)
ollama --version
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MahoRAGa
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Optional Dependencies for Full Functionality

```bash
# For PDF processing
pip install pymupdf

# For Neo4j
pip install neo4j

# For FAISS
pip install faiss-cpu

# For embeddings and cross-encoder
pip install sentence-transformers

# For LangGraph
pip install langgraph

# For Gradio UI
pip install gradio

# For BM25
pip install rank-bm25

# For Pydantic settings
pip install pydantic pydantic-settings
```

## Configuration

### 1. Set Up Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Optional: OpenAI API Key (if using cloud LLMs)
OPENAI_API_KEY=sk-your-openai-key-here

# Optional: Semantic Scholar API Key
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-key-here
```

### 2. Verify Neo4j is Running

Before running the system, ensure Neo4j is running and accessible:

1. Start Neo4j Desktop and ensure your DBMS is running
2. Or start Neo4j Server:
   ```bash
   neo4j start
   ```

### 3. Pull Required Ollama Model (if using local LLM)

```bash
ollama pull qwen2.5:7b
```

## Basic System Tests

### Test 1: System Startup and Health Check

```bash
# Start the Gradio UI
python ui/app.py
```

Expected outcome:
- The Gradio interface should start and be accessible at http://localhost:7860
- You should see the Project Aletheia interface with input box and status indicators
- No error messages in the console

### Test 2: Simple Query Test

1. In the Gradio interface, enter a simple AI-related question:
   ```
   What is the transformer architecture?
   ```
2. Click submit or press Enter
3. Wait for the system to process (this may take 30-60 seconds on first run)

Expected outcome:
- The system should progress through the pipeline steps (visible in status indicators)
- A response should be generated with citations
- The response should contain verifiable information about transformer architecture

### Test 3: Bibliometric Query Test

1. Enter a bibliometric question:
   ```
   Who is Yann LeCun and what are his contributions to AI?
   ```
2. Submit the query

Expected outcome:
- System should route to bibliometric path
- Response should contain information about Yann LeCun with proper citations
- Should include publication years, key contributions, etc.

### Test 4: Trend Analysis Test

1. Enter a trend question:
   ```
   How has the attention mechanism evolved in neural networks from 2017 to 2024?
   ```
2. Submit the query

Expected outcome:
- System should route to trend analysis path
- Response should show evolution of attention mechanisms over time
- Should mention key papers/models (Transformer, BERT, GPT, etc.) with timeline

## Functional Tests

### Test 5: Evidence Evaluation Test

To test the evidence evaluation system:

1. Enter a query that should trigger weak evidence score:
   ```
   What is the exact accuracy percentage of Model-X on Dataset-Y from the paper published on March 15, 2023?
   ```
   (Using a very specific, likely non-existent query)

Expected outcome:
- Evidence evaluator should score below 80
- System should trigger external search via Crossref/OpenAlex/Semantic Scholar
- Response should indicate external sources were consulted
- Final answer should acknowledge limitations if information isn't found

### Test 6: Context Compression Test

To test the attention-guided context compression:

1. Enter a query that would normally retrieve large amounts of context:
   ```
   Explain the complete training process of GPT-4 including all architectural details, hyperparameters, and training data specifications.
   ```
2. Submit the query

Expected outcome:
- System should retrieve substantial context
- AttentionRAG should compress the context by up to 6.3x
- Final response should be concise yet comprehensive
- System should show compression ratio in status indicators (if implemented)

### Test 7: Self-Correction Test

To test the self-correction loop:

1. Enter a query that might initially generate a response with citation issues:
   ```
   What are the exact three novel contributions of the ViT (Vision Transformer) paper?
   ```
2. Submit the query

Expected outcome:
- If the initial draft has citation or hallucination issues
- The Critic node should detect them
- System should trigger one automatic regeneration
- Final response should have proper citations and be factually accurate

## Performance Tests

### Test 8: Response Time Test

1. Time how long it takes to process a medium-complexity query:
   ```
   Compare and contrast the architectural differences between CNN and RNN for image processing tasks.
   ```
2. Use a stopwatch to measure from submission to complete response

Expected outcome:
- Response time should be reasonable (< 2 minutes for local LLM)
- System should show progressive status updates during processing
- Subsequent similar queries may be faster due to caching

### Test 9: Resource Usage Test

Monitor system resources during operation:

```bash
# In a separate terminal while running a query
# Check memory usage
# Check CPU usage
```

Expected outcome:
- Memory usage should be reasonable (< 4GB for typical operation)
- CPU usage should spike during processing but return to baseline
- No memory leaks should be observed during extended testing

### Test 10: Concurrent Query Test

1. Open multiple browser tabs to the Gradio interface
2. Submit different queries in each tab simultaneously
3. Observe system behavior

Expected outcome:
- System should handle concurrent requests (though may queue them)
- No crashes or severe degradation
- Each query should complete successfully

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Neo4j Connection Failure
**Symptoms**: Errors related to Neo4j connection in console logs
**Solutions**:
- Verify Neo4j is running and accessible at bolt://localhost:7687
- Check username/password in .env file
- Ensure Neo4j service is started (not just Desktop app)

#### Issue 2: Ollama Connection Failure
**Symptoms**: Errors connecting to local LLM
**Solutions**:
- Verify Ollama is running: `ollama list`
- Check OLLAMA_BASE_URL in .env (default: http://localhost:11434)
- Ensure the specified model is pulled: `ollama pull qwen2.5:7b`
- Try restarting Ollama service

#### Issue 3: Missing Dependencies
**Symptoms**: Import errors when starting the application
**Solutions**:
- Run `pip install -r requirements.txt` again
- Check for specific missing packages and install them
- Ensure virtual environment is activated

#### Issue 4: Slow Performance
**Symptoms**: Queries taking excessively long (> 5 minutes)
**Solutions**:
- Check if using CPU-only FAISS (consider GPU version for large datasets)
- Verify Ollama is using GPU if available
- Check Neo4j query performance
- Consider reducing context size in configuration

#### Issue 5: Empty or Incomplete Responses
**Symptoms**: System returns minimal or no information
**Solutions**:
- Try simpler, more common AI queries first
- Check if external APIs are accessible (network/firewall issues)
- Verify evidence evaluator threshold is appropriate
- Check if ingestion pipeline has processed any documents

## Testing Best Practices

1. **Start Simple**: Begin with basic, well-known AI concepts to verify core functionality
2. **Isolate Components**: Test individual aspects (routing, retrieval, generation) separately when possible
3. **Document Results**: Keep notes on what works and what doesn't for future reference
4. **Test Edge Cases**: Try unusual queries, misspellings, and ambiguous questions
5. **Verify Citations**: Always check that technical claims are properly cited in responses
6. **Monitor Resources**: Watch CPU/memory usage during testing to identify bottlenecks
7. **Test Both Success and Failure Paths**: Ensure the system handles both strong and weak evidence scenarios correctly

## Advanced Testing

For developers or advanced testers:

### Unit Testing Specific Components

```bash
# Run existing tests
pytest

# Test memory layer specifically
pytest tests/ -k "test_memory"

# Test individual LangGraph nodes
# (Would require setting up mock environments)
```

### API Testing

The system exposes internal APIs that can be tested directly:

```bash
# Example: Test the query router directly
# Would require importing and testing the route_query function
```

### Stress Testing

For performance validation:

1. Run multiple complex queries in sequence
2. Monitor for degradation over time
3. Test with large PDF corpora in the ingestion pipeline
4. Test concurrent user simulation

## Conclusion

Project Aletheia is a complex system with multiple interconnected components. Effective testing requires:

1. Proper setup of all dependencies (Neo4j, Ollama, Python packages)
2. Systematic testing of each major component
3. Validation of both functional correctness and performance characteristics
4. Attention to the system's unique features: evidence evaluation, context compression, and self-correction

By following this guide, you should be able to thoroughly test the system and verify that it meets the requirements outlined in the PRD and design documents.