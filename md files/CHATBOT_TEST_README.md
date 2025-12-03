# Chatbot API Test Scripts

This directory contains several test scripts to interact with the chatbot API for the Regulatory RAG system. These scripts provide a command-line interface for testing the chatbot functionality.

## Available Test Scripts

### 1. `3testchatbot.bat` - Basic Batch Test
**Purpose**: Simple Windows batch file for basic chatbot testing
**Usage**: Double-click or run from command prompt
**Features**:
- Basic API call to chatbot endpoint
- Simple error handling
- Minimal output formatting

### 2. `3testchatbot_enhanced.bat` - Enhanced Batch Test
**Purpose**: Advanced Windows batch file with improved features
**Usage**: Double-click or run from command prompt
**Features**:
- Enhanced JSON parsing using PowerShell
- Detailed response formatting
- Session logging
- Interactive commands (help, clear, log, stats)
- Better error handling

### 3. `3testchatbot.py` - Python Test Script
**Purpose**: Cross-platform Python script with full functionality
**Usage**: `python 3testchatbot.py`
**Features**:
- Complete JSON parsing and formatting
- Session management and logging
- Interactive commands
- Colored output
- Comprehensive error handling
- Token usage analysis
- Performance metrics

### 4. `3testchatbot.ps1` - PowerShell Test Script
**Purpose**: Windows PowerShell script with colored output
**Usage**: `powershell -ExecutionPolicy Bypass -File 3testchatbot.ps1`
**Features**:
- Colored output for better readability
- Detailed response analysis
- Session logging
- Interactive commands
- Token usage summary
- Performance metrics

## Prerequisites

### For Batch Scripts:
- Windows Command Prompt
- `curl` command-line tool (usually pre-installed on Windows 10+)
- PowerShell (for enhanced batch script)

### For Python Script:
- Python 3.6+
- `requests` library: `pip install requests`

### For PowerShell Script:
- Windows PowerShell 5.0+
- No additional dependencies

## Usage Instructions

### 1. Start the Chatbot API
Before running any test script, ensure the chatbot API is running:

```bash
# Navigate to the regaiagent directory
cd c:\Complinova\reglibrarycode\reglibrary\regaiagent

# Start the chatbot API
python chatbot_api.py
```

The API will start on `http://localhost:5001`

### 2. Run the Test Scripts

#### Option A: Basic Batch Test
```cmd
3testchatbot.bat
```

#### Option B: Enhanced Batch Test
```cmd
3testchatbot_enhanced.bat
```

#### Option C: Python Test (Recommended)
```bash
python 3testchatbot.py
```

#### Option D: PowerShell Test
```powershell
powershell -ExecutionPolicy Bypass -File 3testchatbot.ps1
```

## Interactive Commands

All enhanced scripts support the following commands:

### Query Commands
- Type any regulatory question to get a response
- Examples:
  - "What are RBI guidelines for capital adequacy?"
  - "SEBI regulations for mutual funds"
  - "AML compliance requirements"
  - "Cybersecurity guidelines for banks"

### Special Commands
- `help` - Show help message
- `clear` - Clear the screen
- `log` - View session log
- `stats` - Show session statistics
- `quit`/`exit`/`q` - End the session

## Expected Output

### Successful Query Response
```
============================================================
🤖 Bot Response:
============================================================
Based on the RBI guidelines, capital adequacy requirements...

============================================================
📊 Query Analysis:
============================================================
Relevance: Highly Relevant
Regulatory Domains: Banking, Risk Management

============================================================
⚡ Performance Metrics:
============================================================
Processing Time: 1250 ms
Quality Score: 0.85
Safety Score: 0.92

============================================================
💰 Token Usage:
============================================================
Total Tokens: 1250
Models Used: text-embedding-ada-002, gpt-4

📚 Context Sources: 3 regulations used
```

### Error Scenarios

#### API Not Running
```
❌ Error: Chatbot API is not running or not accessible

To start the chatbot API:
1. Open a terminal/command prompt
2. Navigate to the regaiagent directory
3. Run: python chatbot_api.py
```

#### Query Classification
```
Query Relevance: Irrelevant
Response: I can only assist with regulatory compliance questions...
```

## API Endpoint Details

### Endpoint
```
POST http://localhost:5001/chat
```

### Request Body
```json
{
  "query": "What are RBI guidelines for capital adequacy?"
}
```

### Response Format
```json
{
  "llm_response": "Based on the RBI guidelines...",
  "query_classification": {
    "relevance": "Highly Relevant",
    "domains": ["Banking", "Risk Management"]
  },
  "response_validation": {
    "quality_score": 0.85,
    "safety_score": 0.92
  },
  "token_usage": {
    "total_tokens": 1250,
    "models_used": ["text-embedding-ada-002", "gpt-4"]
  },
  "processing_time_ms": 1250,
  "context_regulations": [...],
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

## Session Management

### Log Files
- Each session creates a unique log file
- Format: `chatbot_session_YYYYMMDD_HHMMSS.log`
- Contains all interactions, responses, and metadata

### Session Statistics
- Session ID tracking
- Conversation count
- Processing times
- Token usage summaries

## Troubleshooting

### Common Issues

1. **API Not Running**
   - Solution: Start the chatbot API with `python chatbot_api.py`
   - Check if port 5001 is available

2. **Database Connection Failed**
   - Solution: Check database configuration in `config.py`
   - Ensure database server is running
   - Verify connection credentials

3. **No Embeddings Found**
   - Solution: Run the embedding API first to create vector embeddings
   - Check Pinecone connection and index

4. **Query Classification Issues**
   - Solution: Ensure the production prompts system is working
   - Check query classification logic

5. **PowerShell Execution Policy**
   - Solution: Run with `-ExecutionPolicy Bypass` flag
   - Or change execution policy: `Set-ExecutionPolicy RemoteSigned`

### Performance Considerations

- **Response Time**: First query may take longer due to model loading
- **Token Costs**: Monitor token usage for cost management
- **Memory Usage**: Ensure sufficient RAM for processing
- **Network Timeout**: Increase timeout for slow connections

## Integration with CI/CD

These test scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Test Chatbot API
  run: |
    python chatbot_api.py &
    sleep 10
    python 3testchatbot.py
```

## Security Considerations

- **API Keys**: Ensure OpenAI API keys are properly configured
- **Database Access**: Verify database credentials are secure
- **Input Validation**: The API validates and classifies all queries
- **Response Filtering**: Harmful or irrelevant queries are filtered out

## Support

For issues or questions:
1. Check the log files for detailed error information
2. Verify all prerequisites are met
3. Ensure the chatbot API is running correctly
4. Check database connectivity and data availability
5. Verify Pinecone connection and embeddings

## Example Session

```
============================================================
🤖 Regulatory RAG - Interactive Chatbot
============================================================

Configuration:
- API Endpoint: http://localhost:5001/chat
- Timeout: 60 seconds
- Session ID: session_a1b2c3d4
- Log File: chatbot_session_20240115_103045.log

✅ Chatbot API is running

============================================================
You: What are RBI guidelines for capital adequacy?
🔄 Processing your query...

============================================================
🤖 Bot Response:
============================================================
Based on the RBI guidelines, capital adequacy requirements...

============================================================
📊 Query Analysis:
============================================================
Relevance: Highly Relevant
Regulatory Domains: Banking, Risk Management

============================================================
⚡ Performance Metrics:
============================================================
Processing Time: 1250 ms
Quality Score: 0.85
Safety Score: 0.92

============================================================
💰 Token Usage:
============================================================
Total Tokens: 1250
Models Used: text-embedding-ada-002, gpt-4

📚 Context Sources: 3 regulations used

============================================================
You: help
============================================================
📖 Help - Available Commands
============================================================
...
```
