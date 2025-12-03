# Regulation Library API

A production-ready Flask application for managing regulatory data with vector search capabilities using Pinecone and OpenAI embeddings.

## Features

- **Embedding API**: Fetches regulations from MySQL, creates embeddings, and stores them in Pinecone
- **Chatbot API**: Handles user queries with semantic similarity matching and database updates
- **Vector Search**: Uses Pinecone for efficient similarity search
- **Semantic Matching**: OpenAI LLM-powered similarity checking
- **Database Integration**: MySQL for regulation storage and status tracking

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file with the following variables (use `env_template.txt` as a reference):

```env
# Database Configuration
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=reglib
MYSQL_PORT=3306

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=reglibpinekey
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
EMBED_API_PORT=5000
CHAT_API_PORT=5001

# Application Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
EMBEDDING_DIMENSION=1536
```

**Important**: Replace the placeholder values with your actual API keys and database credentials.

### 3. Database Setup

Create the MySQL database and table:

```sql
CREATE DATABASE reglib;

USE reglib;

CREATE TABLE `regulations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `task_category` varchar(100),
  `task_subcategory` varchar(150),
  `regulator` varchar(150),
  `regulation` varchar(500),
  `reg_number` varchar(100),
  `reg_date` date,
  `reg_category` varchar(100),
  `reg_subject` varchar(500),
  `industry` varchar(150),
  `sub_industry` varchar(150),
  `activity_class` varchar(150),
  `sourced_from` text,
  `summary` text,
  `action_item` json,
  `prev_reg` text,
  `due_date` date,
  `frequency` varchar(50),
  `frequency_interval` varchar(50),
  `risk_category` varchar(150),
  `control_nature` varchar(150),
  `department` varchar(150),
  `notification_text` longtext,
  `status` varchar(150),
  PRIMARY KEY (`id`)
);
```

## Usage

### 1. Start the Embedding API

```bash
python embed_api.py
```

The embedding API runs on port 5000 and provides:
- `POST /embed`: Process regulations and create embeddings

### 2. Start the Chatbot API

```bash
python chatbot_api.py
```

The chatbot API runs on port 5001 and provides:
- `POST /chat`: Handle user queries

### 3. API Endpoints

#### Embedding API (`embed_api.py`)

**POST /embed**
- Triggers data fetch from MySQL
- Creates document chunks using LangChain
- Generates OpenAI embeddings
- Stores vectors in Pinecone

#### Chatbot API (`chatbot_api.py`)

**POST /chat**
- Accepts JSON: `{"query": "your question"}`
- Searches Pinecone for similar regulations
- Performs semantic similarity check
- Updates MySQL status or notification text
- Returns matched regulation details

## Example Usage

### Create Embeddings

```bash
curl -X POST http://localhost:5000/embed
```

### Query Regulations

```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for data privacy compliance?"}'
```

## Architecture

- **MySQL**: Stores regulation data and metadata
- **Pinecone**: Vector database for similarity search
- **OpenAI**: Embeddings and semantic similarity checking
- **LangChain**: Text processing and chunking
- **Flask**: REST API framework
- **Configuration Management**: Centralized config with environment variables

## Configuration Management

The application uses a centralized configuration system:

- **`config.py`**: Centralized configuration management
- **`.env`**: Environment variables for sensitive data
- **`env_template.txt`**: Template for environment variables
- **Configuration validation**: Ensures required variables are present
- **Multiple environments**: Development, production, and testing configs

## Error Handling

Both APIs include comprehensive error handling for:
- Database connection issues
- Pinecone API errors
- OpenAI API failures
- Invalid requests
- Missing data

## Logging

All operations are logged with appropriate levels:
- INFO: Successful operations
- ERROR: Failed operations with details
- WARNING: Non-critical issues
