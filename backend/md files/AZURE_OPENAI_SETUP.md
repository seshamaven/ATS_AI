# Azure OpenAI Configuration Guide

## Required Environment Variables for Azure OpenAI

To replace `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_EMBEDDING_MODEL` with Azure OpenAI, you need to set these environment variables in your `.env` file:

### 1. Azure OpenAI Service Configuration

```bash
# Azure OpenAI Configuration (REQUIRED)
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_MODEL=text-embedding-ada-002
```

### 2. How to Get These Values

1. **Go to Azure Portal** → Create Azure OpenAI resource
2. **AZURE_OPENAI_API_KEY**: Found in "Keys and Endpoint" section
3. **AZURE_OPENAI_ENDPOINT**: Found in "Keys and Endpoint" section (e.g., `https://your-resource.openai.azure.com/`)
4. **AZURE_OPENAI_DEPLOYMENT_NAME**: Name of your deployed model (e.g., `text-embedding-ada-002`)
5. **AZURE_OPENAI_MODEL**: Model name (usually same as deployment name)

### 3. Complete .env File Template

```bash
# MySQL Database Configuration
ATS_MYSQL_HOST=localhost
ATS_MYSQL_USER=root
ATS_MYSQL_PASSWORD=your_password_here
ATS_MYSQL_DATABASE=ats_db
ATS_MYSQL_PORT=3306

# Azure OpenAI Configuration (REQUIRED)
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_MODEL=text-embedding-ada-002

# OpenAI Configuration (LEAVE EMPTY - Use Azure instead)
# OPENAI_API_KEY=
# OPENAI_EMBEDDING_MODEL=

# Pinecone Configuration (Optional)
USE_PINECONE=False
ATS_PINECONE_API_KEY=your_pinecone_key_here
ATS_PINECONE_INDEX_NAME=ats-resumes
ATS_PINECONE_CLOUD=aws
ATS_PINECONE_REGION=us-east-1

# Flask API Configuration
FLASK_ENV=development
FLASK_DEBUG=True
ATS_API_PORT=5002

# File Upload Configuration
MAX_FILE_SIZE_MB=10
UPLOAD_FOLDER=./uploads

# Ranking Algorithm Weights
RANKING_WEIGHT_SKILLS=0.4
RANKING_WEIGHT_EXPERIENCE=0.3
RANKING_WEIGHT_DOMAIN=0.2
RANKING_WEIGHT_EDUCATION=0.1

# Score Matching Thresholds
EXP_MATCH_HIGH=0.9
EXP_MATCH_MEDIUM=0.7
DOMAIN_MATCH_HIGH=0.85
DOMAIN_MATCH_MEDIUM=0.65

# NLP Configuration
NLP_MODEL=en_core_web_sm

# Embedding Configuration
EMBEDDING_DIMENSION=1536
BATCH_EMBEDDING_SIZE=100
```

## How It Works

The code automatically detects if Azure OpenAI is configured and uses it instead of regular OpenAI:

1. **EmbeddingService** in `ats_api.py` checks for `AZURE_OPENAI_ENDPOINT`
2. If found, it uses `AzureOpenAI` client
3. If not found, it falls back to regular `OpenAI` client

## Key Differences

| Regular OpenAI | Azure OpenAI |
|----------------|--------------|
| `OPENAI_API_KEY` | `AZURE_OPENAI_API_KEY` |
| `OPENAI_MODEL` | `AZURE_OPENAI_MODEL` |
| `OPENAI_EMBEDDING_MODEL` | `AZURE_OPENAI_DEPLOYMENT_NAME` |
| N/A | `AZURE_OPENAI_ENDPOINT` |
| N/A | `AZURE_OPENAI_API_VERSION` |

## Testing

After setting up your `.env` file, test the configuration:

```bash
cd backend
python -c "from ats_config import ATSConfig; ATSConfig.print_config(); print('Azure OpenAI configured:', bool(ATSConfig.AZURE_OPENAI_ENDPOINT))"
```

This will show you if Azure OpenAI is properly configured.
