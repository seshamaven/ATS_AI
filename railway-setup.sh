# Railway Environment Variables Setup
# Run these commands to set environment variables via Railway CLI

# Install Railway CLI (if not already installed)
# npm install -g @railway/cli

# Login to Railway
# railway login

# Set environment variables for your service
railway variables set OPENAI_API_KEY=your_openai_api_key_here
railway variables set OPENAI_MODEL=gpt-3.5-turbo
railway variables set OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
railway variables set MYSQL_HOST=your_mysql_host
railway variables set MYSQL_USER=your_mysql_user
railway variables set MYSQL_PASSWORD=your_mysql_password
railway variables set MYSQL_DATABASE=ats_db
railway variables set PINECONE_API_KEY=your_pinecone_api_key_here
railway variables set PINECONE_INDEX_NAME=atsscore
railway variables set PINECONE_CLOUD=aws
railway variables set PINECONE_REGION=us-east-1
railway variables set FLASK_ENV=production

# Deploy after setting variables
railway up
