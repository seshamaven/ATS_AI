# How to Set Pinecone API Key

The error occurs because the `PINECONE_API_KEY` environment variable is not set. Here are several ways to fix it:

## Method 1: Set Environment Variable (Temporary - Current Session Only)

### Windows PowerShell:
```powershell
$env:PINECONE_API_KEY="your-api-key-here"
python create_java_index.py
```

### Windows CMD:
```cmd
set PINECONE_API_KEY=your-api-key-here
python create_java_index.py
```

### Linux/Mac:
```bash
export PINECONE_API_KEY=your-api-key-here
python create_java_index.py
```

## Method 2: Create .env File (Recommended)

1. Create a file named `.env` in the `backend` directory
2. Add this line to the file:
   ```
   PINECONE_API_KEY=your-api-key-here
   ```
3. Install python-dotenv (if not already installed):
   ```bash
   pip install python-dotenv
   ```
4. Run the script:
   ```bash
   python create_java_index.py
   ```

## Method 3: Set System Environment Variable (Permanent)

### Windows:
1. Open System Properties → Environment Variables
2. Add new User variable:
   - Variable name: `PINECONE_API_KEY`
   - Variable value: `your-api-key-here`
3. Restart your terminal/IDE

### Linux/Mac:
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export PINECONE_API_KEY="your-api-key-here"
```
Then run:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Method 4: Pass API Key Directly in Code

Modify `create_java_index.py` to pass the API key:

```python
from create_dynamic_pinecone_index import DynamicPineconeIndexCreator

creator = DynamicPineconeIndexCreator(api_key="your-api-key-here")
```

## Get Your Pinecone API Key

1. Go to https://app.pinecone.io/
2. Sign in or create an account
3. Navigate to API Keys section
4. Copy your API key

## Verify API Key is Set

### Windows PowerShell:
```powershell
echo $env:PINECONE_API_KEY
```

### Windows CMD:
```cmd
echo %PINECONE_API_KEY%
```

### Linux/Mac:
```bash
echo $PINECONE_API_KEY
```

## Alternative: Use ATS_PINECONE_API_KEY

The script also checks for `ATS_PINECONE_API_KEY` environment variable, so you can use that instead:

```powershell
$env:ATS_PINECONE_API_KEY="your-api-key-here"
python create_java_index.py
```

