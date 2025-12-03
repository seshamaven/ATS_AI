# Create Multiple Pinecone Indexes Guide

This guide shows you how to create all the required indexes: **python**, **.net**, **business-analyst**, **sql**, **project-manager**, and **others**.

## Quick Method: Create All at Once

### Option 1: Using Python Script (Recommended)

```bash
# With API key from environment variable
python create_multiple_indexes.py

# Or pass API key directly
python create_multiple_indexes.py --api-key YOUR_API_KEY
```

### Option 2: Using Batch Script (Windows)

```cmd
# With API key from environment variable
create_all_indexes.bat

# Or pass API key directly
create_all_indexes.bat YOUR_API_KEY
```

### Option 3: Set API Key First (Windows PowerShell)

```powershell
$env:PINECONE_API_KEY="your-api-key"
python create_multiple_indexes.py
```

## Indexes That Will Be Created

The script will create these indexes:

1. ✅ **python** - For Python developers
2. ✅ **dotnet** - For .NET developers (normalized from ".net")
3. ✅ **business-analyst** - For Business Analyst profiles
4. ✅ **sql** - For SQL/Database profiles
5. ✅ **project-manager** - For Project Manager profiles
6. ✅ **others** - For other profile types

## Create Individual Indexes

If you want to create indexes one by one:

### Python Index
```bash
python create_python_index.py --api-key YOUR_API_KEY
```

### .NET Index
```bash
python create_dynamic_pinecone_index.py dotnet 1536
```

### Business Analyst Index
```bash
python create_dynamic_pinecone_index.py business-analyst 1536
```

### SQL Index
```bash
python create_dynamic_pinecone_index.py sql 1536
```

### Project Manager Index
```bash
python create_dynamic_pinecone_index.py project-manager 1536
```

### Others Index
```bash
python create_dynamic_pinecone_index.py others 1536
```

## Verify All Indexes Were Created

Run the check script to see all your indexes:

```bash
python check_java_index.py YOUR_API_KEY
```

This will list all indexes in your Pinecone account.

## Index Specifications

All indexes are created with:
- **Dimension**: 1536 (for OpenAI embeddings)
- **Metric**: cosine (for similarity search)
- **Cloud**: AWS
- **Region**: us-east-1

## Expected Output

When you run `create_multiple_indexes.py`, you'll see:

```
============================================================
Creating Multiple Pinecone Indexes
============================================================
Indexes to create: python, dotnet, business-analyst, sql, project-manager, others

------------------------------------------------------------
Creating index: 'python'
------------------------------------------------------------
✅ Successfully created 'python' index!
   Dimension: 1536
   Metric: cosine
   Status: Ready

... (similar for each index)

============================================================
Summary
============================================================

✅ Created (6): python, dotnet, business-analyst, sql, project-manager, others
```

## Troubleshooting

### If an index already exists:
- The script will skip it and show "Already exists"
- This is normal - the index won't be recreated

### If creation fails:
- Check your API key is correct
- Verify you have internet connection
- Make sure you have available quota in Pinecone
- Try creating indexes one by one to identify the problematic one

## Notes

- Index names are normalized (lowercase, hyphens instead of spaces/special chars)
- ".net" becomes "dotnet"
- "business analyst" becomes "business-analyst"
- "project manager" becomes "project-manager"
- All indexes use the same dimension (1536) for consistency

