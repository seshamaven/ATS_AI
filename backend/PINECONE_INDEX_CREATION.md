# Dynamic Pinecone Index Creation Guide

This guide explains how to create Pinecone indexes dynamically using Python.

## Files Created

1. **`create_dynamic_pinecone_index.py`** - Main utility class for dynamic index creation
2. **`create_java_index.py`** - Simple script to create a 'java' index
3. **`enhanced_pinecone_manager.py`** - Enhanced with static method for dynamic index creation

## Quick Start: Create a 'java' Index

### Method 1: Using the Simple Script

```bash
cd backend
python create_java_index.py
```

This will create a Pinecone index named `java` with:
- Dimension: 1536 (default for OpenAI embeddings)
- Metric: cosine
- Cloud: AWS
- Region: us-east-1

### Method 2: Using Python Code

```python
from create_dynamic_pinecone_index import DynamicPineconeIndexCreator

# Initialize the creator
creator = DynamicPineconeIndexCreator()

# Create a 'java' index
creator.create_index(
    index_name='java',
    dimension=1536,
    metric='cosine',
    cloud='aws',
    region='us-east-1'
)
```

### Method 3: Using EnhancedPineconeManager Static Method

```python
from enhanced_pinecone_manager import EnhancedPineconeManager

# Create index using static method
EnhancedPineconeManager.create_dynamic_index(
    index_name='java',
    dimension=1536
)
```

## Create Index for Profile Types

You can create indexes for different profile types:

```python
from create_dynamic_pinecone_index import DynamicPineconeIndexCreator

creator = DynamicPineconeIndexCreator()

# Create index for Java profiles
creator.create_profile_type_index('java')

# Create index for Python profiles
creator.create_profile_type_index('python')

# Create index for .NET profiles
creator.create_profile_type_index('.net')  # Will be normalized to 'ats-dotnet-resumes'
```

## Command Line Usage

### Create a custom index

```bash
python create_dynamic_pinecone_index.py <index_name> [dimension]
```

Examples:
```bash
# Create 'java' index with default dimension (1536)
python create_dynamic_pinecone_index.py java

# Create 'python' index with custom dimension
python create_dynamic_pinecone_index.py python 1536

# Create 'ats-java-resumes' index
python create_dynamic_pinecone_index.py ats-java-resumes
```

## Available Methods

### DynamicPineconeIndexCreator Class

#### `create_index(index_name, dimension=1536, metric='cosine', cloud='aws', region='us-east-1')`
Creates a new Pinecone index with specified parameters.

#### `create_profile_type_index(profile_type, dimension=1536)`
Creates an index for a specific profile type (normalizes the name automatically).

#### `list_all_indexes()`
Lists all existing Pinecone indexes.

#### `delete_index(index_name)`
Deletes a Pinecone index (use with caution!).

#### `get_index_info(index_name)`
Gets detailed information about a specific index.

## Example: Create Multiple Profile Type Indexes

```python
from create_dynamic_pinecone_index import DynamicPineconeIndexCreator

creator = DynamicPineconeIndexCreator()

# List of profile types
profile_types = ['java', 'python', '.net', 'javascript', 'devops', 'data-engineering']

# Create indexes for each profile type
for profile_type in profile_types:
    try:
        creator.create_profile_type_index(profile_type)
        print(f"✅ Created index for {profile_type}")
    except Exception as e:
        print(f"❌ Failed to create index for {profile_type}: {e}")

# List all created indexes
print("\nAll indexes:")
for idx in creator.list_all_indexes():
    print(f"  - {idx}")
```

## Index Naming Conventions

- **Simple index**: `java`, `python`, `dotnet`
- **Profile type index**: `ats-java-resumes`, `ats-python-resumes` (auto-generated)
- **Custom index**: Any valid Pinecone index name (lowercase, alphanumeric with hyphens/underscores)

## Requirements

- Pinecone API key must be set in environment variable `PINECONE_API_KEY`
- Pinecone Python SDK installed: `pip install pinecone-client`
- Valid Pinecone account with serverless plan

## Error Handling

The scripts handle common errors:
- Index already exists → Returns False, logs warning
- Invalid index name → Raises ValueError
- API key missing → Raises ValueError
- Network errors → Raises Exception with details

## Notes

- Index creation may take a few seconds
- Once created, indexes persist until deleted
- Each index has its own vector space
- Index names must be unique across your Pinecone account
- Serverless indexes are created in the specified cloud region

