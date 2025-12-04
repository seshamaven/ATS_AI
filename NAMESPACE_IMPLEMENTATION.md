# Pinecone Namespace Implementation

## Overview
The codebase has been updated to use Pinecone namespaces instead of separate indexes. This solves the 5-index limit on Pinecone's free tier by using a single index with multiple namespaces.

## Changes Made

### 1. Enhanced Pinecone Manager (`enhanced_pinecone_manager.py`)
- **`upsert_vectors()`**: Now accepts an optional `namespace` parameter
  - If provided, vectors are inserted into that specific namespace
  - If `None`, vectors are inserted into the default namespace
- **`query_vectors()`**: Now accepts an optional `namespace` parameter
  - If provided, queries only that namespace
  - If `None`, queries the default namespace

### 2. Namespace Mapping (`ats_api.py`)
- **`get_namespace_from_profile_type()`**: New helper function that maps profile types to namespace names
  - Maps common profile types: Java → 'java', Python → 'python', .NET → 'dotnet', etc.
  - Handles multi-profile types (uses first one)
  - Normalizes names (lowercase, spaces to hyphens)
  - Defaults to 'others' for unknown profile types

### 3. Resume Processing (`ats_api.py`)
All resume processing endpoints now use namespaces:

- **`process_resume()`**: Determines namespace from `profile_type` and passes it to `upsert_vectors()`
- **`process_resume_base64()`**: Same namespace logic as above
- **`index_existing_resumes()`**: Groups vectors by namespace and upserts them in batches per namespace

### 4. Search/Query Operations (`ats_api.py`)
- **Ranking/Search queries**: Updated to query across multiple namespaces
  - Queries common namespaces: 'java', 'python', 'dotnet', 'business-analyst', 'sql', 'project-manager', 'others'
  - Also queries the default namespace
  - Deduplicates results by candidate_id (keeps highest score)
  - Merges results from all namespaces

## Namespace Mapping

| Profile Type | Namespace |
|-------------|-----------|
| Java | `java` |
| Python | `python` |
| .NET / C# / DotNet | `dotnet` |
| Business Analyst / BA | `business-analyst` |
| Project Manager / PM | `project-manager` |
| SQL / Database / DBA | `sql` |
| Others / Generalist | `others` |

## Benefits

1. **No Index Limit**: Can support unlimited profile types without hitting Pinecone's 5-index limit
2. **Better Organization**: Data is logically separated by profile type within a single index
3. **Efficient Queries**: Can query specific namespaces or all namespaces as needed
4. **Backward Compatible**: Still works with default namespace if namespace is not specified

## Usage Examples

### Inserting a Resume
```python
# Automatically determines namespace from profile_type
profile_type = "Java Developer"
namespace = get_namespace_from_profile_type(profile_type)  # Returns 'java'
pinecone_manager.upsert_vectors([vector_data], namespace=namespace)
```

### Querying Resumes
```python
# Query specific namespace
results = pinecone_manager.query_vectors(
    query_vector=embedding,
    top_k=10,
    namespace='java'
)

# Query all namespaces (done automatically in search endpoints)
# The code queries multiple namespaces and merges results
```

## Migration Notes

- Existing resumes in the default namespace will continue to work
- New resumes will be automatically placed in the appropriate namespace based on their profile_type
- Search operations query across all namespaces to ensure comprehensive results
- No changes needed to existing API endpoints - namespace logic is handled internally

## Testing

To verify namespace implementation:
1. Process a resume with profile_type="Java" - should be in 'java' namespace
2. Process a resume with profile_type="Python" - should be in 'python' namespace
3. Search for resumes - should return results from all namespaces
4. Check Pinecone dashboard - should see vectors in different namespaces within the same index

