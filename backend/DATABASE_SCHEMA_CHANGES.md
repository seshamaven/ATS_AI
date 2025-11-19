# Database Schema Changes Summary

## Overview
Updated the codebase to reflect the removal of the following columns from the `resume_metadata` table:
- `embedding`
- `embedding_model`
- `source`
- `resume_text`

## Changes Made

### 1. Schema File (`ats_schema.sql`)
- Removed columns `embedding`, `embedding_model`, `source`, and `resume_text` from `resume_metadata` table definition
- Removed FULLTEXT index on `resume_text` column

### 2. Database Module (`ats_database.py`)
- Updated `insert_resume()` method:
  - Removed `embedding` parameter
  - Removed references to `resume_text`, `source`, `embedding`, and `embedding_model` in INSERT query
  - Removed embedding JSON conversion logic
- Updated `get_all_resumes()` methods:
  - Removed `resume_text`, `embedding`, `embedding_model`, and `source` from SELECT queries
  - Removed embedding parsing logic
- Updated `get_resume_by_id()` method:
  - Removed embedding parsing logic
- Updated `update_resume()` method:
  - Removed special handling for embedding field

### 3. API Module (`ats_api.py`)
- Updated `process_resume()` endpoint:
  - Modified to not pass embedding to `insert_resume()` method
  - Embeddings are now only stored in Pinecone, not in MySQL
  - Updated response to indicate embeddings are stored in Pinecone only
- Updated `index_existing_resumes()` endpoint:
  - Modified to skip indexing existing resumes (since `resume_text` is no longer in database)
  - Added warning messages explaining the limitation
- Updated `get_candidate()` endpoint:
  - Removed references to embedding and resume_text fields
  - Added comment explaining these fields are handled in Pinecone only

## Impact on System Architecture

### Before Changes:
- Embeddings were stored in both MySQL (as JSON) and Pinecone
- Full resume text was stored in MySQL
- Source information was tracked per resume

### After Changes:
- Embeddings are stored ONLY in Pinecone
- Resume text is NOT stored in MySQL
- Source information is NOT tracked in MySQL
- Semantic search relies entirely on Pinecone

## Important Notes

1. **Existing Data**: Existing resumes in the database will not have embeddings or resume text. 
   - The `index_existing_resumes` endpoint will skip these resumes as they cannot be indexed without resume text.

2. **New Uploads**: Only newly uploaded resumes will be processed and indexed in Pinecone with their embeddings.

3. **Ranking**: The ranking engine will still work, but semantic similarity boosts may not be available for candidates retrieved from the database (embeddings are only in Pinecone).

4. **Search**: Resume search functionality will now rely entirely on Pinecone for vector similarity search.

## Migration Recommendations

1. If you need to retain existing resume data with embeddings, consider:
   - Exporting existing embeddings before dropping the columns
   - Re-indexing them in Pinecone before running the schema changes

2. The system is now optimized for:
   - Vector search via Pinecone
   - Metadata-only storage in MySQL
   - Faster inserts (no large embedding/text storage in MySQL)

## Testing Checklist

- [ ] Verify new resume uploads work correctly
- [ ] Verify Pinecone indexing works for new uploads
- [ ] Verify resume search via Pinecone works
- [ ] Verify ranking functionality still works (may have reduced semantic boost)
- [ ] Verify candidate retrieval endpoints work without embedding/resume_text fields

## 2025-11-19 Update: `profile_type` Column

- Added `profile_type` column to `resume_metadata` to persist a canonical role classification (Java, .Net, SAP, etc.).
- Updated `ats_database.py` insert/select logic to read/write the new column and introduced `filter_candidates()` for SQL-layer filtering.
- Resume parser now infers `profile_type` based on extracted skills so every new profile stores the value automatically.
- Pinecone metadata and search/ranking endpoints use `profile_type` to reduce the candidate pool before semantic/LLM layers.

