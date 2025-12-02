# Stateless Architecture Documentation

## Overview

This document describes how the ATS (Application Tracking System) ensures **complete request isolation** and **statelessness** - meaning that each user's request is processed independently with no data or state from one request influencing another.

## Core Principle

**Every request must be completely isolated** - no skill data, resume parsing results, or AI interpretations from one user should influence another user's query or results.

## Current Implementation

### 1. Database Operations - ✅ Stateless

- **Database connections** use Python context managers (`with create_ats_database()`)
- Each request gets a **fresh database connection**
- No shared database state between requests
- Each request reads/writes only its own data

```python
with create_ats_database() as db:
    candidates = db.get_all_resumes(status='active')
    # Database connection is isolated to this request
```

### 2. Ranking Engine - ✅ Stateless

- A **new instance** of the ranking engine is created for each ranking request
- No shared state between requests
- Each ranking operation uses only the input data provided

```python
# Each request creates fresh instance
ranking_engine = create_ranking_engine()

# Ranking uses only request-specific data
ranked_profiles = ranking_engine.rank_candidates(
    candidates=candidates,  # From database
    job_requirements=job_requirements,  # From current request
    jd_embedding=jd_embedding  # Generated for current request
)
```

### 3. Embedding Service - ✅ Stateless (with caveats)

- The `EmbeddingService` is a global singleton, but:
  - It only holds **immutable configuration** (API keys, model names)
  - The OpenAI/Azure client is **thread-safe** and stateless
  - Each `generate_embedding()` call is independent
  - No embeddings are cached or reused between requests

```python
# Global instance, but stateless
embedding_service = EmbeddingService()

# Each request generates fresh embeddings
resume_embedding = embedding_service.generate_embedding(parsed_data['resume_text'])
```

### 4. Resume Parser - ✅ Stateless (with caveats)

- The `ResumeParser` is a global singleton, but:
  - It only holds the **NLP model** (read-only, thread-safe)
  - Each `parse_resume()` call is independent
  - No parsing results are cached or stored
  - AI extraction is stateless (makes API calls per request)

```python
# Global instance, but stateless
resume_parser = ResumeParser()

# Each request parses independently
parsed_data = resume_parser.parse_resume(file_path, file_type)
```

### 5. AI/LLM Operations - ✅ Stateless

- Each AI call is **independent**
- No conversation history is maintained between requests
- No cached responses
- Each request to OpenAI/Azure is a fresh API call

```python
# AI extraction is stateless - fresh call per request
ai_data = self.extract_comprehensive_data_with_ai(resume_text)
```

## Key Design Decisions

### Why Global Instances are Safe

The global instances of `EmbeddingService` and `ResumeParser` are safe because:

1. **They only hold read-only configuration** - API keys, model names, etc.
2. **They don't maintain request data** - No caches or shared state
3. **Each method call is independent** - Processing happens per request
4. **Clients are thread-safe** - OpenAI/Azure clients handle concurrency

### Database Isolation

- Each API endpoint opens its own database connection
- Connection is closed when the context manager exits
- No connection pooling or shared state between requests
- Results from one query never influence another

### Request Processing Flow

```
Request 1 → Parse Resume → Generate Embedding → Store in DB → Return Response
  ↓
Request 2 → Parse Resume → Generate Embedding → Store in DB → Return Response
  ↓
  (No interaction between requests)
```

## Verification Checklist

✅ **Database operations use context managers**
✅ **Ranking engine creates fresh instances per request**
✅ **Embedding generation is independent per request**
✅ **Resume parsing is independent per request**
✅ **No global caches or shared data structures**
✅ **No request history maintained**
✅ **No cross-request data leakage**

## Multi-Threading Safety

The API is designed to handle multiple concurrent requests:

1. **Each thread gets isolated database connections**
2. **OpenAI/Azure clients are thread-safe**
3. **spaCy NLP models are thread-safe for read operations**
4. **No mutable global state is shared**

## Testing Request Isolation

To verify statelessness, you can:

1. **Send two concurrent requests** with different data
2. **Verify results are independent** - no cross-contamination
3. **Check logs** - each request should process its own data
4. **Monitor database** - each request writes only its own records

## Future Considerations

### If Adding Caching

If caching is added in the future (for performance), ensure:

1. **Cache keys include user/session context**
2. **Cache is cleared or updated appropriately**
3. **No request data leaks into cache for other users**
4. **Implement cache isolation**

### If Adding Session Management

If sessions are added:

1. **Each session is isolated**
2. **No shared session state**
3. **Session data is request-specific**
4. **No data leakage between sessions**

## Security Implications

Statelessness provides:

- ✅ **Data isolation** - Users can't see each other's data
- ✅ **Security** - No cross-user data access
- ✅ **Scalability** - Requests can be distributed across servers
- ✅ **Reliability** - One request failure doesn't affect others

## Conclusion

The ATS API is **designed to be stateless** with complete request isolation. Each user's request is processed independently with no data or state from one request influencing another. This is achieved through:

1. Fresh database connections per request
2. Independent AI/LLM calls per request
3. No caching or shared state
4. Thread-safe components
5. Isolated processing for each request

