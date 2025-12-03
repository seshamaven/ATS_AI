# Pinecone Index Limit - Solution Guide

## Problem

You've reached the **maximum limit of 5 serverless indexes** in your Pinecone free plan.

**Current indexes:**
1. ✅ ats-resumes
2. ✅ ats-resumes-1536
3. ✅ java
4. ✅ python
5. ✅ dotnet

**Missing indexes (can't create due to limit):**
- ❌ business-analyst
- ❌ project-manager
- ❌ sql
- ❌ others

## Solution Options

### Option 1: Use Namespaces (RECOMMENDED) ⭐

**Best approach** - Use namespaces within a single index instead of separate indexes.

**Benefits:**
- ✅ No limit on namespaces (unlimited!)
- ✅ Better organization
- ✅ Recommended by Pinecone
- ✅ Same performance as separate indexes
- ✅ No need to delete existing indexes

**How it works:**
- Use your existing `ats-resumes` index
- Store different profile types in different namespaces:
  - `java` namespace for Java profiles
  - `python` namespace for Python profiles
  - `business-analyst` namespace for BA profiles
  - etc.

**Setup:**
```bash
python setup_pinecone_namespaces.py --api-key YOUR_API_KEY
```

**Usage in code:**
```python
# Insert vector with namespace
index.upsert(
    vectors=[vector_data],
    namespace='java'  # Profile type as namespace
)

# Query specific namespace
results = index.query(
    vector=query_vector,
    namespace='java',
    top_k=10
)
```

### Option 2: Delete Unused Indexes

If you want separate indexes, delete unused ones first.

**Check what you have:**
```bash
python setup_pinecone_namespaces.py --list --api-key YOUR_API_KEY
```

**Delete duplicate/unused index:**
```bash
# Delete ats-resumes-1536 (might be duplicate of ats-resumes)
python setup_pinecone_namespaces.py --delete ats-resumes-1536 --api-key YOUR_API_KEY
```

**Then create missing indexes:**
```bash
python create_pinecone_indexes.py --api-key YOUR_API_KEY
```

### Option 3: Upgrade Pinecone Plan

Upgrade to a paid plan for more indexes:
- Visit: https://app.pinecone.io/
- Check pricing for higher limits

## Recommended Approach

**Use Option 1 (Namespaces)** because:
1. It's what Pinecone recommends
2. No limits - unlimited namespaces
3. Better organization
4. No need to manage multiple indexes
5. Same query performance

## Next Steps

1. **Check your current setup:**
   ```bash
   python setup_pinecone_namespaces.py --list --api-key YOUR_API_KEY
   ```

2. **Setup namespaces:**
   ```bash
   python setup_pinecone_namespaces.py --api-key YOUR_API_KEY
   ```

3. **Update your code** to use namespaces when inserting/querying vectors.

## Namespace Structure

Instead of:
- Index: `java` → Java profiles
- Index: `python` → Python profiles
- Index: `business-analyst` → BA profiles

Use:
- Index: `ats-resumes`
  - Namespace: `java` → Java profiles
  - Namespace: `python` → Python profiles
  - Namespace: `business-analyst` → BA profiles
  - Namespace: `project-manager` → PM profiles
  - Namespace: `sql` → SQL profiles
  - Namespace: `others` → Other profiles

This gives you unlimited profile types without hitting index limits!

