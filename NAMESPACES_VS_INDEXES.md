# Namespaces vs Indexes in Pinecone

## Key Differences

### Indexes
- **Separate vector databases** - Each index is completely independent
- **Limited** - Free plan allows only 5 indexes
- **Separate resources** - Each index uses its own quota
- **Separate connections** - Need to connect to each index separately

### Namespaces
- **Logical partitions within ONE index** - Like folders in a file system
- **Unlimited** - No limit on number of namespaces
- **Shared resources** - All namespaces share the same index quota
- **Single connection** - Connect once, query any namespace

## Visual Comparison

### Using Separate Indexes (Current - Limited to 5)
```
Index: java
  └─ Vector 1, Vector 2, Vector 3...

Index: python
  └─ Vector 1, Vector 2, Vector 3...

Index: business-analyst
  └─ Vector 1, Vector 2, Vector 3...
```

### Using Namespaces (Recommended - Unlimited)
```
Index: ats-resumes
  ├─ Namespace: java
  │   └─ Vector 1, Vector 2, Vector 3...
  ├─ Namespace: python
  │   └─ Vector 1, Vector 2, Vector 3...
  ├─ Namespace: business-analyst
  │   └─ Vector 1, Vector 2, Vector 3...
  ├─ Namespace: project-manager
  │   └─ Vector 1, Vector 2, Vector 3...
  └─ Namespace: sql
      └─ Vector 1, Vector 2, Vector 3...
```

## Code Comparison

### Using Separate Indexes (Current Approach)
```python
# Need separate connections for each index
java_index = pc.Index("java")
python_index = pc.Index("python")
business_analyst_index = pc.Index("business-analyst")

# Insert into Java index
java_index.upsert(vectors=[java_vector])

# Query Java index
results = java_index.query(vector=query_vector, top_k=10)
```

### Using Namespaces (Better Approach)
```python
# Single connection to one index
index = pc.Index("ats-resumes")

# Insert into Java namespace
index.upsert(vectors=[java_vector], namespace="java")

# Insert into Python namespace
index.upsert(vectors=[python_vector], namespace="python")

# Query Java namespace
results = index.query(vector=query_vector, namespace="java", top_k=10)

# Query Python namespace
results = index.query(vector=query_vector, namespace="python", top_k=10)
```

## Similarities

✅ **Both provide isolation** - Data in one namespace/index doesn't mix with another
✅ **Both support filtering** - Can filter within namespace/index
✅ **Same query performance** - Namespaces are just as fast
✅ **Same vector operations** - upsert, query, delete work the same way

## Differences

| Feature | Indexes | Namespaces |
|---------|---------|------------|
| **Limit** | 5 (free plan) | Unlimited |
| **Isolation** | Complete | Logical (within same index) |
| **Connection** | Separate per index | Single connection |
| **Quota** | Separate per index | Shared across namespaces |
| **Management** | More complex | Simpler |
| **Cost** | Higher (multiple indexes) | Lower (one index) |

## When to Use Each

### Use Separate Indexes When:
- You need completely separate vector spaces
- Different dimensions or metrics
- Different access controls
- You have a paid plan with higher limits

### Use Namespaces When:
- You want to organize data by category (like profile types)
- You're on a free plan with index limits
- All data has same dimension/metric
- You want simpler management

## For Your Use Case

**Perfect for namespaces!** Because:
- ✅ All profiles use same dimension (1536)
- ✅ All use same metric (cosine)
- ✅ Just organizing by profile type
- ✅ You've hit the 5-index limit

## Migration Path

You can keep your existing indexes (java, python, dotnet) and:
1. Use them as-is for those profile types
2. Use namespaces in `ats-resumes` for the other 4 types
3. Gradually migrate to all namespaces later

Or:
1. Keep one main index (`ats-resumes`)
2. Use namespaces for all profile types
3. Delete the separate indexes to free up space

## Example: Your Setup

### Current (5 indexes - at limit)
```
ats-resumes (general)
ats-resumes-1536 (duplicate?)
java (Java profiles)
python (Python profiles)
dotnet (.NET profiles)
```

### With Namespaces (1 index, 7 namespaces)
```
ats-resumes
  ├─ namespace: java
  ├─ namespace: python
  ├─ namespace: dotnet
  ├─ namespace: business-analyst
  ├─ namespace: project-manager
  ├─ namespace: sql
  └─ namespace: others
```

This gives you the same functionality but within the free plan limit!

