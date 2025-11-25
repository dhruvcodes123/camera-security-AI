# Database Logic Documentation

## **🎯 Problem Solved**

**Original Issue**: The system was only storing data for newly identified persons, but not for re-identified persons.

**Solution**: Implemented a **dual database system** that separates:
1. **Vector Database**: Only stores new person embeddings (for re-identification matching)
2. **SQLite Database**: Stores ALL person entries (both new and re-identified) for tracking history

## **📊 Database Schema**

### **SQLite Database (`database.sqlite`)**

```sql
CREATE TABLE person_tracking (
    unique_id TEXT PRIMARY KEY,           -- UUID for each tracking session
    person_id INTEGER NOT NULL,           -- Permanent person ID (1, 2, 3...)
    entry_timestamp TIMESTAMP NOT NULL,   -- When person entered frame
    is_new_person BOOLEAN DEFAULT FALSE,  -- True if newly identified
    match_type TEXT DEFAULT 'reidentified', -- Type of match ('new', 'fused', etc.)
    similarity_score REAL,                -- Similarity score if re-identified
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### **Vector Database (`fused.index`)**

- **Purpose**: Only stores embeddings for **new persons**
- **Usage**: For similarity matching during re-identification
- **Storage**: FAISS index with 3840D fused embeddings

## **🔄 Logic Flow**

### **1. New Person Detection**
```python
# When a person is NEWLY identified:
# ✅ Vector Database: ADD embedding for similarity matching
self.fused_db.add_person(new_id, fused_embedding)

# ✅ SQLite Database: ADD entry for tracking history
self.person_tracking_db.record_person_entry(
    person_id=new_id,
    is_new_person=True,
    match_type="new",
    similarity_score=None
)
```

### **2. Re-Identified Person Detection**
```python
# When a person is RE-IDENTIFIED:
# ❌ Vector Database: DO NOT ADD (already exists)
# ✅ SQLite Database: ADD entry for tracking history
self.person_tracking_db.record_person_entry(
    person_id=existing_id,
    is_new_person=False,
    match_type="fused",
    similarity_score=0.85  # Actual similarity score
)
```

## **📈 Data Storage Examples**

### **Example 1: New Person**
```sql
-- Vector Database (fused.index)
-- Stores: 3840D embedding for Person ID 1

-- SQLite Database (database.sqlite)
INSERT INTO person_tracking VALUES (
    'uuid-1234-5678',  -- unique_id
    1,                  -- person_id
    '2024-07-30 14:30:00',  -- entry_timestamp
    TRUE,               -- is_new_person
    'new',              -- match_type
    NULL,               -- similarity_score
    '2024-07-30 14:30:00'   -- created_at
);
```

### **Example 2: Re-Identified Person**
```sql
-- Vector Database (fused.index)
-- NO CHANGE - embedding already exists

-- SQLite Database (database.sqlite)
INSERT INTO person_tracking VALUES (
    'uuid-5678-9012',  -- unique_id
    1,                  -- person_id (same person)
    '2024-07-30 14:35:00',  -- entry_timestamp (different time)
    FALSE,              -- is_new_person
    'fused',            -- match_type
    0.85,               -- similarity_score
    '2024-07-30 14:35:00'   -- created_at
);
```

## **🎯 Key Features**

| Feature | Vector Database | SQLite Database |
|---------|----------------|-----------------|
| **New Persons** | ✅ Store embedding | ✅ Store entry |
| **Re-Identified** | ❌ No storage | ✅ Store entry |
| **Purpose** | Similarity matching | Tracking history |
| **Data Type** | 3840D embeddings | Entry timestamps |
| **Query Type** | Similarity search | Historical queries |

## **📊 Query Examples**

### **Get Person History**
```python
# Get all entries for Person ID 1
history = tracking_db.get_person_tracking_history(1)
# Returns: All entries (new + re-identified) for Person ID 1
```

### **Get New Persons Only**
```python
# Get only newly identified persons
new_entries = tracking_db.get_new_persons_only()
# Returns: Only entries where is_new_person = TRUE
```

### **Get Re-Identified Persons Only**
```python
# Get only re-identified persons
reid_entries = tracking_db.get_reidentified_persons_only()
# Returns: Only entries where is_new_person = FALSE
```

### **Get Summary Statistics**
```python
# Get summary for specific person
summary = tracking_db.get_tracking_summary(person_id=1)
# Returns: {
#   'total_entries': 3,
#   'new_person_entries': 1,
#   'reidentified_entries': 2,
#   'first_seen': '2024-07-30 14:30:00',
#   'last_seen': '2024-07-30 14:35:00',
#   'avg_similarity': 0.85
# }
```

## **🔧 Integration Points**

### **1. PersonTracker (`tracking/tracker.py`)**
```python
# Records ALL entries (new + re-identified)
self.person_tracking_db.record_person_entry(
    person_id=persistent_id,
    is_new_person=is_new_person,
    match_type=match_type,
    similarity_score=fusion_similarity
)
```

### **2. UnifiedReIdentificationSystem (`scripts/reid_system.py`)**
```python
# Only stores new person embeddings in Vector Database
if new_person:
    self.fused_db.add_person(new_id, fused_embedding)
```

### **3. VectorDatabase (`db/database.py`)**
```python
# Only called for new persons
def add_person(self, person_id, embedding, view_type="default"):
    # Stores embedding in FAISS index
    # Stores embedding in SQLite table
```

## **✅ Benefits**

1. **Complete Tracking History**: All person entries are recorded
2. **Efficient Vector Storage**: Only new embeddings stored for matching
3. **Detailed Analytics**: Track new vs re-identified persons separately
4. **Performance Optimized**: Vector database doesn't grow unnecessarily
5. **Flexible Queries**: Can query by person, type, time, etc.

## **🎯 Conclusion**

The new system provides:
- **Vector Database**: Efficient similarity matching (new persons only)
- **SQLite Database**: Complete tracking history (all persons)
- **Separation of Concerns**: Each database has a specific purpose
- **Comprehensive Analytics**: Full visibility into person tracking patterns

This solves the original problem where re-identified persons were not being tracked in the database. 