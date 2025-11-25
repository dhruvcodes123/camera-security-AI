import sqlite3
import numpy as np
import faiss
import os
from pathlib import Path

def normalize_embeddings(embeddings):
    """Normalizes a single or multiple embeddings to unit length."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        if norm == 0:
            return embeddings
        return embeddings / norm
    else:
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return embeddings / norm

class VectorDatabase:
    def __init__(self, db_type: str, embedding_dim: int, db_path="db/database.sqlite", index_dir="db/vector_db"):
        if db_type not in ['face', 'gait', 'gait_shoe', 'fused']:
            raise ValueError("db_type must be either 'face', 'gait', 'gait_shoe', or 'fused'")
        
        self.db_type = db_type
        self.embedding_dim = embedding_dim
        self.db_path = db_path
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / f"{self.db_type}.index"
        self.table_name = f"{self.db_type}_embeddings"

        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(Path(self.db_path).parent, exist_ok=True)
        
        self._create_tables()
        self.index = self._load_or_create_faiss_index()
        
        if self.index.ntotal == 0:
            with self._get_db_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT person_id, embedding FROM {self.table_name}")
                rows = cursor.fetchall()
                if rows:
                    embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows], dtype=np.float32)
                    if embeddings.size > 0:
                        self.index.add(embeddings)
                        # print(f"Loaded {len(rows)} '{self.db_type}' embeddings into FAISS index from DB.")

    def _get_db_conn(self):
        return sqlite3.connect(self.db_path)

    def _load_or_create_faiss_index(self):
        if os.path.exists(self.index_path):
            # print(f"Loading FAISS index from {self.index_path}...")
            return faiss.read_index(str(self.index_path))
        else:
            # print(f"Creating new FAISS IP (Cosine Similarity) index at {self.index_path} with dim={self.embedding_dim}.")
            return faiss.IndexFlatIP(self.embedding_dim)

    def _create_tables(self):
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                embedding BLOB,
                view_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )""")
            conn.commit()

    def get_next_person_id(self):
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM persons")
            max_id = cursor.fetchone()[0]
        return (max_id or 0) + 1

    def add_person(self, person_id, embedding, view_type="default"):
        if person_id is None:
            return

        normalized_embedding = normalize_embeddings(embedding)

        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO persons (id) VALUES (?)", (person_id,))
            cursor.execute(
                f"INSERT INTO {self.table_name} (person_id, embedding, view_type) VALUES (?, ?, ?)",
                (person_id, normalized_embedding.tobytes(), view_type)
            )
            conn.commit()
        
        embedding_to_add = np.array([normalized_embedding], dtype=np.float32)
        self.index.add(embedding_to_add)
        # print(f"Added new '{self.db_type}' embedding for Person ID {person_id} to DB and FAISS index.")

    def query(self, query_embedding, k=1):
        if self.index.ntotal == 0:
            return None, None
            
        normalized_query = normalize_embeddings(query_embedding).reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(normalized_query, k)
        
        if len(indices) == 0 or indices[0][0] == -1:
            return None, None
            
        db_id = indices[0][0] + 1 
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT person_id FROM {self.table_name} WHERE id=?", (int(db_id),))
            result = cursor.fetchone()

        if result:
            person_id = result[0]
            similarity = distances[0][0]
            return person_id, similarity
        return None, None

    def get_person_views(self, person_id):
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT DISTINCT view_type FROM {self.table_name} WHERE person_id=?", (person_id,))
            return {row[0] for row in cursor.fetchall()}

    def log_reappearance(self, person_id, new_embedding, view_type="default"):
        self.add_person(person_id, new_embedding, view_type)
        # print(f"RE-APPEARANCE: Logged new '{self.db_type}' embedding for existing Person ID {person_id}.")

    def clear_all_for_type(self):
        # print(f"Clearing data for type '{self.db_type}'...")
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
            # print(f"Removed index: {self.index_path}")
        
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            # print(f"Dropped table: {self.table_name}")
            conn.commit()

    def close(self):
        # print(f"Closing '{self.db_type}' database and saving FAISS index...")
        faiss.write_index(self.index, str(self.index_path))
        # print(f"Saved index to {self.index_path}")

def clear_entire_database(db_path="db/database.sqlite", index_dir="db/vector_db"):
    """Deletes the main SQLite database and all FAISS index files."""
    print("--- CLEARING ENTIRE DATABASE ---")
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed main database: {db_path}")

    index_dir = Path(index_dir)
    if index_dir.exists():
        for f in index_dir.glob("*.index"):
            os.remove(f)
            print(f"Removed index file: {f}")
    print("--- ENTIRE DATABASE CLEARED ---")


if __name__ == '__main__':
    try:
        print("--- Database Example ---")
        clear_entire_database()

        face_db = VectorDatabase(db_type='face', embedding_dim=512)
        gait_db = VectorDatabase(db_type='gait', embedding_dim=256)

        print("\n--- Simulating Face Detection ---")
        face_embedding = np.random.rand(512)
        person_id = face_db.get_next_person_id()
        face_db.add_person(person_id, face_embedding, view_type="frontal_face")
        
        print("\n--- Simulating Gait Detection (Same Person) ---")
        gait_embedding = np.random.rand(256)
        gait_db.add_person(person_id, gait_embedding, view_type="side_gait")

        print("\n--- Simulating New Face Detection ---")
        new_face_embedding = np.random.rand(512)
        
        matched_id, similarity = face_db.query(new_face_embedding)
        
        if matched_id is not None and similarity is not None and similarity > 0.9:
            print(f"Face query returned match with Person ID {matched_id} (Similarity: {similarity:.4f})")
        else:
            print("No confident face match. Assigning new ID.")
            new_person_id = face_db.get_next_person_id()
            face_db.add_person(new_person_id, new_face_embedding, "frontal_face")

        face_db.close()
        gait_db.close()

    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")
        import traceback
        traceback.print_exc() 