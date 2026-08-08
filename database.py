import sqlite3
import datetime
import hashlib
import secrets
from contextlib import contextmanager

def hash_password(password: str, salt: str = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    # 100 000 iterációs PBKDF2 védelem szivárványtáblák és brute-force ellen
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return key.hex(), salt

class DatabaseRepository:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_file, check_same_thread=False, timeout=10.0)
        try: yield conn
        finally: conn.close()

    def _init_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, content TEXT, type TEXT, caption TEXT, timestamp TEXT, thread_id TEXT DEFAULT 'default')''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS document_vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, doc_name TEXT, chunk_text TEXT, embedding BLOB, file_size TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS latency_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, duration REAL, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS token_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, tokens INTEGER, cost REAL, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS system_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT, timestamp TEXT)''')
            
            # Dinamikus sémafrissítések a visszafelé kompatibilitásért
            try: cursor.execute("ALTER TABLE chat_history ADD COLUMN thread_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE token_logs ADD COLUMN username TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE token_logs ADD COLUMN tokens INTEGER")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE token_logs ADD COLUMN cost REAL")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE token_logs ADD COLUMN timestamp TEXT")
            except sqlite3.OperationalError: pass
            
            # ÚJ: Biztonsági só (salt) oszlop hozzáadása a meglévő adatbázishoz
            try: cursor.execute("ALTER TABLE users ADD COLUMN salt TEXT")
            except sqlite3.OperationalError: pass
            
            conn.commit()

    def get_user(self, username: str):
        """Lekéri a felhasználó adatait (jelszó hash és salt) a felhasználónév alapján."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Ellenőrizzük, hogy létezik-e a salt oszlop, és annak megfelelően kérdezzük le
            try:
                cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
                return cursor.fetchone()
            except sqlite3.OperationalError:
                # Fallback, ha a salt oszlop még nem létezne
                cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()
                if row:
                    return row[0], None
                return None

    def create_user(self, username: str, password_hash: str, salt: str):
        """Létrehoz egy új felhasználót az adatbázisban."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, password_hash, salt)
            )
            conn.commit()
