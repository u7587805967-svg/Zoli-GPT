import aiosqlite

async def init_db(db_path: str):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

async def save_message(db_path: str, role: str, content: str):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO chat_history (role, content) VALUES (?, ?)",
            (role, content)
        )
        await db.commit()

async def get_chat_history(db_path: str, limit: int = 50):
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?", 
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return rows[::-1]