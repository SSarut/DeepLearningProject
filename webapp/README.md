# FastAPI DL Backend

This is a simple FastAPI backend for hosting Deep Learning models with a Redis queue and PostgreSQL database (via Docker).

## Development Setup

We use `uv` and `nix` to manage dependencies.

1. **Install Dependencies**:
```bash
nix develop -c uv pip install -r requirements.txt
```

2. **Run Local Services**:
You can run Redis and Postgres locally using Docker:
```bash
docker-compose up redis db -d
```

3. **Run Application**:
```bash
nix develop -c uvicorn app.main:app --reload --port 8080
```

## Docker Compose Deployment

To run the full stack (FastAPI + Redis + Postgres) together using Docker Compose:
```bash
docker-compose up --build -d
```
*   The FastAPI application runs on `http://localhost:8080`
*   PostgreSQL is exposed on `localhost:5433` (User: `postgres`, Password: `postgrespassword`, DB: `appdb`)
*   Redis is exposed on `localhost:6379`

### Postgres Usage Snippet
If you need to connect to the PostgreSQL database from Python (e.g., using `psycopg2` or `asyncpg`/`SQLAlchemy`), you can use the connection string parsed from the `DATABASE_URL` environment variable.

```python
import os
import psycopg2 # Make sure to install psycopg2-binary or asyncpg

# Inside Docker, the database URL is:
# postgresql://postgres:postgrespassword@db:5432/appdb

# From your host machine, the database URL is:
# postgresql://postgres:postgrespassword@localhost:5433/appdb

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgrespassword@localhost:5433/appdb")

# Example using SQLAlchemy:
# from sqlalchemy import create_engine
# engine = create_engine(DB_URL)
# connection = engine.connect()

# Example using psycopg2:
# conn = psycopg2.connect(DB_URL)
# cursor = conn.cursor()
# cursor.execute("SELECT version();")
# print(cursor.fetchone())
# cursor.close()
# conn.close()
```
