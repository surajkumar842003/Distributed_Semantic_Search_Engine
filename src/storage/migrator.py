"""Database migration manager for PostgreSQL schema versions."""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import asyncpg

from src.common.logging import get_logger

logger = get_logger("storage.migrator")

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


class DatabaseMigrator:
    """Manages versioned SQL schema migrations for PostgreSQL."""

    def __init__(self, migrations_dir: Optional[Path] = None):
        self.migrations_dir = migrations_dir or DEFAULT_MIGRATIONS_DIR

    def discover_migrations(self) -> List[Tuple[int, str, Path]]:
        """Discovers and sorts all SQL migration files in the migrations directory.

        Files must follow the pattern: {version}_{name}.sql (e.g. 001_initial_schema.sql).
        """
        if not self.migrations_dir.is_dir():
            logger.warning(f"Migrations directory does not exist: {self.migrations_dir}")
            return []

        migrations = []
        for file_path in sorted(self.migrations_dir.glob("*.sql")):
            filename = file_path.name
            parts = filename.split("_", 1)
            try:
                version = int(parts[0])
                name = parts[1].replace(".sql", "") if len(parts) > 1 else filename
                migrations.append((version, name, file_path))
            except ValueError:
                logger.warning(f"Skipping migration file with invalid version prefix: {filename}")

        return sorted(migrations, key=lambda x: x[0])

    async def ensure_migrations_table(self, conn: asyncpg.Connection):
        """Creates the schema_migrations tracking table if it does not exist."""
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )

    async def get_applied_versions(self, conn: asyncpg.Connection) -> List[int]:
        """Returns list of already applied migration version numbers."""
        await self.ensure_migrations_table(conn)
        rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version ASC;")
        return [r["version"] for r in rows]

    async def apply_migrations(self, conn_or_pool: Any) -> List[str]:
        """Applies all pending migrations in ascending order inside transactions.

        Returns:
            List[str]: Names of applied migrations.
        """
        migrations = self.discover_migrations()
        if not migrations:
            logger.info("No migration files found.")
            return []

        # Acquire connection
        if isinstance(conn_or_pool, asyncpg.Pool):
            async with conn_or_pool.acquire() as conn:
                return await self._apply_with_conn(conn, migrations)
        else:
            return await self._apply_with_conn(conn_or_pool, migrations)

    async def _apply_with_conn(
        self, conn: asyncpg.Connection, migrations: List[Tuple[int, str, Path]]
    ) -> List[str]:
        await self.ensure_migrations_table(conn)
        applied_versions = set(await self.get_applied_versions(conn))

        applied_now: List[str] = []
        for version, name, file_path in migrations:
            if version in applied_versions:
                continue

            logger.info(f"Applying migration {version:03d}_{name} from {file_path.name}...")
            with open(file_path, "r", encoding="utf-8") as f:
                sql_content = f.read()

            async with conn.transaction():
                await conn.execute(sql_content)
                await conn.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES ($1, $2);",
                    version,
                    name,
                )
            applied_now.append(f"{version:03d}_{name}")
            logger.info(f"Successfully applied migration {version:03d}_{name}")

        if not applied_now:
            logger.info("All database migrations are already up to date.")
        return applied_now

