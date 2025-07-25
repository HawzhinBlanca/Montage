"""Secure database module with SQL injection prevention"""

import re
import logging
import os
from psycopg2 import extras, pool
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from ..config import DATABASE_URL
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATABASE_URL

# Fallback to SQLite for local development/testing
if not DATABASE_URL or DATABASE_URL == "postgresql://":
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./montage-local.db")
    logger.warning("No PostgreSQL configured, using SQLite fallback")

# Fail fast in production
if DATABASE_URL and DATABASE_URL.startswith("sqlite") and os.getenv("ENV") == "prod":
    logger.critical("PostgreSQL required in production - set DATABASE_URL")
    raise RuntimeError("PostgreSQL required in production - set DATABASE_URL")


class DatabaseError(Exception):
    """Database operation error"""

    pass


class DatabasePool:
    """Thread-safe connection pool with SQL injection prevention"""

    # Whitelist of allowed table names to prevent SQL injection
    ALLOWED_TABLES = {
        "video_job",
        "api_cost_log",
        "job_checkpoint",
        "transcript_cache",
        "highlight",
        "alert_log",
        "job",
        "checkpoint",
        "cost_log",
    }

    def __init__(self):
        self.pool = None
        self._init_pool()

    def _validate_table_name(self, table: str) -> bool:
        """Validate table name against whitelist"""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
            return False
        return table in self.ALLOWED_TABLES

    def _validate_column_name(self, column: str) -> bool:
        """Validate column name to prevent injection"""
        return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", column))

    def _quote_identifier(self, identifier: str) -> str:
        """SECURITY: Safely quote a SQL identifier (table/column name)"""
        # Double any quotes in the identifier
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _init_pool(self):
        """Initialize connection pool"""
        try:
            # Skip PostgreSQL pool if using SQLite fallback
            if DATABASE_URL.startswith("sqlite"):
                logger.warning("SQLite mode - database operations will be disabled")
                self.pool = None
                return

            self.pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=DATABASE_URL,
                cursor_factory=extras.RealDictCursor,
            )
            logger.info("Database pool initialized: min=1, max=10")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            # Don't fail hard - allow pipeline to continue without DB
            logger.warning("Continuing without database - caching disabled")
            self.pool = None

    @contextmanager
    def get_connection(self):
        """Get a connection from pool"""
        if self.pool is None:
            # No database available - yield None
            yield None
            return

        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit: bool = True):
        """Get a cursor with automatic commit/rollback"""
        with self.get_connection() as conn:
            if conn is None:
                # No database - yield None
                yield None
                return

            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()

    def close(self):
        """Close all connections in pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database pool closed")


# Global pool instance
_db_pool = None


def get_db_pool():
    """Get or create the database pool instance"""
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool()
    return _db_pool


class SecureDatabase:
    """High-level database operations with SQL injection prevention"""

    def __init__(self):
        self.pool = get_db_pool()

    def execute(
        self, query: str, params: Optional[Tuple] = None, commit: bool = True
    ) -> Optional[List[Dict]]:
        """Execute a query and return results"""
        with self.pool.get_cursor(commit=commit) as cursor:
            if cursor is None:
                logger.debug("No database connection - skipping query")
                return None
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            return None

    def execute_many(
        self, query: str, params_list: List[Tuple], commit: bool = True
    ) -> None:
        """Execute a query multiple times with different parameters"""
        with self.pool.get_cursor(commit=commit) as cursor:
            if cursor is None:
                logger.debug("No database connection - skipping execute_many")
                return
            cursor.executemany(query, params_list)

    def insert(
        self, table: str, data: Dict[str, Any], returning: Optional[str] = "id"
    ) -> Optional[Any]:
        """Insert a record and return specified column"""
        # Validate table name to prevent SQL injection
        if not self.pool._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")

        # Validate column names
        for column in data.keys():
            if not self.pool._validate_column_name(column):
                raise ValueError(f"Invalid column name: {column}")

        if returning and not self.pool._validate_column_name(returning):
            raise ValueError(f"Invalid returning column: {returning}")

        columns = list(data.keys())
        values = list(data.values())
        placeholders = ", ".join(["%s"] * len(values))

        # SECURITY: Use quoted identifiers to prevent SQL injection
        quoted_table = self.pool._quote_identifier(table)
        quoted_columns = [self.pool._quote_identifier(col) for col in columns]
        columns_str = ", ".join(quoted_columns)

        query = f"INSERT INTO {quoted_table} ({columns_str}) VALUES ({placeholders})"
        if returning:
            quoted_returning = self.pool._quote_identifier(returning)
            query += f" RETURNING {quoted_returning}"

        with self.pool.get_cursor() as cursor:
            if cursor is None:
                logger.debug("No database connection - skipping insert")
                return None
            cursor.execute(query, values)
            if returning:
                result = cursor.fetchone()
                return result[returning] if result else None
            return None

    def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
        """Update records and return affected row count"""
        # Validate table name
        if not self.pool._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")

        # Validate all column names
        for column in list(data.keys()) + list(where.keys()):
            if not self.pool._validate_column_name(column):
                raise ValueError(f"Invalid column name: {column}")

        # SECURITY: Use quoted identifiers
        quoted_table = self.pool._quote_identifier(table)
        set_parts = [f"{self.pool._quote_identifier(k)} = %s" for k in data.keys()]
        where_parts = [f"{self.pool._quote_identifier(k)} = %s" for k in where.keys()]

        query = f"UPDATE {quoted_table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        params = list(data.values()) + list(where.values())

        with self.pool.get_cursor() as cursor:
            if cursor is None:
                logger.debug("No database connection - skipping update")
                return 0
            cursor.execute(query, params)
            return cursor.rowcount

    def find_one(
        self, table: str, where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict]:
        """Find a single record"""
        # Validate table name
        if not self.pool._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")

        # SECURITY: Use quoted identifiers
        quoted_table = self.pool._quote_identifier(table)
        query = f"SELECT * FROM {quoted_table}"
        params = None

        if where:
            # Validate column names
            for column in where.keys():
                if not self.pool._validate_column_name(column):
                    raise ValueError(f"Invalid column name: {column}")

            where_parts = [
                f"{self.pool._quote_identifier(k)} = %s" for k in where.keys()
            ]
            query += f" WHERE {' AND '.join(where_parts)}"
            params = list(where.values())

        query += " LIMIT 1"

        results = self.execute(query, params)
        return results[0] if results else None

    def find_many(
        self,
        table: str,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Find multiple records"""
        # Validate table name
        if not self.pool._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")

        # SECURITY: Use quoted identifiers
        quoted_table = self.pool._quote_identifier(table)
        query = f"SELECT * FROM {quoted_table}"
        params: List[Any] = []

        if where:
            # Validate column names
            for column in where.keys():
                if not self.pool._validate_column_name(column):
                    raise ValueError(f"Invalid column name: {column}")

            where_parts = [
                f"{self.pool._quote_identifier(k)} = %s" for k in where.keys()
            ]
            query += f" WHERE {' AND '.join(where_parts)}"
            params.extend(where.values())

        if order_by:
            # SECURITY: Parse and quote ORDER BY clause
            order_match = re.match(
                r"^([a-zA-Z_][a-zA-Z0-9_]*)(\s+(ASC|DESC))?$", order_by
            )
            if not order_match:
                raise ValueError(f"Invalid ORDER BY clause: {order_by}")
            column_name = order_match.group(1)
            direction = order_match.group(3) or ""
            quoted_column = self.pool._quote_identifier(column_name)
            query += f" ORDER BY {quoted_column} {direction}"

        if limit:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"Invalid LIMIT value: {limit}")
            query += f" LIMIT {limit}"

        return self.execute(query, tuple(params)) or []

    def delete(self, table: str, where: Dict[str, Any]) -> int:
        """Delete records and return affected row count"""
        # Validate table name
        if not self.pool._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")

        if not where:
            raise ValueError("WHERE clause is required for DELETE")

        # Validate column names
        for column in where.keys():
            if not self.pool._validate_column_name(column):
                raise ValueError(f"Invalid column name: {column}")

        # SECURITY: Use quoted identifiers
        quoted_table = self.pool._quote_identifier(table)
        where_parts = [f"{self.pool._quote_identifier(k)} = %s" for k in where.keys()]
        query = f"DELETE FROM {quoted_table} WHERE {' AND '.join(where_parts)}"

        with self.pool.get_cursor() as cursor:
            if cursor is None:
                logger.debug("No database connection - skipping delete")
                return 0
            cursor.execute(query, list(where.values()))
            return cursor.rowcount

    def count(self, table: str, where: Optional[Dict[str, Any]] = None) -> int:
        """Count records in table"""
        # Validate table name
        if not self.pool._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")

        # SECURITY: Use quoted identifiers
        quoted_table = self.pool._quote_identifier(table)
        query = f"SELECT COUNT(*) as count FROM {quoted_table}"
        params = None

        if where:
            # Validate column names
            for column in where.keys():
                if not self.pool._validate_column_name(column):
                    raise ValueError(f"Invalid column name: {column}")

            where_parts = [
                f"{self.pool._quote_identifier(k)} = %s" for k in where.keys()
            ]
            query += f" WHERE {' AND '.join(where_parts)}"
            params = list(where.values())

        result = self.execute(query, params)
        return result[0]["count"] if result else 0

    @contextmanager
    def transaction(self):
        """Execute multiple operations in a transaction"""
        with self.pool.get_connection() as conn:
            if conn is None:
                # No database - yield None
                yield None
                return

            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()


# Alias for backward compatibility
Database = SecureDatabase


# Create a proxy object for backward compatibility with db_pool
class _DBPoolProxy:
    def __getattr__(self, name):
        return getattr(get_db_pool(), name)


db_pool = _DBPoolProxy()
