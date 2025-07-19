"""Secure database module with SQL injection prevention"""

import re
import logging
import psycopg2
from psycopg2 import extras
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

from legacy_config import Config

logger = logging.getLogger(__name__)


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

    def _init_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=Config.MIN_POOL_SIZE,
                maxconn=Config.MAX_POOL_SIZE,
                host=Config.POSTGRES_HOST,
                port=Config.POSTGRES_PORT,
                database=Config.POSTGRES_DB,
                user=Config.POSTGRES_USER,
                password=Config.POSTGRES_PASSWORD,
                cursor_factory=extras.RealDictCursor,
            )
            logger.info(
                f"Database pool initialized: min={Config.MIN_POOL_SIZE}, max={Config.MAX_POOL_SIZE}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseError(f"Database pool initialization failed: {e}")

    @contextmanager
    def get_connection(self):
        """Get a connection from pool"""
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
db_pool = DatabasePool()


class SecureDatabase:
    """High-level database operations with SQL injection prevention"""

    def __init__(self):
        self.pool = db_pool

    def execute(
        self, query: str, params: Optional[Tuple] = None, commit: bool = True
    ) -> Optional[List[Dict]]:
        """Execute a query and return results"""
        with self.pool.get_cursor(commit=commit) as cursor:
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            return None

    def execute_many(
        self, query: str, params_list: List[Tuple], commit: bool = True
    ) -> None:
        """Execute a query multiple times with different parameters"""
        with self.pool.get_cursor(commit=commit) as cursor:
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
        columns_str = ", ".join(columns)

        query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
        if returning:
            query += f" RETURNING {returning}"

        with self.pool.get_cursor() as cursor:
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

        set_parts = [f"{k} = %s" for k in data.keys()]
        where_parts = [f"{k} = %s" for k in where.keys()]

        query = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        params = list(data.values()) + list(where.values())

        with self.pool.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def find_one(
        self, table: str, where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict]:
        """Find a single record"""
        # Validate table name
        if not self.pool._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")

        query = f"SELECT * FROM {table}"
        params = None

        if where:
            # Validate column names
            for column in where.keys():
                if not self.pool._validate_column_name(column):
                    raise ValueError(f"Invalid column name: {column}")

            where_parts = [f"{k} = %s" for k in where.keys()]
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

        query = f"SELECT * FROM {table}"
        params = []

        if where:
            # Validate column names
            for column in where.keys():
                if not self.pool._validate_column_name(column):
                    raise ValueError(f"Invalid column name: {column}")

            where_parts = [f"{k} = %s" for k in where.keys()]
            query += f" WHERE {' AND '.join(where_parts)}"
            params.extend(where.values())

        if order_by:
            # Validate order by clause
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\s+(ASC|DESC))?$", order_by):
                raise ValueError(f"Invalid ORDER BY clause: {order_by}")
            query += f" ORDER BY {order_by}"

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

        where_parts = [f"{k} = %s" for k in where.keys()]
        query = f"DELETE FROM {table} WHERE {' AND '.join(where_parts)}"

        with self.pool.get_cursor() as cursor:
            cursor.execute(query, list(where.values()))
            return cursor.rowcount

    def count(self, table: str, where: Optional[Dict[str, Any]] = None) -> int:
        """Count records in table"""
        # Validate table name
        if not self.pool._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")

        query = f"SELECT COUNT(*) as count FROM {table}"
        params = None

        if where:
            # Validate column names
            for column in where.keys():
                if not self.pool._validate_column_name(column):
                    raise ValueError(f"Invalid column name: {column}")

            where_parts = [f"{k} = %s" for k in where.keys()]
            query += f" WHERE {' AND '.join(where_parts)}"
            params = list(where.values())

        result = self.execute(query, params)
        return result[0]["count"] if result else 0

    @contextmanager
    def transaction(self):
        """Execute multiple operations in a transaction"""
        with self.pool.get_connection() as conn:
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
