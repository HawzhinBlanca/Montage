"""Thread-safe database connection pool and utilities"""

import logging
import contextlib
import time
import psycopg2
from psycopg2 import pool, extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import Dict, List, Any, Optional, Tuple
import threading
from config import Config

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations"""
    pass


class DatabasePool:
    """Thread-safe connection pool manager"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=Config.MIN_POOL_SIZE,
                maxconn=Config.MAX_POOL_SIZE,
                host=Config.POSTGRES_HOST,
                port=Config.POSTGRES_PORT,
                database=Config.POSTGRES_DB,
                user=Config.POSTGRES_USER,
                password=Config.POSTGRES_PASSWORD,
                cursor_factory=extras.RealDictCursor
            )
            self._initialized = True
            logger.info(f"Database pool initialized with {Config.MIN_POOL_SIZE}-{Config.MAX_POOL_SIZE} connections")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseError(f"Database pool initialization failed: {e}")
    
    @contextlib.contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic cleanup"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextlib.contextmanager
    def get_cursor(self, commit=True):
        """Get a cursor with automatic transaction management"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def close_all(self):
        """Close all connections in the pool"""
        if hasattr(self, 'pool'):
            self.pool.closeall()
            logger.info("Database pool closed")


# Singleton instance
db_pool = DatabasePool()


class Database:
    """High-level database operations"""
    
    def __init__(self):
        self.pool = db_pool
    
    def execute(self, query: str, params: Optional[Tuple] = None, commit: bool = True) -> Optional[List[Dict]]:
        """Execute a query and return results"""
        with self.pool.get_cursor(commit=commit) as cursor:
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            return None
    
    def execute_many(self, query: str, params_list: List[Tuple], commit: bool = True) -> None:
        """Execute a query multiple times with different parameters"""
        with self.pool.get_cursor(commit=commit) as cursor:
            cursor.executemany(query, params_list)
    
    def insert(self, table: str, data: Dict[str, Any], returning: Optional[str] = 'id') -> Optional[Any]:
        """Insert a record and return specified column"""
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ', '.join(['%s'] * len(values))
        columns_str = ', '.join(columns)
        
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
        set_parts = [f"{k} = %s" for k in data.keys()]
        where_parts = [f"{k} = %s" for k in where.keys()]
        
        query = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        params = list(data.values()) + list(where.values())
        
        with self.pool.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    
    def find_one(self, table: str, where: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """Find a single record"""
        query = f"SELECT * FROM {table}"
        params = None
        
        if where:
            where_parts = [f"{k} = %s" for k in where.keys()]
            query += f" WHERE {' AND '.join(where_parts)}"
            params = list(where.values())
        
        query += " LIMIT 1"
        
        results = self.execute(query, params)
        return results[0] if results else None
    
    def find_many(self, table: str, where: Optional[Dict[str, Any]] = None, 
                  order_by: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Find multiple records"""
        query = f"SELECT * FROM {table}"
        params = []
        
        if where:
            where_parts = [f"{k} = %s" for k in where.keys()]
            query += f" WHERE {' AND '.join(where_parts)}"
            params.extend(where.values())
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute(query, tuple(params)) or []
    
    def count(self, table: str, where: Optional[Dict[str, Any]] = None) -> int:
        """Count records in a table"""
        query = f"SELECT COUNT(*) as count FROM {table}"
        params = None
        
        if where:
            where_parts = [f"{k} = %s" for k in where.keys()]
            query += f" WHERE {' AND '.join(where_parts)}"
            params = tuple(where.values())
        
        result = self.execute(query, params)
        return result[0]['count'] if result else 0
    
    def transaction(self):
        """Create a transaction context manager"""
        return Transaction(self.pool)


class Transaction:
    """Transaction context manager for complex operations"""
    
    def __init__(self, pool):
        self.pool = pool
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        self.conn = self.pool.pool.getconn()
        self.cursor = self.conn.cursor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.conn.rollback()
        else:
            self.conn.commit()
        
        self.cursor.close()
        self.pool.pool.putconn(self.conn)
        
        return False  # Don't suppress exceptions
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> Optional[List[Dict]]:
        """Execute a query within the transaction"""
        self.cursor.execute(query, params)
        if self.cursor.description:
            return self.cursor.fetchall()
        return None
    
    def savepoint(self, name: str):
        """Create a savepoint"""
        self.cursor.execute(f"SAVEPOINT {name}")
    
    def rollback_to_savepoint(self, name: str):
        """Rollback to a savepoint"""
        self.cursor.execute(f"ROLLBACK TO SAVEPOINT {name}")


# Helper functions
def with_retry(func, max_attempts: int = 3, delay: float = 1.0):
    """Retry a database operation on failure"""
    for attempt in range(max_attempts):
        try:
            return func()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            if attempt < max_attempts - 1:
                logger.warning(f"Database operation failed, retrying in {delay}s: {e}")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
    

def test_connection():
    """Test the database connection"""
    try:
        db = Database()
        result = db.execute("SELECT 1 as test")
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection
    logging.basicConfig(level=logging.INFO)
    test_connection()