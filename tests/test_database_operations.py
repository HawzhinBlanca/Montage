"""
Comprehensive tests for database operations and security
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from montage.api.auth import APIKeyAuth, UserRole
from montage.core.db import Database, DatabasePool


class TestDatabasePool:
    """Test database connection pool functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.db_pool = DatabasePool()

    def test_validate_table_name_valid(self):
        """Test validation of valid table names"""
        valid_names = [
            "jobs", "api_keys", "video_job", "cost_log",
            "transcript_cache", "highlight", "alert_log"
        ]

        for name in valid_names:
            assert self.db_pool._validate_table_name(name), f"'{name}' should be valid"

    def test_validate_table_name_invalid(self):
        """Test rejection of invalid table names"""
        invalid_names = [
            "'; DROP TABLE jobs; --",  # SQL injection
            "users; DELETE FROM jobs",  # Multiple statements
            "../../../etc/passwd",      # Path traversal
            "table-name",              # Invalid characters
            "123invalid",              # Starts with number
            "",                        # Empty string
            "very_long_table_name_that_might_cause_issues_" * 10,  # Too long
        ]

        for name in invalid_names:
            assert not self.db_pool._validate_table_name(name), f"'{name}' should be invalid"

    def test_validate_column_name_valid(self):
        """Test validation of valid column names"""
        valid_names = [
            "id", "user_id", "created_at", "updated_at",
            "job_status", "video_path", "api_key_hash"
        ]

        for name in valid_names:
            assert self.db_pool._validate_column_name(name), f"'{name}' should be valid"

    def test_validate_column_name_invalid(self):
        """Test rejection of invalid column names"""
        invalid_names = [
            "'; DROP TABLE",  # SQL injection
            "col-name",       # Invalid characters
            "123col",         # Starts with number
            "",               # Empty string
            None,             # None value
            "user id",        # Space in name
        ]

        for name in invalid_names:
            assert not self.db_pool._validate_column_name(name), f"'{name}' should be invalid"

    def test_quote_identifier(self):
        """Test SQL identifier quoting"""
        # Normal identifiers
        assert self.db_pool._quote_identifier("table_name") == '"table_name"'
        assert self.db_pool._quote_identifier("column") == '"column"'

        # Identifiers with quotes (should be escaped)
        assert self.db_pool._quote_identifier('table"name') == '"table""name"'
        assert self.db_pool._quote_identifier('col"umn"name') == '"col""umn""name"'

    @patch('src.core.db.pool.ThreadedConnectionPool')
    def test_pool_initialization_success(self, mock_pool_class):
        """Test successful database pool initialization"""
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool

        with patch('src.core.db.DATABASE_URL', 'postgresql://user:pass@localhost/db'):
            db_pool = DatabasePool()

            # Should initialize pool
            mock_pool_class.assert_called_once()
            assert db_pool.pool == mock_pool

    @patch('src.core.db.pool.ThreadedConnectionPool')
    def test_pool_initialization_failure(self, mock_pool_class):
        """Test database pool initialization failure handling"""
        mock_pool_class.side_effect = Exception("Connection failed")

        with patch('src.core.db.DATABASE_URL', 'postgresql://user:pass@localhost/db'):
            db_pool = DatabasePool()

            # Should handle failure gracefully
            assert db_pool.pool is None

    def test_non_postgresql_rejection(self):
        """Test non-PostgreSQL database rejection"""
        with patch('src.core.db.DATABASE_URL', 'mysql://test.db'):
            db_pool = DatabasePool()

            # Should skip non-PostgreSQL database initialization
            assert db_pool.pool is None

    @patch('src.core.db.pool.ThreadedConnectionPool')
    def test_get_connection_success(self, mock_pool_class):
        """Test getting connection from pool"""
        mock_pool = Mock()
        mock_conn = Mock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        with patch('src.core.db.DATABASE_URL', 'postgresql://user:pass@localhost/db'):
            db_pool = DatabasePool()

            with db_pool.get_connection() as conn:
                assert conn == mock_conn

            # Should return connection to pool
            mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_get_connection_no_pool(self):
        """Test getting connection when pool is None"""
        db_pool = DatabasePool()
        db_pool.pool = None

        with db_pool.get_connection() as conn:
            assert conn is None

    @patch('src.core.db.pool.ThreadedConnectionPool')
    def test_get_cursor_success(self, mock_pool_class):
        """Test getting cursor with auto-commit"""
        mock_pool = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        with patch('src.core.db.DATABASE_URL', 'postgresql://user:pass@localhost/db'):
            db_pool = DatabasePool()

            with db_pool.get_cursor() as cursor:
                assert cursor == mock_cursor

            # Should commit and close cursor
            mock_conn.commit.assert_called_once()
            mock_cursor.close.assert_called_once()

    @patch('src.core.db.pool.ThreadedConnectionPool')
    def test_get_cursor_exception_rollback(self, mock_pool_class):
        """Test cursor exception handling with rollback"""
        mock_pool = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        with patch('src.core.db.DATABASE_URL', 'postgresql://user:pass@localhost/db'):
            db_pool = DatabasePool()

            with pytest.raises(Exception):
                with db_pool.get_cursor() as cursor:
                    raise Exception("Test exception")

            # Should rollback on exception
            mock_conn.rollback.assert_called_once()
            mock_cursor.close.assert_called_once()


class TestDatabase:
    """Test Database class functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.db = Database()

    @patch('src.core.db.get_db_pool')
    def test_query_execution(self, mock_get_pool):
        """Test query execution with parameters"""
        # Mock database pool and connection
        mock_pool = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [{"id": 1, "name": "test"}]
        mock_cursor.description = [("id",), ("name",)]
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Execute query
        result = self.db.query("SELECT * FROM jobs WHERE id = %s", (1,))

        # Verify execution
        mock_cursor.execute.assert_called_once_with("SELECT * FROM jobs WHERE id = %s", (1,))
        assert result == [{"id": 1, "name": "test"}]

    @patch('src.core.db.get_db_pool')
    def test_query_no_results(self, mock_get_pool):
        """Test query execution with no results"""
        mock_pool = Mock()
        mock_cursor = Mock()
        mock_cursor.description = None  # No results
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        result = self.db.query("INSERT INTO jobs VALUES (%s)", ("test",))

        assert result is None

    @patch('src.core.db.get_db_pool')
    def test_insert_operation(self, mock_get_pool):
        """Test insert operation"""
        mock_pool = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"id": 123}
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Test insert
        result = self.db.insert(
            "jobs",
            {"user_id": "test_user", "status": "queued"},
            returning="id"
        )

        # Verify query construction
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        query, params = call_args

        assert "INSERT INTO" in query
        assert '"jobs"' in query  # Table name should be quoted
        assert "test_user" in params
        assert "queued" in params
        assert result == 123

    @patch('src.core.db.get_db_pool')
    def test_insert_invalid_table(self, mock_get_pool):
        """Test insert with invalid table name"""
        mock_pool = Mock()
        mock_get_pool.return_value = mock_pool

        with pytest.raises(ValueError, match="Invalid table name"):
            self.db.insert(
                "'; DROP TABLE jobs; --",
                {"user_id": "test"}
            )

    @patch('src.core.db.get_db_pool')
    def test_update_operation(self, mock_get_pool):
        """Test update operation"""
        mock_pool = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 1
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Test update
        rows_affected = self.db.update(
            "jobs",
            {"status": "completed"},
            {"id": 123}
        )

        # Verify query construction
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        query, params = call_args

        assert "UPDATE" in query
        assert '"jobs"' in query
        assert "completed" in params
        assert 123 in params
        assert rows_affected == 1

    @patch('src.core.db.get_db_pool')
    def test_find_one_operation(self, mock_get_pool):
        """Test find_one operation"""
        mock_pool = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"id": 1, "status": "completed"}
        mock_cursor.description = [("id",), ("status",)]
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Test find_one
        result = self.db.find_one(
            "jobs",
            where={"id": 1},
            columns=["id", "status"]
        )

        # Verify query construction
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        query, params = call_args

        assert "SELECT" in query
        assert '"id"' in query
        assert '"status"' in query
        assert "WHERE" in query
        assert 1 in params
        assert result == {"id": 1, "status": "completed"}

    @patch('src.core.db.get_db_pool')
    def test_find_operation(self, mock_get_pool):
        """Test find operation with limit and order"""
        mock_pool = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "status": "completed"},
            {"id": 2, "status": "processing"}
        ]
        mock_cursor.description = [("id",), ("status",)]
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Test find with options
        result = self.db.find(
            "jobs",
            where={"user_id": "test_user"},
            order_by="created_at DESC",
            limit=10
        )

        # Verify query construction
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        query, params = call_args

        assert "SELECT" in query
        assert "ORDER BY" in query
        assert "LIMIT" in query
        assert "test_user" in params
        assert len(result) == 2

    @patch('src.core.db.get_db_pool')
    def test_delete_operation(self, mock_get_pool):
        """Test delete operation"""
        mock_pool = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 5
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Test delete
        rows_deleted = self.db.delete(
            "jobs",
            {"status": "failed"}
        )

        # Verify query construction
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        query, params = call_args

        assert "DELETE FROM" in query
        assert '"jobs"' in query
        assert "WHERE" in query
        assert "failed" in params
        assert rows_deleted == 5

    @patch('src.core.db.get_db_pool')
    def test_count_operation(self, mock_get_pool):
        """Test count operation"""
        mock_pool = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"count": 42}
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Test count
        count = self.db.count("jobs", {"status": "completed"})

        # Verify query construction
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        query, params = call_args

        assert "SELECT COUNT(*)" in query
        assert "completed" in params
        assert count == 42


class TestAPIKeyDatabaseOperations:
    """Test API key database operations"""

    def setup_method(self):
        """Setup test fixtures"""
        self.api_key_auth = APIKeyAuth()

    @patch('src.api.auth.Database')
    def test_ensure_api_keys_table(self, mock_db_class):
        """Test API keys table creation"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db = Mock()
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db_class.return_value = mock_db

        api_key_auth = APIKeyAuth()

        # Should create table and index
        assert mock_cursor.execute.call_count >= 2  # Table + index creation
        mock_conn.commit.assert_called()

    @patch('src.api.auth.Database')
    def test_generate_api_key_database_storage(self, mock_db_class):
        """Test that API key generation stores hash in database"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db = Mock()
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db_class.return_value = mock_db

        api_key_auth = APIKeyAuth()

        # Generate key
        raw_key = api_key_auth.generate_api_key("test_user", "Test Key", UserRole.USER)

        # Verify database operations
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()

        # Check that hash was stored, not raw key
        call_args = mock_cursor.execute.call_args[0]
        query, params = call_args

        assert "INSERT INTO api_keys" in query
        assert raw_key not in str(params)  # Raw key should not be in DB
        assert "test_user" in params
        assert "Test Key" in params
        assert "user" in params

    @patch('src.api.auth.Database')
    def test_validate_api_key_database_lookup(self, mock_db_class):
        """Test API key validation database lookup"""
        mock_conn = Mock()
        mock_cursor = Mock()
        test_time = datetime.now()
        mock_cursor.fetchone.return_value = (
            "test_user", "Test Key", "user", ["video:upload"], test_time, None
        )
        mock_conn.cursor.return_value = mock_cursor
        mock_db = Mock()
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db_class.return_value = mock_db

        api_key_auth = APIKeyAuth()

        # Validate key
        user = api_key_auth.validate_api_key("test_api_key")

        # Verify database operations
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()  # Should update last_used

        # Check query used hash for lookup
        call_args = mock_cursor.execute.call_args_list[0][0]
        query, params = call_args

        assert "SELECT" in query
        assert "FROM api_keys" in query
        assert "WHERE key_hash =" in query
        assert "test_api_key" not in str(params)  # Raw key should not be in query

        # Verify user object
        assert user.user_id == "test_user"
        assert user.username == "Test Key"
        assert user.role == UserRole.USER


class TestDatabaseErrorHandling:
    """Test database error handling"""

    @patch('src.core.db.get_db_pool')
    def test_connection_error_handling(self, mock_get_pool):
        """Test handling of database connection errors"""
        mock_pool = Mock()
        mock_pool.get_cursor.side_effect = Exception("Connection failed")
        mock_get_pool.return_value = mock_pool

        db = Database()

        # Should handle connection errors gracefully
        with pytest.raises(Exception, match="Connection failed"):
            db.query("SELECT 1")

    @patch('src.core.db.get_db_pool')
    def test_query_error_handling(self, mock_get_pool):
        """Test handling of SQL query errors"""
        mock_pool = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("SQL syntax error")
        mock_pool.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        db = Database()

        # Should propagate SQL errors
        with pytest.raises(Exception, match="SQL syntax error"):
            db.query("INVALID SQL")

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        db = Database()

        # Test invalid table name
        with pytest.raises(ValueError):
            db.insert("'; DROP TABLE", {"data": "value"})

        # Test empty data
        with pytest.raises(ValueError):
            db.insert("jobs", {})

        # Test invalid column names
        with pytest.raises(ValueError):
            db.update("jobs", {"'; DROP": "value"}, {"id": 1})
