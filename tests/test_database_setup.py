import pytest
import psycopg2

try:
    from src.config import get
except ImportError:
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.config import get


class TestDatabaseSetup:
    """Test that the database schema was deployed correctly"""

    @pytest.fixture
    def db_connection(self):
        """Create a test database connection"""
        # Parse DATABASE_URL or use individual env vars
        database_url = get(
            "DATABASE_URL",
            "postgresql://postgres:pass@localhost:5432/postgres",  # pragma: allowlist secret
        )

        if database_url.startswith("postgresql://"):
            conn = psycopg2.connect(database_url)
        else:
            conn = psycopg2.connect(
                host=get("POSTGRES_HOST", "localhost"),
                port=int(get("POSTGRES_PORT", "5432")),
                user=get("POSTGRES_USER", "postgres"),
                password=get("POSTGRES_PASSWORD", "password"),
                database=get("POSTGRES_DB", "postgres"),
            )
        yield conn
        conn.close()

    def test_tables_exist(self, db_connection):
        """Verify all required tables exist"""
        cursor = db_connection.cursor()

        expected_tables = [
            "video_job",
            "transcript_cache",
            "highlight",
            "job_checkpoint",
            "api_cost_log",
            "processing_metrics",
            "schema_migrations",
        ]

        cursor.execute(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """
        )

        actual_tables = [row[0] for row in cursor.fetchall()]

        for table in expected_tables:
            assert table in actual_tables, f"Table {table} does not exist"

        cursor.close()

    def test_video_job_columns(self, db_connection):
        """Verify video_job table has all required columns"""
        cursor = db_connection.cursor()

        cursor.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'video_job'
        """
        )

        columns = {
            row[0]: {"type": row[1], "nullable": row[2]} for row in cursor.fetchall()
        }

        # Check required columns exist
        required_columns = ["id", "src_hash", "status", "created_at", "input_path"]
        for col in required_columns:
            assert col in columns, f"Column {col} missing from video_job table"

        # Check data types
        assert columns["src_hash"]["type"] == "character"
        assert columns["status"]["nullable"] == "NO"

        cursor.close()

    def test_indexes_exist(self, db_connection):
        """Verify performance indexes were created"""
        cursor = db_connection.cursor()

        cursor.execute(
            """
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public'
        """
        )

        indexes = [row[0] for row in cursor.fetchall()]

        expected_indexes = [
            "idx_video_job_status",
            "idx_video_job_created_at",
            "idx_transcript_cache_src_hash",
            "idx_highlight_job_id",
            "idx_highlight_score",
        ]

        for idx in expected_indexes:
            assert idx in indexes, f"Index {idx} does not exist"

        cursor.close()

    def test_uuid_extension(self, db_connection):
        """Verify uuid-ossp extension is installed"""
        cursor = db_connection.cursor()

        cursor.execute(
            """
            SELECT extname 
            FROM pg_extension 
            WHERE extname = 'uuid-ossp'
        """
        )

        result = cursor.fetchone()
        assert result is not None, "uuid-ossp extension not installed"

        cursor.close()

    def test_foreign_keys(self, db_connection):
        """Verify foreign key constraints are properly set up"""
        cursor = db_connection.cursor()

        cursor.execute(
            """
            SELECT 
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
        """
        )

        foreign_keys = cursor.fetchall()

        # Check that highlight table references video_job
        highlight_fk = next(
            (fk for fk in foreign_keys if fk[0] == "highlight" and fk[1] == "job_id"),
            None,
        )
        assert highlight_fk is not None, "highlight.job_id foreign key not found"
        assert (
            highlight_fk[2] == "video_job"
        ), "highlight.job_id should reference video_job"

        cursor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
