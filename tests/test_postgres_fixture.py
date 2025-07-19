"""Real PostgreSQL test fixture using testcontainers"""

import pytest
import psycopg2
import logging
from testcontainers.postgres import PostgresContainer
from typing import Dict, Generator, Any
import time
import os

logger = logging.getLogger(__name__)


class PostgresTestFixture:
    """Real PostgreSQL test environment using Docker containers"""

    def __init__(self):
        self.container = None
        self.connection = None
        self.cursor = None

    def start(self) -> Dict[str, Any]:
        """Start PostgreSQL container and return connection details"""
        logger.info("Starting PostgreSQL test container...")

        self.container = PostgresContainer(
            image="postgres:15-alpine",
            port=5432,
            username="testuser",
            password="testpass",  # pragma: allowlist secret
            dbname="testdb",
        )

        # Start container
        self.container.start()

        # Wait for container to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                conn = psycopg2.connect(
                    host=self.container.get_container_host_ip(),
                    port=self.container.get_exposed_port(5432),
                    user="testuser",
                    password="testpass",  # pragma: allowlist secret
                    database="testdb",
                )
                conn.close()
                break
            except psycopg2.OperationalError:
                if i == max_retries - 1:
                    raise
                time.sleep(0.5)

        connection_info = {
            "host": self.container.get_container_host_ip(),
            "port": self.container.get_exposed_port(5432),
            "user": "testuser",
            "password": "testpass",
            "database": "testdb",
            "url": f"postgresql://testuser:testpass@{self.container.get_container_host_ip()}:{self.container.get_exposed_port(5432)}/testdb",  # pragma: allowlist secret
        }

        logger.info(f"PostgreSQL container ready at {connection_info['url']}")
        return connection_info

    def get_connection(self):
        """Get a fresh database connection"""
        if not self.container:
            raise RuntimeError("Container not started")

        return psycopg2.connect(
            host=self.container.get_container_host_ip(),
            port=self.container.get_exposed_port(5432),
            user="testuser",
            password="testpass",  # pragma: allowlist secret
            database="testdb",
        )

    def execute_sql_file(self, sql_file_path: str):
        """Execute SQL file against the test database"""
        if not os.path.exists(sql_file_path):
            raise FileNotFoundError(f"SQL file not found: {sql_file_path}")

        with open(sql_file_path, "r") as f:
            sql_content = f.read()

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql_content)
            conn.commit()
            cursor.close()
        finally:
            conn.close()

        logger.info(f"Executed SQL file: {sql_file_path}")

    def reset_database(self):
        """Reset database to clean state"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Drop all tables
            cursor.execute(
                """
                DROP SCHEMA public CASCADE;
                CREATE SCHEMA public;
                GRANT ALL ON SCHEMA public TO testuser;
                GRANT ALL ON SCHEMA public TO public;
            """
            )

            conn.commit()
            cursor.close()
        finally:
            conn.close()

        logger.info("Database reset to clean state")

    def stop(self):
        """Stop and cleanup container"""
        if self.container:
            self.container.stop()
            logger.info("PostgreSQL container stopped")


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresTestFixture, None, None]:
    """Session-scoped PostgreSQL container fixture"""
    fixture = PostgresTestFixture()

    try:
        connection_info = fixture.start()

        # Set environment variables for other tests
        os.environ["TEST_POSTGRES_HOST"] = connection_info["host"]
        os.environ["TEST_POSTGRES_PORT"] = str(connection_info["port"])
        os.environ["TEST_POSTGRES_USER"] = connection_info["user"]
        os.environ["TEST_POSTGRES_PASSWORD"] = connection_info["password"]
        os.environ["TEST_POSTGRES_DB"] = connection_info["database"]
        os.environ["TEST_DATABASE_URL"] = connection_info["url"]

        yield fixture

    finally:
        fixture.stop()


@pytest.fixture(scope="function")
def clean_postgres(postgres_container: PostgresTestFixture) -> PostgresTestFixture:
    """Function-scoped fixture that resets database before each test"""
    postgres_container.reset_database()

    # Load base schema if it exists
    schema_file = "tests/data/schema.sql"
    if os.path.exists(schema_file):
        postgres_container.execute_sql_file(schema_file)

    return postgres_container


@pytest.fixture(scope="function")
def postgres_connection(clean_postgres: PostgresTestFixture):
    """Get a database connection for a test"""
    conn = clean_postgres.get_connection()
    conn.autocommit = False

    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def test_postgres_fixture_basic(clean_postgres: PostgresTestFixture):
    """Test that PostgreSQL fixture works correctly"""
    conn = clean_postgres.get_connection()

    try:
        cursor = conn.cursor()

        # Test basic connectivity
        cursor.execute("SELECT version()")
        result = cursor.fetchone()
        assert result[0].startswith("PostgreSQL")

        # Test table creation
        cursor.execute(
            """
            CREATE TABLE test_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """
        )

        # Test data insertion
        cursor.execute("INSERT INTO test_table (name) VALUES (%s)", ("test_name",))

        # Test data retrieval
        cursor.execute("SELECT name FROM test_table WHERE id = 1")
        result = cursor.fetchone()
        assert result[0] == "test_name"

        conn.commit()
        cursor.close()

    finally:
        conn.close()


def test_postgres_isolation(clean_postgres: PostgresTestFixture):
    """Test that tests are properly isolated"""
    conn = clean_postgres.get_connection()

    try:
        cursor = conn.cursor()

        # This table should not exist from previous test
        cursor.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'test_table'
        """
        )
        result = cursor.fetchone()
        assert result[0] == 0  # Table should not exist

        cursor.close()

    finally:
        conn.close()


if __name__ == "__main__":
    # Manual test of the fixture
    logging.basicConfig(level=logging.INFO)

    fixture = PostgresTestFixture()
    try:
        connection_info = fixture.start()
        print(f"Container started: {connection_info}")

        # Test connection
        conn = fixture.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 'Hello PostgreSQL!'")
        result = cursor.fetchone()
        print(f"Query result: {result[0]}")
        cursor.close()
        conn.close()

    finally:
        fixture.stop()
