#!/usr/bin/env python3
"""Database migration runner"""

import os
import sys
import glob
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from legacy_config import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class MigrationRunner:
    def __init__(self):
        self.config = Config

    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist"""
        try:
            # Connect to postgres database to create our target database
            conn = psycopg2.connect(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD,
                database="postgres",
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config.POSTGRES_DB,),
            )
            exists = cursor.fetchone()

            if not exists:
                cursor.execute(f"CREATE DATABASE {self.config.POSTGRES_DB}")
                logger.info(f"Created database: {self.config.POSTGRES_DB}")
            else:
                logger.info(f"Database already exists: {self.config.POSTGRES_DB}")

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise

    def create_migrations_table(self, conn):
        """Create migrations tracking table"""
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """
        )
        conn.commit()
        cursor.close()
        logger.info("Migrations table ready")

    def get_applied_migrations(self, conn):
        """Get list of already applied migrations"""
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
        applied = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return applied

    def apply_migration(self, conn, migration_file):
        """Apply a single migration file"""
        version = os.path.basename(migration_file)

        logger.info(f"Applying migration: {version}")

        with open(migration_file, "r") as f:
            sql = f.read()

        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            cursor.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
            )
            conn.commit()
            logger.info(f"Successfully applied: {version}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to apply {version}: {e}")
            raise
        finally:
            cursor.close()

    def run(self):
        """Run all pending migrations"""
        try:
            # Validate configuration
            self.config.validate()

            # Create database if needed
            self.create_database_if_not_exists()

            # Connect to target database
            conn = psycopg2.connect(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD,
                database=self.config.POSTGRES_DB,
            )

            # Create migrations table
            self.create_migrations_table(conn)

            # Get applied migrations
            applied = self.get_applied_migrations(conn)

            # Find migration files
            migration_files = sorted(glob.glob("migrations/*.sql"))

            if not migration_files:
                logger.info("No migration files found")
                return

            # Apply pending migrations
            pending_count = 0
            for migration_file in migration_files:
                version = os.path.basename(migration_file)
                if version not in applied:
                    self.apply_migration(conn, migration_file)
                    pending_count += 1

            if pending_count == 0:
                logger.info("All migrations already applied")
            else:
                logger.info(f"Applied {pending_count} migration(s)")

            conn.close()

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    runner = MigrationRunner()
    runner.run()
