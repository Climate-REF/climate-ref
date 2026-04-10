def test_without_subcommand(invoke_cli):
    result = invoke_cli(["db"], expected_exit_code=2)
    assert "Missing command." in result.stderr


def test_db_help(invoke_cli):
    result = invoke_cli(["db", "--help"])

    assert "Database management commands" in result.stdout


class TestDbStatus:
    def test_status_fresh_database(self, invoke_cli):
        result = invoke_cli(["db", "status"])

        assert "Current revision:" in result.stdout
        assert "Head revision:" in result.stdout
        assert "Database URL:" in result.stdout
        assert "no revision stamp" in result.stdout

    def test_status_up_to_date(self, invoke_cli):
        # Migrate first, then check status
        invoke_cli(["db", "migrate"])

        result = invoke_cli(["db", "status"])

        assert "Database is up to date" in result.stdout


class TestDbMigrate:
    def test_migrate_fresh_database(self, invoke_cli):
        result = invoke_cli(["db", "migrate"])

        assert "Migrations applied successfully" in result.stdout

    def test_migrate_already_up_to_date(self, invoke_cli):
        # Migrate first
        invoke_cli(["db", "migrate"])

        # Second migrate should report up to date
        result = invoke_cli(["db", "migrate"])

        assert "already up to date" in result.stdout


class TestDbHeads:
    def test_heads(self, invoke_cli):
        result = invoke_cli(["db", "heads"])

        # Should show at least one head revision
        assert result.exit_code == 0
        assert result.stdout.strip() != ""


class TestDbHistory:
    def test_history(self, invoke_cli):
        result = invoke_cli(["db", "history"])

        assert "Migration History" in result.stdout
        assert "Revision" in result.stdout

    def test_history_last(self, invoke_cli):
        result = invoke_cli(["db", "history", "--last", "3"])

        assert "Migration History" in result.stdout


class TestDbBackup:
    def test_backup(self, invoke_cli):
        # Trigger DB creation first
        invoke_cli(["db", "migrate"])

        result = invoke_cli(["db", "backup"])

        assert "Backup created at" in result.stdout


class TestDbSql:
    def test_select_query(self, invoke_cli):
        # Trigger DB creation first
        invoke_cli(["db", "migrate"])

        result = invoke_cli(["db", "sql", "SELECT COUNT(*) AS cnt FROM provider"])

        assert "cnt" in result.stdout
        assert "Results" in result.stdout

    def test_select_empty_table(self, invoke_cli):
        invoke_cli(["db", "migrate"])

        result = invoke_cli(["db", "sql", "SELECT * FROM provider"])

        assert "Results (0 rows)" in result.stdout

    def test_update_query(self, invoke_cli):
        invoke_cli(["db", "migrate"])

        result = invoke_cli(
            ["db", "sql", "INSERT INTO provider (slug, name, version) VALUES ('test', 'Test', '1.0')"]
        )

        assert "Query executed successfully" in result.stdout


class TestDbTables:
    def test_tables(self, invoke_cli):
        invoke_cli(["db", "migrate"])

        result = invoke_cli(["db", "tables"])

        assert "Database Tables" in result.stdout
        assert "provider" in result.stdout
        assert "dataset" in result.stdout
        assert "execution" in result.stdout
