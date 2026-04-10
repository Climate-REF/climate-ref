"""
Database management commands
"""

from typing import Annotated

import sqlalchemy
import typer
from alembic.script import ScriptDirectory
from rich.table import Table

from climate_ref.config import Config
from climate_ref.database import Database, _create_backup, _get_database_revision, _get_sqlite_path

app = typer.Typer(help=__doc__)


def _get_script_directory(db: Database, config: Config) -> ScriptDirectory:
    """Build an Alembic ScriptDirectory from a Database and Config."""
    alembic_cfg = db.alembic_config(config)
    return ScriptDirectory.from_config(alembic_cfg)


@app.command()
def migrate(ctx: typer.Context) -> None:
    """
    Run database migrations to bring the schema up to date.

    This applies any pending Alembic migrations. A backup is created
    before migrating (SQLite only).
    """
    db = ctx.obj.database_unmigrated
    config = ctx.obj.config
    console = ctx.obj.console

    script = _get_script_directory(db, config)
    head_rev = script.get_current_head()

    with db._engine.connect() as connection:
        current_rev = _get_database_revision(connection)

    if current_rev == head_rev:
        console.print(f"Database is already up to date at revision [bold]{current_rev}[/bold].")
        return

    console.print(f"Current revision: [yellow]{current_rev or '(empty)'}[/yellow]")
    console.print(f"Target revision:  [green]{head_rev}[/green]")
    console.print("Running migrations...")

    db.migrate(config, skip_backup=False)
    console.print("[green]Migrations applied successfully.[/green]")


@app.command()
def status(ctx: typer.Context) -> None:
    """
    Check if the database schema is up to date.

    Shows the current revision, the latest available revision,
    and whether any migrations are pending.
    """
    db = ctx.obj.database_unmigrated
    config = ctx.obj.config
    console = ctx.obj.console

    script = _get_script_directory(db, config)
    head_rev = script.get_current_head()

    with db._engine.connect() as connection:
        current_rev = _get_database_revision(connection)

    console.print(f"Database URL:     [bold]{db.url}[/bold]")
    console.print(f"Current revision: [bold]{current_rev or '(empty)'}[/bold]")
    console.print(f"Head revision:    [bold]{head_rev}[/bold]")

    if current_rev == head_rev:
        console.print("[green]Database is up to date.[/green]")
    elif current_rev is None:
        console.print("[yellow]Database has no revision stamp (new or unmanaged).[/yellow]")
    else:
        console.print(
            "[yellow]Database is behind. Run 'ref db migrate' to apply pending migrations.[/yellow]"
        )


@app.command()
def heads(ctx: typer.Context) -> None:
    """
    Show the latest migration revision(s).
    """
    db = ctx.obj.database_unmigrated
    config = ctx.obj.config
    console = ctx.obj.console

    script = _get_script_directory(db, config)

    for head in script.get_heads():
        revision = script.get_revision(head)
        if revision is not None:
            console.print(f"[bold]{revision.revision}[/bold] — {revision.doc or '(no description)'}")


@app.command()
def history(
    ctx: typer.Context,
    last: Annotated[
        int | None,
        typer.Option("--last", "-n", help="Show only the last N migrations"),
    ] = None,
) -> None:
    """
    Show the migration history.
    """
    db = ctx.obj.database_unmigrated
    config = ctx.obj.config
    console = ctx.obj.console

    script = _get_script_directory(db, config)

    with db._engine.connect() as connection:
        current_rev = _get_database_revision(connection)

    revisions = list(script.walk_revisions())
    if last is not None:
        revisions = revisions[:last]

    table = Table(title="Migration History")
    table.add_column("Revision", style="bold")
    table.add_column("Description")
    table.add_column("Status")

    for rev in revisions:
        is_current = rev.revision == current_rev
        status_text = "[green]current[/green]" if is_current else ""
        table.add_row(
            rev.revision[:12],
            rev.doc or "(no description)",
            status_text,
        )

    console.print(table)


@app.command()
def backup(ctx: typer.Context) -> None:
    """
    Create a manual backup of the database (SQLite only).
    """
    config = ctx.obj.config
    console = ctx.obj.console

    db_path = _get_sqlite_path(config.db.database_url)
    if db_path is None:
        console.print("[red]Backup is only supported for local SQLite databases.[/red]")
        raise typer.Exit(1)

    if not db_path.exists():
        console.print(f"[red]Database file not found: {db_path}[/red]")
        raise typer.Exit(1)

    backup_path = _create_backup(db_path, config.db.max_backups)
    console.print(f"[green]Backup created at: {backup_path}[/green]")


@app.command()
def sql(
    ctx: typer.Context,
    query: Annotated[
        str,
        typer.Argument(help="SQL query to execute"),
    ],
) -> None:
    """
    Execute an arbitrary SQL query against the database.

    SELECT queries display results as a table.
    Other statements report the number of rows affected.
    """
    db = ctx.obj.database_unmigrated
    console = ctx.obj.console

    with db._engine.connect() as connection:
        result = connection.execute(sqlalchemy.text(query))

        if result.returns_rows:
            columns = list(result.keys())
            rows = result.fetchall()

            table = Table(title=f"Results ({len(rows)} rows)")
            for col in columns:
                table.add_column(str(col))

            for row in rows:
                table.add_row(*(str(v) for v in row))

            console.print(table)
        else:
            connection.commit()
            console.print(f"[green]Query executed successfully. Rows affected: {result.rowcount}[/green]")


@app.command()
def tables(ctx: typer.Context) -> None:
    """
    List all tables in the database.
    """
    db = ctx.obj.database_unmigrated
    console = ctx.obj.console

    with db._engine.connect() as connection:
        inspector = sqlalchemy.inspect(connection)
        table_names = inspector.get_table_names()

        table = Table(title="Database Tables")
        table.add_column("Table Name", style="bold")
        table.add_column("Columns", justify="right")

        for name in sorted(table_names):
            columns = inspector.get_columns(name)
            table.add_row(name, str(len(columns)))

        console.print(table)
