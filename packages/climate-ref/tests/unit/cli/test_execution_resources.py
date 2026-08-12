"""Unit tests for the ``ref executions resources`` command."""

import datetime
import json

import pytest
from climate_ref_pmp import provider as pmp_provider

from climate_ref.cli.executions import _format_duration, _parse_since
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.provider_registry import _register_provider

GIB = 1024**3


def _add_execution(session, group_id: int, index: int, **columns):
    session.add(
        Execution(
            execution_group_id=group_id,
            output_fragment=f"out{group_id}-{index}",
            dataset_hash=f"hash{group_id}-{index}",
            **columns,
        )
    )


@pytest.fixture
def db_with_resources(db_seeded):
    """
    Two pmp diagnostics with measurements, one comfortably inside its limit and one over it.

    ``enso_tel`` peaks at 4 GiB against an 8 GiB limit.
    ``extratropical-modes-of-variability-nao`` peaks at 20 GiB against the same limit,
    and also carries a failed execution that recorded nothing.
    """
    with db_seeded.session.begin():
        _register_provider(db_seeded, pmp_provider)

        groups = {}
        for key, slug in [
            ("enso", "enso_tel"),
            ("nao", "extratropical-modes-of-variability-nao"),
        ]:
            diagnostic = db_seeded.session.query(Diagnostic).filter_by(slug=slug).first()
            group = ExecutionGroup(key=key, diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            groups[key] = group
        db_seeded.session.flush()

        for index in range(12):
            _add_execution(
                db_seeded.session,
                groups["enso"].id,
                index,
                successful=True,
                wall_seconds=840.0,
                cpu_seconds=924.0,
                peak_memory_bytes=4 * GIB,
                memory_source="cgroup",
                memory_limit_bytes=8 * GIB,
                resources_exclusive=True,
            )

        for index in range(3):
            _add_execution(
                db_seeded.session,
                groups["nao"].id,
                index,
                successful=True,
                wall_seconds=3000.0,
                cpu_seconds=11400.0,
                peak_memory_bytes=20 * GIB,
                memory_source="cgroup",
                memory_limit_bytes=8 * GIB,
                resources_exclusive=True,
            )
        _add_execution(db_seeded.session, groups["nao"].id, 100, successful=False)

    db_seeded.session.commit()
    return db_seeded


class TestParseSince:
    @pytest.mark.parametrize(
        "value, seconds",
        [("12h", 12 * 3600), ("90d", 90 * 86400), ("4w", 4 * 604800)],
    )
    def test_relative(self, value, seconds):
        now = datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None)

        parsed = _parse_since(value)

        assert abs((now - parsed).total_seconds() - seconds) < 5

    def test_iso_date(self):
        assert _parse_since("2026-07-01") == datetime.datetime(2026, 7, 1)

    @pytest.mark.parametrize("value", ["", "soon", "12x", "nonsense-d"])
    def test_rejects_nonsense(self, value):
        with pytest.raises(ValueError, match="Invalid --since value"):
            _parse_since(value)


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds, expected",
        [(45.0, "45s"), (840.0, "14m"), (3120.0, "52m"), (7200.0, "2.0h")],
    )
    def test_units(self, seconds, expected):
        assert _format_duration(seconds) == expected


class TestResourcesCommand:
    def test_table(self, db_with_resources, invoke_cli):
        result = invoke_cli(["executions", "resources"])

        assert "diagnostic" in result.stdout
        assert "headroom" in result.stdout
        assert "enso_tel" in result.stdout
        assert "extratropical-modes-of-variability-nao" in result.stdout
        # Sorted by recommended memory descending, so the heavy diagnostic leads.
        assert result.stdout.index("extratropical") < result.stdout.index("enso_tel")

    def test_flags_over_limit_and_qualifies_it(self, db_with_resources, invoke_cli):
        result = invoke_cli(["executions", "resources"])

        assert "OVER" in result.stdout
        # The thin sample count and the failure count both have to be visible.
        assert "thin" in result.stdout
        assert "good" in result.stdout
        assert "biased low" in result.stderr
        assert "fewer than 10 samples" in result.stderr

    def test_json_still_warns_about_bias(self, db_with_resources, invoke_cli):
        """stderr is the only place a JSON caller sees the bias signals, so it still has to carry them."""
        result = invoke_cli(["executions", "resources", "--format", "json"])

        # stdout stays machine readable, the warnings go alongside it.
        json.loads(result.stdout)
        assert "biased low" in result.stderr
        assert "fewer than 10 samples" in result.stderr

    def test_by_provider(self, db_with_resources, invoke_cli):
        result = invoke_cli(["executions", "resources", "--by", "provider"])

        assert "provider" in result.stdout
        assert "pmp" in result.stdout
        assert "enso_tel" not in result.stdout

    def test_filters(self, db_with_resources, invoke_cli):
        result = invoke_cli(["executions", "resources", "--diagnostic", "enso_tel"])

        assert "enso_tel" in result.stdout
        assert "extratropical" not in result.stdout

    def test_json(self, db_with_resources, invoke_cli):
        result = invoke_cli(["executions", "resources", "--format", "json"])

        records = {record["diagnostic"]: record for record in json.loads(result.stdout)}

        assert records["enso_tel"]["n_samples"] == 12
        assert records["enso_tel"]["confidence"] == "good"
        assert records["enso_tel"]["memory_source"] == "cgroup"
        assert records["enso_tel"]["peak_memory_p95"] == 4 * GIB
        assert records["enso_tel"]["recommended_memory_bytes"] == 6 * GIB
        assert records["enso_tel"]["over_limit"] is False

        nao = records["extratropical-modes-of-variability-nao"]
        assert nao["n_samples"] == 3
        assert nao["n_failed"] == 1
        assert nao["confidence"] == "thin"
        assert nao["headroom_ratio"] == pytest.approx(0.4)
        assert nao["over_limit"] is True
        assert nao["recommended_cpus"] == 4

    def test_safety_factor(self, db_with_resources, invoke_cli):
        result = invoke_cli(["executions", "resources", "--safety", "2.0", "--format", "json"])

        records = {record["diagnostic"]: record for record in json.loads(result.stdout)}

        assert records["enso_tel"]["recommended_memory_bytes"] == 8 * GIB

    def test_include_shared(self, db_seeded, invoke_cli):
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            diagnostic = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
            group = ExecutionGroup(key="shared", diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            db_seeded.session.flush()
            _add_execution(
                db_seeded.session,
                group.id,
                0,
                successful=True,
                wall_seconds=10.0,
                cpu_seconds=10.0,
                peak_memory_bytes=3 * GIB,
                memory_source="cgroup",
                resources_exclusive=False,
            )
        db_seeded.session.commit()

        default = invoke_cli(["executions", "resources", "--format", "json"])
        assert json.loads(default.stdout)[0]["n_samples"] == 0

        shared = invoke_cli(["executions", "resources", "--include-shared", "--format", "json"])
        assert json.loads(shared.stdout)[0]["n_samples"] == 1

    def test_since_filters_everything_out(self, db_with_resources, invoke_cli):
        result = invoke_cli(["executions", "resources", "--since", "2099-01-01"])

        assert "No resource measurements found." in result.stdout

    def test_invalid_since(self, db_with_resources, invoke_cli):
        result = invoke_cli(["executions", "resources", "--since", "yesterday"], expected_exit_code=1)

        assert "Invalid --since value" in result.stderr

    def test_no_measurements(self, db_seeded, invoke_cli):
        result = invoke_cli(["executions", "resources"])

        assert "No resource measurements found." in result.stdout
