"""Unit tests for the per-execution resource read layer."""

import datetime

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.provider_registry import _register_provider
from climate_ref.results import Reader, ResourceFilter
from climate_ref.results.resources import _percentile

GIB = 1024**3


def _add_execution(session, group_id: int, index: int, **columns):
    """Add one execution to a group, writing the resource columns directly."""
    execution = Execution(
        execution_group_id=group_id,
        output_fragment=f"out{group_id}-{index}",
        dataset_hash=f"hash{group_id}-{index}",
        **columns,
    )
    session.add(execution)
    return execution


def _sample(**overrides):
    """A complete, exclusive, successful measurement, unless overridden."""
    base = dict(
        successful=True,
        wall_seconds=100.0,
        cpu_seconds=100.0,
        peak_memory_bytes=GIB,
        memory_source="cgroup",
        memory_limit_bytes=8 * GIB,
        resources_exclusive=True,
    )
    base.update(overrides)
    return base


@pytest.fixture
def db_with_resources(db_seeded):
    """
    A seeded DB carrying hand-written resource measurements.

    ``enso_tel`` (pmp) has ten clean exclusive samples with a rising memory peak,
    plus a shared sample, an incomplete sample and an unmeasured failure.
    ``extratropical-modes-of-variability-nao`` (pmp) has three heavy samples that blow the limit.
    ``enso-characteristics`` (esmvaltool) has a mixed set of memory sources.
    ``sea-ice-area-basic`` (esmvaltool) has only unmeasured executions.
    """
    with db_seeded.session.begin():
        _register_provider(db_seeded, pmp_provider)
        _register_provider(db_seeded, esmvaltool_provider)

        groups = {}
        for key, slug in [
            ("enso", "enso_tel"),
            ("nao", "extratropical-modes-of-variability-nao"),
            ("chars", "enso-characteristics"),
            ("seaice", "sea-ice-area-basic"),
        ]:
            diagnostic = db_seeded.session.query(Diagnostic).filter_by(slug=slug).first()
            group = ExecutionGroup(key=key, diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            groups[key] = group
        db_seeded.session.flush()

        # Peaks of 1..10 GiB, so p50 is 5.5 GiB and p95 is 9.55 GiB.
        for index in range(10):
            _add_execution(
                db_seeded.session,
                groups["enso"].id,
                index,
                **_sample(peak_memory_bytes=(index + 1) * GIB, cpu_seconds=110.0),
            )
        _add_execution(
            db_seeded.session,
            groups["enso"].id,
            100,
            **_sample(peak_memory_bytes=40 * GIB, resources_exclusive=False),
        )
        _add_execution(
            db_seeded.session, groups["enso"].id, 101, **_sample(cpu_seconds=None, peak_memory_bytes=2 * GIB)
        )
        _add_execution(db_seeded.session, groups["enso"].id, 102, successful=False)

        for index in range(3):
            _add_execution(
                db_seeded.session,
                groups["nao"].id,
                index,
                **_sample(peak_memory_bytes=20 * GIB, wall_seconds=3000.0, cpu_seconds=11400.0),
            )

        for index in range(3):
            _add_execution(db_seeded.session, groups["chars"].id, index, **_sample(peak_memory_bytes=4 * GIB))
        _add_execution(
            db_seeded.session,
            groups["chars"].id,
            100,
            **_sample(peak_memory_bytes=100 * GIB, memory_source="rusage"),
        )

        _add_execution(db_seeded.session, groups["seaice"].id, 0, successful=True)
        _add_execution(db_seeded.session, groups["seaice"].id, 1, successful=None)

    db_seeded.session.commit()
    return db_seeded


def _by_slug(profiles):
    return {profile.diagnostic_slug: profile for profile in profiles}


class TestPercentile:
    def test_single_value(self):
        assert _percentile([4.0], 0.95) == 4.0

    def test_interpolates(self):
        assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5

    def test_ignores_input_order(self):
        assert _percentile([3.0, 1.0, 2.0], 0.5) == _percentile([1.0, 2.0, 3.0], 0.5)


class TestProfiles:
    def test_percentiles(self, db_with_resources):
        profile = _by_slug(Reader(db_with_resources).resources.profiles())["enso_tel"]

        assert profile.provider_slug == "pmp"
        assert profile.n_samples == 10
        assert profile.peak_memory_p50 == round(5.5 * GIB)
        assert profile.peak_memory_p95 == round(9.55 * GIB)
        assert profile.peak_memory_max == 10 * GIB
        assert profile.wall_p95 == 100.0
        assert profile.cpu_seconds_p95 == 110.0
        assert profile.parallelism_p95 == pytest.approx(1.1)

    def test_excludes_shared_incomplete_and_counts_failures(self, db_with_resources):
        profile = _by_slug(Reader(db_with_resources).resources.profiles())["enso_tel"]

        # The 40 GiB shared reading and the sample missing a CPU time, and nothing else.
        assert profile.n_excluded == 2
        assert profile.n_failed == 1
        assert profile.peak_memory_max == 10 * GIB

    def test_include_shared(self, db_with_resources):
        profile = _by_slug(Reader(db_with_resources).resources.profiles(exclusive_only=False))["enso_tel"]

        assert profile.n_samples == 11
        assert profile.peak_memory_max == 40 * GIB
        assert profile.n_excluded == 1

    def test_a_process_tree_reading_survives_a_busy_worker(self, db_seeded):
        """
        Exclusivity only disqualifies a cgroup reading.

        A process tree sweep covers the execution's own processes,
        so it stays usable however many siblings shared the worker.
        Excluding those rows too would empty the profile of every diagnostic run by a pool,
        which is every diagnostic.
        """
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            diagnostic = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
            group = ExecutionGroup(key="enso", diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            db_seeded.session.flush()

            for index in range(3):
                _add_execution(
                    db_seeded.session,
                    group.id,
                    index,
                    **_sample(
                        peak_memory_bytes=2 * GIB,
                        memory_source="proc_tree",
                        resources_exclusive=False,
                    ),
                )
        db_seeded.session.commit()

        profile = _by_slug(Reader(db_seeded).resources.profiles())["enso_tel"]

        assert profile.memory_source == "proc_tree"
        assert profile.n_samples == 3
        assert profile.n_excluded == 0
        assert profile.peak_memory_max == 2 * GIB

    def test_mixed_memory_sources_take_the_dominant_one(self, db_with_resources):
        profile = _by_slug(Reader(db_with_resources).resources.profiles())["enso-characteristics"]

        assert profile.memory_source == "cgroup"
        assert profile.n_samples == 3
        # The lone rusage reading is excluded rather than averaged into the cgroup ones.
        assert profile.n_excluded == 1
        assert profile.peak_memory_max == 4 * GIB

    def test_unmeasured_executions_are_absent(self, db_with_resources):
        profiles = _by_slug(Reader(db_with_resources).resources.profiles())

        assert "sea-ice-area-basic" not in profiles

    def test_unmeasured_failure_yields_an_empty_profile(self, db_seeded):
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            diagnostic = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
            group = ExecutionGroup(key="only-failures", diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            db_seeded.session.flush()
            _add_execution(db_seeded.session, group.id, 0, successful=False)
        db_seeded.session.commit()

        profile = _by_slug(Reader(db_seeded).resources.profiles())["enso_tel"]

        assert profile.n_samples == 0
        assert profile.n_failed == 1
        assert profile.confidence == "none"
        assert profile.memory_source is None
        assert profile.headroom_ratio is None
        assert profile.recommended_memory_bytes == GIB
        assert profile.recommended_cpus == 1

    def test_headroom_and_over_limit(self, db_with_resources):
        profiles = _by_slug(Reader(db_with_resources).resources.profiles())

        slightly_over = profiles["enso_tel"]
        assert slightly_over.memory_limit_seen == 8 * GIB
        assert slightly_over.headroom_ratio == pytest.approx(8 / 9.55, rel=1e-3)
        assert slightly_over.over_limit is True

        blown = profiles["extratropical-modes-of-variability-nao"]
        assert blown.headroom_ratio == pytest.approx(0.4)
        assert blown.over_limit is True

    def test_recommendations(self, db_with_resources):
        profile = _by_slug(Reader(db_with_resources).resources.profiles())[
            "extratropical-modes-of-variability-nao"
        ]

        # 20 GiB * 1.3 = 26 GiB exactly, and 11400 / 3000 = 3.8 cores.
        assert profile.recommended_memory_bytes == 26 * GIB
        assert profile.parallelism_p95 == pytest.approx(3.8)
        assert profile.recommended_cpus == 4

    def test_safety_factor_changes_the_recommendation(self, db_with_resources):
        profiles = Reader(db_with_resources).resources.profiles(safety_factor=2.0)
        profile = _by_slug(profiles)["extratropical-modes-of-variability-nao"]

        assert profile.recommended_memory_bytes == 40 * GIB

    def test_confidence_thresholds(self, db_with_resources):
        profiles = _by_slug(Reader(db_with_resources).resources.profiles())

        assert profiles["enso_tel"].n_samples == 10
        assert profiles["enso_tel"].confidence == "good"
        assert profiles["extratropical-modes-of-variability-nao"].n_samples == 3
        assert profiles["extratropical-modes-of-variability-nao"].confidence == "thin"

    def test_measured_failure_is_aggregated(self, db_seeded):
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            diagnostic = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
            group = ExecutionGroup(key="failed-but-measured", diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            db_seeded.session.flush()
            _add_execution(
                db_seeded.session, group.id, 0, **_sample(successful=False, peak_memory_bytes=40 * GIB)
            )
        db_seeded.session.commit()

        # A run that died at 40 GiB is the row that most needs to reach a recommendation.
        profile = _by_slug(Reader(db_seeded).resources.profiles())["enso_tel"]
        assert profile.n_samples == 1
        assert profile.n_failed == 1
        assert profile.n_excluded == 0
        assert profile.peak_memory_max == 40 * GIB

    def test_filters(self, db_with_resources):
        reader = Reader(db_with_resources).resources

        assert {p.diagnostic_slug for p in reader.profiles(provider_contains=["pmp"])} == {
            "enso_tel",
            "extratropical-modes-of-variability-nao",
        }
        assert {p.diagnostic_slug for p in reader.profiles(diagnostic_contains=["nao"])} == {
            "extratropical-modes-of-variability-nao"
        }

    def test_since_excludes_older_rows(self, db_with_resources):
        future = datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None) + datetime.timedelta(days=1)

        assert Reader(db_with_resources).resources.profiles(since=future) == ()

    def test_empty_database(self, db_seeded):
        assert Reader(db_seeded).resources.profiles() == ()

    def test_ordering_is_deterministic(self, db_with_resources):
        reader = Reader(db_with_resources).resources

        first = [(p.provider_slug, p.diagnostic_slug) for p in reader.profiles()]
        assert first == sorted(first)
        assert first == [(p.provider_slug, p.diagnostic_slug) for p in reader.profiles()]


class TestProviderRollUp:
    def test_takes_the_max_over_diagnostics(self, db_with_resources):
        rolled_up = Reader(db_with_resources).resources.profiles(group_by="provider")
        profiles = {p.provider_slug: p for p in rolled_up}

        pmp = profiles["pmp"]
        assert pmp.diagnostic_slug is None
        # nao peaks at 20 GiB, enso_tel at 10 GiB, and a worker has to fit the larger one.
        assert pmp.peak_memory_p95 == 20 * GIB
        assert pmp.peak_memory_max == 20 * GIB
        # Cores take the p95 of the per-diagnostic parallelism (1.1 and 3.8), not the max.
        assert pmp.parallelism_p95 == pytest.approx(3.665)
        assert pmp.recommended_cpus == 4
        assert pmp.n_samples == 13
        assert pmp.n_failed == 1

    def test_headroom_is_recomputed_against_the_roll_up(self, db_with_resources):
        rolled_up = Reader(db_with_resources).resources.profiles(group_by="provider")
        profiles = {p.provider_slug: p for p in rolled_up}

        assert profiles["pmp"].memory_limit_seen == 8 * GIB
        assert profiles["pmp"].headroom_ratio == pytest.approx(0.4)
        assert profiles["pmp"].over_limit is True

    def test_limit_follows_the_peak_driving_diagnostic(self, db_seeded):
        # A small diagnostic in a small container must not lend its limit to a big
        # diagnostic that ran in a big one, which a minimum across limits would do.
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            for key, slug, peak, limit in [
                ("big", "enso_tel", 20 * GIB, 64 * GIB),
                ("small", "extratropical-modes-of-variability-nao", 1 * GIB, 2 * GIB),
            ]:
                diagnostic = db_seeded.session.query(Diagnostic).filter_by(slug=slug).first()
                group = ExecutionGroup(key=key, diagnostic_id=diagnostic.id, selectors={})
                db_seeded.session.add(group)
                db_seeded.session.flush()
                _add_execution(
                    db_seeded.session,
                    group.id,
                    0,
                    **_sample(peak_memory_bytes=peak, memory_limit_bytes=limit),
                )
        db_seeded.session.commit()

        rolled_up = Reader(db_seeded).resources.profiles(group_by="provider")
        pmp = {p.provider_slug: p for p in rolled_up}["pmp"]

        assert pmp.peak_memory_p95 == 20 * GIB
        assert pmp.memory_limit_seen == 64 * GIB
        assert pmp.over_limit is False

    def test_does_not_mix_sources_across_diagnostics(self, db_with_resources):
        # esmvaltool's only measured diagnostic is cgroup-sourced, so a rusage-only diagnostic
        # alongside it must not contribute its far larger peak to the roll-up.
        with db_with_resources.session.begin():
            diagnostic = (
                db_with_resources.session.query(Diagnostic).filter_by(slug="sea-ice-area-basic").first()
            )
            group = ExecutionGroup(key="rusage-only", diagnostic_id=diagnostic.id, selectors={})
            db_with_resources.session.add(group)
            db_with_resources.session.flush()
            _add_execution(
                db_with_resources.session,
                group.id,
                0,
                **_sample(peak_memory_bytes=200 * GIB, memory_source="rusage"),
            )
        db_with_resources.session.commit()

        rolled_up = Reader(db_with_resources).resources.profiles(group_by="provider")
        esmvaltool = {p.provider_slug: p for p in rolled_up}["esmvaltool"]

        assert esmvaltool.memory_source == "cgroup"
        assert esmvaltool.peak_memory_max == 4 * GIB
        assert esmvaltool.n_samples == 3

    def test_provider_with_no_samples(self, db_seeded):
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            diagnostic = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
            group = ExecutionGroup(key="all-failed", diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            db_seeded.session.flush()
            _add_execution(db_seeded.session, group.id, 0, successful=False)
        db_seeded.session.commit()

        (profile,) = Reader(db_seeded).resources.profiles(group_by="provider")

        assert profile.provider_slug == "pmp"
        assert profile.n_samples == 0
        assert profile.confidence == "none"


class TestMeasurements:
    def test_pagination(self, db_with_resources):
        reader = Reader(db_with_resources).resources

        page = reader.measurements(ResourceFilter(diagnostic_contains=["enso_tel"]), offset=1, limit=3)

        assert page.total_count == 13
        assert page.offset == 1
        assert page.limit == 3
        assert len(page) == 3
        assert [item.execution_id for item in page] == sorted(item.execution_id for item in page)

    def test_to_pandas_columns(self, db_with_resources):
        page = Reader(db_with_resources).resources.measurements(limit=1)

        assert "peak_memory_bytes" in page.to_pandas().columns

    def test_get_by_id(self, db_with_resources):
        reader = Reader(db_with_resources).resources
        first = reader.measurements(limit=1).items[0]

        fetched = reader.measurement(first.execution_id)

        assert fetched is not None
        assert fetched.execution_id == first.execution_id
        assert fetched.parallelism == pytest.approx(first.parallelism)

    def test_get_by_id_missing(self, db_with_resources):
        assert Reader(db_with_resources).resources.measurement(-1) is None
