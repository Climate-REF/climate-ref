"""
Measurement of real work rather than of a mocked probe.

Each test spends a known amount of a resource and checks what was recorded against it.

Every memory probe runs in a fresh interpreter.
An interpreter does not return freed pages to the operating system,
so a probe sharing a process with an earlier one would measure that one's leftovers.
That is the subject of one test here rather than a hazard for all of them.

The tolerances are wide on purpose.
An interpreter allocates for its own reasons while the block runs,
so the assertions bound the measurement rather than pin it.
"""

import json
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from climate_ref_core.resources import measure_resources

# These tests time real CPU and memory spends so we can't run tests in parallel accurately.
pytestmark = pytest.mark.resource_intensive

MIB = 1024 * 1024

ALLOCATION_MIB = 512
"""Large enough to stand clear of interpreter noise, small enough to run anywhere."""

PROBE_PRELUDE = f"""
import json, subprocess, sys, time
from climate_ref_core.resources import measure_resources

MIB = 1024 * 1024
ALLOCATION_MIB = {ALLOCATION_MIB}


def allocate(mib):
    buf = bytearray(mib * MIB)
    for offset in range(0, len(buf), 4096):
        buf[offset] = 1
    return buf


def allocate_in_pieces(mib):
    # A single large allocation is handed straight back to the operating system when it is freed.
    # Many small ones stay in the interpreter's arenas, which is what a worker's floor is made of.
    pieces = [bytearray(1024) for _ in range(mib * 1024)]
    for piece in pieces[::64]:
        piece[0] = 1
    return pieces


def hold(mib, seconds):
    buf = allocate(mib)
    time.sleep(seconds)
    del buf


def measure(block):
    with measure_resources(interval=0.05) as recorder:
        block()
    usage = recorder.usage
    return {{
        "peak": usage.peak_memory_bytes,
        "source": usage.memory_source,
        "exclusive": usage.exclusive,
    }}
"""

HOG_SCRIPT = f"""
import time

buf = bytearray({ALLOCATION_MIB} * 1024 * 1024)
for offset in range(0, len(buf), 4096):
    buf[offset] = 1
print("ready", flush=True)
time.sleep(120)
"""
"""A neighbour that holds a large allocation until it is killed."""


def start_probe(body: str) -> subprocess.Popen:
    """
    Start a measurement in a fresh interpreter.

    The body is appended to :data:`PROBE_PRELUDE` and must print a JSON object.
    """
    script = PROBE_PRELUDE + textwrap.dedent(body)
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def read_probe(process: subprocess.Popen) -> dict:
    """Wait for a probe to finish and return the object it reported."""
    stdout, stderr = process.communicate(timeout=180)
    assert process.returncode == 0, stderr
    return json.loads(stdout.strip().splitlines()[-1])


def probe(body: str) -> dict:
    """Run a measurement in a fresh interpreter and return what it reported."""
    return read_probe(start_probe(body))


def burn_cpu(seconds: float) -> None:
    """Occupy the calling thread for `seconds` of its own CPU time."""
    start = time.thread_time()
    x = 0.0
    while time.thread_time() - start < seconds:
        for i in range(10000):
            x += i * 1.000001


def assert_covers_allocation(result: dict) -> None:
    """Assert the measured peak grew by the allocation the probe made."""
    grew = result["peak"] - result["baseline"]
    assert grew >= 0.8 * ALLOCATION_MIB * MIB, (
        f"{result['source']} recorded a growth of {grew / MIB:.0f} MiB "
        f"for an allocation of {ALLOCATION_MIB} MiB"
    )


def test_cpu_seconds_match_the_cpu_spent():
    with measure_resources(interval=0.05) as recorder:
        burn_cpu(1.0)

    usage = recorder.usage
    assert usage.cpu_seconds == pytest.approx(1.0, abs=0.4)
    # A loaded machine stretches wall time without limit,
    # so only the lower bound holds: one thread's CPU time cannot exceed the wall it ran in.
    assert usage.wall_seconds >= 0.9 * usage.cpu_seconds


def test_sleeping_costs_wall_time_and_not_cpu_time():
    with measure_resources(interval=0.05) as recorder:
        time.sleep(1.0)

    usage = recorder.usage
    assert usage.wall_seconds >= 1.0
    assert usage.cpu_seconds < 0.5


def test_cpu_seconds_cover_more_than_one_thread():
    """Two threads spending half a second each cost one CPU second, not half of one."""

    def spend() -> None:
        workers = [threading.Thread(target=burn_cpu, args=(0.5,)) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

    with measure_resources(interval=0.05) as recorder:
        spend()

    # The lower bound has to sit above 0.5, or a count of a single thread would pass.
    assert recorder.usage.cpu_seconds == pytest.approx(1.0, abs=0.4)


def test_peak_covers_an_allocation_made_in_process():
    """
    The shape of a diagnostic that runs inside the worker process, such as an ILAMB one.

    Nothing is spawned, so the peak has to come from the worker's own footprint.
    """
    result = probe(
        """
        baseline = measure(lambda: None)


        def spend():
            buf = allocate(ALLOCATION_MIB)
            # Held across several sampling intervals so the sweep sees the whole allocation.
            time.sleep(0.5)
            del buf


        held = measure(spend)
        print(json.dumps({"baseline": baseline["peak"], **held}))
        """
    )

    assert_covers_allocation(result)


def test_peak_covers_an_allocation_made_by_a_child_process():
    """
    The shape of a diagnostic that shells out, such as an ESMValTool or PMP one.

    The work happens in a process the recorder never sees start,
    so a measurement of the worker process alone would miss all of it.
    """
    result = probe(
        """
        child = (
            "buf = bytearray(%d * 1024 * 1024)\\n"
            "for offset in range(0, len(buf), 4096):\\n"
            "    buf[offset] = 1\\n"
            "import time; time.sleep(0.5)\\n" % ALLOCATION_MIB
        )

        baseline = measure(lambda: None)
        held = measure(lambda: subprocess.run([sys.executable, "-c", child], check=True))
        print(json.dumps({"baseline": baseline["peak"], **held}))
        """
    )

    assert_covers_allocation(result)


def test_a_failing_block_still_reports_what_it_spent():
    """A run that dies is the run whose cost is most worth knowing."""
    with pytest.raises(RuntimeError, match="deliberate"):
        with measure_resources(interval=0.05) as recorder:
            burn_cpu(0.5)
            raise RuntimeError("deliberate")

    usage = recorder.usage
    assert usage is not None
    assert usage.cpu_seconds >= 0.4
    assert usage.wall_seconds >= 0.4


def test_overlapping_blocks_in_one_process_are_not_exclusive():
    """Two measurements that overlap describe the same footprint, so neither owns it."""
    started = threading.Event()
    release = threading.Event()
    recorders = []

    def measure_until_released() -> None:
        with measure_resources(interval=0.05) as recorder:
            started.set()
            release.wait(timeout=30)
        recorders.append(recorder)

    other = threading.Thread(target=measure_until_released)
    other.start()
    assert started.wait(timeout=30), "the overlapping block never started"

    with measure_resources(interval=0.05) as recorder:
        time.sleep(0.1)

    release.set()
    other.join(timeout=30)

    assert recorder.usage.exclusive is False
    assert recorders[0].usage.exclusive is False


def test_workers_running_at_the_same_time_are_not_exclusive():
    """
    The shape of a process pool, or of a Celery worker with concurrency above one.

    Two executions overlap on the same host and inside the same cgroup,
    so a reading taken from that cgroup belongs to neither of them.
    Neither worker can see the other, which is why neither claims exclusivity:
    the executor that spawned them is the only party that knows they overlap,
    and it declared nothing.
    """
    body = """
        print(json.dumps(measure(lambda: hold(ALLOCATION_MIB, 1.0))))
        """
    workers = [start_probe(body) for _ in range(2)]
    results = [read_probe(worker) for worker in workers]

    assert [result["exclusive"] for result in results] == [False, False]
    # And each one reports its own process tree rather than the cgroup they share.
    assert [result["source"] for result in results] == ["proc_tree", "proc_tree"]


def test_a_neighbour_in_the_same_cgroup_does_not_inflate_a_measurement():
    """
    A neighbour allocates half a gigabyte and holds it while a cheap block is measured.

    This is what a saturated worker looks like from inside one of its executions,
    and it is the contamination that no in-process check can catch.
    The measured block allocates nothing,
    so a peak anywhere near the neighbour's allocation means the container was measured
    in place of the block.
    """
    neighbour = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", HOG_SCRIPT],
        stdout=subprocess.PIPE,
        text=True,
    )

    try:
        assert neighbour.stdout is not None
        assert neighbour.stdout.readline().strip() == "ready", "the neighbour never finished allocating"

        result = probe(
            """
            print(json.dumps(measure(lambda: time.sleep(0.3))))
            """
        )
    finally:
        neighbour.kill()
        neighbour.wait(timeout=30)

    assert result["source"] == "proc_tree"
    assert result["peak"] < 0.5 * ALLOCATION_MIB * MIB, (
        f"a block that allocated nothing was charged {result['peak'] / MIB:.0f} MiB "
        f"while a neighbour held {ALLOCATION_MIB} MiB"
    )


@pytest.mark.xfail(
    strict=True,
    reason="A worker keeps the resident pages of the execution before it, "
    "so the next execution inherits its peak",
)
def test_a_cheap_block_is_not_charged_for_an_earlier_expensive_one():
    """
    A worker runs many executions in turn, and one of them is the expensive one.

    The cheap block here allocates nothing.
    Charging it for the block before it sizes every diagnostic for the worst on the worker.

    The default executors sidestep this by running one task per child process,
    so it bites the synchronous executor and any worker configured to reuse processes.
    """
    result = probe(
        """
        def spend():
            pieces = allocate_in_pieces(ALLOCATION_MIB)
            time.sleep(0.2)
            del pieces


        baseline = measure(lambda: None)
        expensive = measure(spend)
        cheap = measure(lambda: time.sleep(0.2))
        print(json.dumps({
            "baseline": baseline["peak"],
            "expensive": expensive["peak"],
            "peak": cheap["peak"],
            "source": cheap["source"],
        }))
        """
    )

    charged = (result["peak"] - result["baseline"]) / MIB
    assert charged < 0.5 * ALLOCATION_MIB, (
        f"{result['source']} charged a block that allocated nothing with {charged:.0f} MiB"
    )
