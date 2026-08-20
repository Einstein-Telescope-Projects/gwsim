"""Give every mutmut mutant worker a process that inherited no locks.

Why this file exists
--------------------
``mutmut`` executes the test suite *in the driver process* before it tests a single
mutant: the stats phase, the "clean tests" run and the forced-fail run all call
``pytest.main()`` in the current interpreter. Only afterwards does the per-mutant
loop call ``os.fork()``, once per mutant, from that same interpreter.

That ordering is fatal for a suite that drives a native runtime with its own thread
pool -- which this one does, through the array backends the signal path uses. Running
the suite starts those pools in the driver, and ``fork()`` gives the child a single
thread plus every mutex the other threads happened to be holding, now held by nobody.
The child blocks forever on the first one it needs. JAX says so itself, once per run,
straight from mutmut's fork site::

    RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded
    code, and JAX is multithreaded, so this will likely lead to a deadlock.

Measured on this project: the driver carries ~106 OS threads by the time the fork loop
starts, and a full-package run left 2,573 of the 14,968 mutants that have a test (17%)
as process-level timeouts -- both worker threads parked in ``futex_wait_queue`` with
their CPU time frozen at ~1.5 s and unchanged 75 s later. Deadlocked, not slow.
Killing the pools does not help: the same mutants still hang with the Eigen, OpenMP and
BLAS thread counts all pinned to one, so the lock belongs to some other native runtime
and only the general fix is reliable.

A mutant with no verdict is worse than a slow one, because ``mutmut`` reports the
process-level timeout under a status that reads like a kill. Until the workers finish,
the mutation score's denominator is unusable in either direction.

What this does
--------------
The forked worker immediately ``execve()``s a fresh interpreter that runs the same
pytest invocation ``mutmut`` would have run in-process. ``exec`` replaces the process
image, so every inherited lock, thread and half-initialised native runtime is gone,
while the things the driver relies on survive untouched:

* the **pid** is unchanged, so the driver's ``os.wait()`` and its wall-clock
  ``SIGXCPU`` timeout still address the right process;
* the **CPU rlimit** ``mutmut`` set just before this call is preserved across ``exec``;
* the **exit status** is pytest's own, exactly as when ``mutmut`` called
  ``pytest.main()`` itself, so every verdict maps the same way.

The driver cannot tell the difference.

The cost is real and worth knowing: a worker no longer inherits the driver's warm
``sys.modules``, so it re-imports pytest and the code under test. On a module whose
tests are cheap, throughput went from ~33 to ~1.8 mutants/second with 16 workers -- a
fresh worker costs roughly 3 s in isolation, most of it imports. Set against that, a
deadlocked mutant used to burn its whole timeout budget (15x its estimated test time)
and still return no verdict, so a full-package run spent hours reaching 17% fewer
answers than it does now.

How it is wired up
------------------
``tests/conftest.py`` calls :func:`install_if_mutmut_driver` at import time. That
conftest is imported by *every* pytest run, but the patch is applied only when the
importing process is a ``mutmut`` driver in one of its own phases -- never in a plain
``pytest`` run and never inside a worker. ``mutmut`` copies ``tests/`` into
``mutants/`` itself, so no extra configuration is needed to carry this along.

If a future ``mutmut`` renames the internals this leans on, installation raises
rather than returning quietly: a silent no-op here brings the deadlocks back, and
they masquerade as results.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

__all__ = [
    "build_worker_pytest_args",
    "exec_mutant_worker",
    "install_fork_safe_mutant_workers",
    "install_if_mutmut_driver",
    "run_mutant_in_fresh_interpreter",
]

#: Environment variables carrying the worker's instructions across ``exec``.
ENV_SYS_PATH = "GWMOCK_MUTMUT_WORKER_SYS_PATH"
ENV_PYTEST_ARGS = "GWMOCK_MUTMUT_WORKER_PYTEST_ARGS"
ENV_ROOT = "GWMOCK_MUTMUT_WORKER_ROOT"
ENV_CWD = "GWMOCK_MUTMUT_WORKER_CWD"
ENV_DEBUG = "GWMOCK_MUTMUT_WORKER_DEBUG"

#: Set to a false-ish value to send mutant runs back through mutmut's own in-process
#: fork. Only useful for measuring the difference the fresh interpreter makes -- a
#: mutation run with this off inherits the deadlocks described above.
ENV_ENABLED = "GWMOCK_MUTMUT_FORK_SAFE_WORKERS"

#: Values of :data:`ENV_ENABLED` that turn the hook off.
DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

#: Marker attribute so a second import of the conftest does not stack patches.
PATCHED_MARKER = "_gwmock_fork_safe_workers"

#: Exit code used when the worker fails before pytest starts. Deliberately outside
#: mutmut's exit-code table, which maps anything unknown to "suspicious": a bootstrap
#: failure must never be able to read as a killed mutant.
WORKER_BOOTSTRAP_FAILURE_EXIT_CODE = 70

#: The values ``mutmut`` gives ``MUTANT_UNDER_TEST`` while it is driving. Anything
#: else is the name of a mutant, i.e. we are the worker and must not patch.
DRIVER_PHASES = frozenset({"", "stats", "fail", "list_all_tests", "mutant_generation"})

#: The name of the directory mutmut mutates into, relative to the project root.
MUTANTS_DIR = "mutants"

# The program the fresh interpreter runs. It restores the driver's ``sys.path``
# (which mutmut rewrote so that `mutants/src` shadows the real `src`), loads mutmut's
# config while the cwd is still the project root -- the config resolves paths against
# the cwd, and mutmut's own driver loads it from there -- and only then enters
# `mutants/` to run pytest, which is where mutmut runs its tests from.
#
# stdout and stderr are sent to /dev/null only once the setup has succeeded, so a
# broken bootstrap is loud on the terminal while ordinary pytest chatter (which the
# driver discarded anyway, once per mutant) stays out of the way.
WORKER_BOOTSTRAP = f"""\
import json
import os
import sys
import traceback

try:
    sys.path[:] = json.loads(os.environ[{ENV_SYS_PATH!r}])
    os.chdir(os.environ[{ENV_ROOT!r}])
    from mutmut.configuration import Config

    Config.ensure_loaded()
    os.chdir(os.environ[{ENV_CWD!r}])
    import pytest

    pytest_args = json.loads(os.environ[{ENV_PYTEST_ARGS!r}])
except BaseException:
    traceback.print_exc()
    raise SystemExit({WORKER_BOOTSTRAP_FAILURE_EXIT_CODE!r})

if not os.environ.get({ENV_DEBUG!r}):
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)

raise SystemExit(pytest.main(pytest_args))
"""


def _mutmut_debug() -> bool:
    """Whether mutmut is in debug mode.

    mutmut is imported lazily, and only from the code paths a mutation run reaches, so
    an ordinary ``pytest`` run -- which imports this module through the conftest -- never
    needs mutmut to be installed at all.
    """
    from mutmut.configuration import Config

    return bool(Config.get().debug)


def build_worker_pytest_args(runner: Any, tests: list[str]) -> list[str]:
    """Reproduce the argument list ``mutmut`` would have passed to ``pytest.main``.

    The pieces come from the runner itself rather than being restated here, so a
    change to mutmut's own argument handling is picked up instead of drifting.
    """
    args = [
        "--rootdir=.",
        "--tb=native",
        *runner._pytest_args_regular_run(tests),
        *runner._pytest_add_cli_args,
    ]
    if _mutmut_debug():
        args = ["-vv", *args]
    return args


def exec_mutant_worker(runner: Any, *, mutant_name: str, tests: list[str]) -> None:
    """Replace this (forked) process with a fresh interpreter running the tests.

    Never returns: on success the process image is gone, and on failure ``execve``
    raises ``OSError``.
    """
    root = os.getcwd()
    env = dict(os.environ)
    # Relative entries would mean something different after the chdir into `mutants`,
    # and an empty entry means "the current directory", which is the same trap.
    env[ENV_SYS_PATH] = json.dumps([os.path.abspath(entry) for entry in sys.path if entry])
    env[ENV_PYTEST_ARGS] = json.dumps(build_worker_pytest_args(runner, tests))
    env[ENV_ROOT] = root
    env[ENV_CWD] = os.path.join(root, MUTANTS_DIR)
    if _mutmut_debug():
        env[ENV_DEBUG] = "1"

    # The mutant name is passed as an argument purely so `ps` still identifies the
    # worker; the bootstrap reads its instructions from the environment.
    os.execve(  # noqa: S606 -- a fixed interpreter and a fixed program, no shell involved
        sys.executable,
        [sys.executable, "-c", WORKER_BOOTSTRAP, mutant_name],
        env,
    )


def run_mutant_in_fresh_interpreter(runner: Any, *, mutant_name: str, tests: list[str]) -> None:
    """Hand this worker over to a fresh interpreter, or die loudly trying.

    Anything that goes wrong before ``exec`` exits with a code mutmut does not know,
    so it lands in "suspicious" and is visible. Letting the exception propagate
    instead would leave the worker exiting 1, which mutmut reads as a killed mutant --
    a harness failure silently recorded as a result.
    """
    try:
        exec_mutant_worker(runner, mutant_name=mutant_name, tests=tests)
    except BaseException:
        # sys.stderr is redirected into a buffer mutmut discards for workers; the
        # original stream is the one a human can actually see.
        traceback.print_exc(file=sys.__stderr__ or sys.stderr)
    os._exit(WORKER_BOOTSTRAP_FAILURE_EXIT_CODE)


def install_fork_safe_mutant_workers(runner_cls: type) -> bool:
    """Patch ``runner_cls.run_tests`` so mutant runs happen in a fresh interpreter.

    Returns ``True`` if the patch was applied, ``False`` if it was already in place.
    Raises ``RuntimeError`` if the runner does not look like the one this was written
    against, because failing quietly here reintroduces the deadlocks.
    """
    if getattr(runner_cls, PATCHED_MARKER, False):
        return False

    missing = [name for name in ("run_tests", "_pytest_args_regular_run") if not hasattr(runner_cls, name)]
    if missing:
        raise RuntimeError(
            f"Cannot make mutmut's mutant workers fork-safe: {runner_cls.__name__} has no "
            f"{', '.join(missing)}. mutmut's runner has changed shape; update "
            "tests/mutmut_fork_safety.py before trusting a mutation run."
        )

    original_run_tests = runner_cls.run_tests

    def run_tests(self: Any, *, mutant_name: str | None, tests: list[str]) -> int:
        # `mutant_name is None` is mutmut's clean-test and forced-fail runs, which
        # happen in the driver and are supposed to stay in-process.
        if mutant_name is None:
            return int(original_run_tests(self, mutant_name=mutant_name, tests=tests))
        run_mutant_in_fresh_interpreter(self, mutant_name=mutant_name, tests=tests)
        raise AssertionError("unreachable: the worker either exec'd or exited")  # pragma: no cover

    runner_cls.run_tests = run_tests
    # Kept so the patch can be inspected, and undone, from a debugging session.
    runner_cls._gwmock_original_run_tests = staticmethod(original_run_tests)
    setattr(runner_cls, PATCHED_MARKER, True)
    return True


def install_if_mutmut_driver() -> bool:
    """Install the patch when this process is a ``mutmut`` driver, else do nothing.

    Returns ``True`` only when a patch was applied by this call.
    """
    if os.environ.get(ENV_ENABLED, "").strip().lower() in DISABLED_VALUES:
        return False

    mutmut_main = sys.modules.get("mutmut.__main__")
    if mutmut_main is None:
        # A plain pytest run, or a mutant worker: nothing to patch either way.
        return False
    if os.environ.get("MUTANT_UNDER_TEST", "") not in DRIVER_PHASES:
        # A mutant name means we are inside a worker that has already been exec'd.
        return False

    runner_cls = getattr(mutmut_main, "PytestRunner", None)
    if runner_cls is None:
        raise RuntimeError(
            "Cannot make mutmut's mutant workers fork-safe: mutmut.__main__ has no "
            "PytestRunner. Update tests/mutmut_fork_safety.py before trusting a mutation run."
        )
    return install_fork_safe_mutant_workers(runner_cls)
