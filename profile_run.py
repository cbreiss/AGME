"""cProfile run for AGME performance analysis.

Profiles one complete fit() call on the ASCII suffix corpus (small, representative)
and writes two output files:
  profile_results.txt   — human-readable pstats report (top 40 functions by cumtime)
  profile_results.prof  — binary pstats dump for further analysis with snakeviz, etc.

Run via:
    conda run -n AGME python profile_run.py

Or as a detached process (survives terminal/sleep):
    cmd /c start /b conda run -n AGME python profile_run.py > profile_log.txt 2>&1
"""

import cProfile
import io
import pstats
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus and model parameters
# ---------------------------------------------------------------------------

# Use the suffix corpus — small enough to complete in a few minutes,
# large enough to exercise all hot paths (Gibbs DP + MaxEnt update).
CORPUS = [
    "abz", "cdz", "efz", "ghz",
    "ijz", "klz", "mnz", "opz",
]

OUTPUT_DIR = Path(__file__).parent
PROF_FILE  = OUTPUT_DIR / "profile_results.prof"
TEXT_FILE  = OUTPUT_DIR / "profile_results.txt"
LOG_FILE   = OUTPUT_DIR / "profile_log.txt"

# ---------------------------------------------------------------------------
# Profiling target
# ---------------------------------------------------------------------------

def run():
    """Fit the model — this is the function we profile."""
    from agme import Model

    m = Model(
        morpheme_classes=["stem", "suffix"],
        alphabet=list("abcdefghijklmnopz"),
    )
    # 50 sweeps with burn_in=10 gives a representative sample of all code paths.
    # Increase n_sweeps here for a longer / more stable profile.
    m.fit(CORPUS, n_sweeps=50, burn_in=10, print_every=10, seed=0)
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Profiling AGME fit() on suffix corpus...", flush=True)

    profiler = cProfile.Profile()
    profiler.enable()
    model = run()
    profiler.disable()

    # Save binary dump for snakeviz / further analysis
    profiler.dump_stats(str(PROF_FILE))
    print(f"Binary profile saved: {PROF_FILE}", flush=True)

    # Build human-readable report
    stream = io.StringIO()
    stats  = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")

    # ---- Top 40 by cumulative time ----
    stream.write("\n" + "=" * 72 + "\n")
    stream.write("TOP 40 FUNCTIONS BY CUMULATIVE TIME\n")
    stream.write("=" * 72 + "\n")
    stats.print_stats(40)

    # ---- Top 40 by total (self) time ----
    stream2 = io.StringIO()
    stats2  = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stream2.write("\n" + "=" * 72 + "\n")
    stream2.write("TOP 40 FUNCTIONS BY SELF TIME (tottime)\n")
    stream2.write("=" * 72 + "\n")
    stats2.print_stats(40)

    # ---- Callers for the top hot spot (levenshtein_alignment) ----
    stream3 = io.StringIO()
    stats3  = pstats.Stats(profiler, stream=stream3)
    stats3.sort_stats("tottime")
    stream3.write("\n" + "=" * 72 + "\n")
    stream3.write("CALLERS OF levenshtein_alignment\n")
    stream3.write("=" * 72 + "\n")
    stats3.print_callers("levenshtein_alignment")

    report = stream.getvalue() + stream2.getvalue() + stream3.getvalue()

    TEXT_FILE.write_text(report, encoding="utf-8")
    print(f"Text report saved: {TEXT_FILE}", flush=True)

    # Also print a short summary to stdout
    summary_stream = io.StringIO()
    pstats.Stats(profiler, stream=summary_stream).sort_stats("cumulative").print_stats(15)
    print(summary_stream.getvalue())
    print("Done.", flush=True)
