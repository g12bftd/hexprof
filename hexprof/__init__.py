"""hexprof: run, profile and bound VLM workloads on the Snapdragon Hexagon NPU.

Two execution paths are covered, matching how Qualcomm ships the two models:

* Gemma 4 E4B runs through GenieX (``geniex-bench`` on an adb-connected phone),
  the runtime Qualcomm AI Hub publishes for it.
* SigLIP 2 runs as a compiled QNN graph on an AI Hub hosted Galaxy S25.

On top of that, :mod:`hexprof.ceilings` measures the NPU's attainable compute
and memory-bandwidth ceilings with synthetic graphs, and
:mod:`hexprof.aihub.hwtrace` classifies operators from optrace / QHAS data.
"""

__version__ = "0.1.0"
