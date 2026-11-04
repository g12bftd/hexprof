"""Readable reports from AI Hub's NPU hardware artifacts (QHAS summaries and optrace timelines).

File formats follow Qualcomm's own QAIRT Visualizer (``qairt-visualizer`` 0.12):

* QHAS summary JSON: ``header`` (``artifact_type`` "QHAS", schema ``version``
  1.x) and five tables under ``data``: ``htp_overall_summary`` (per HTP core,
  with nested ``htp_resources``), ``qnn_op_types``, ``qnn_op_instances_nodes``,
  ``htp_op_types`` and ``htp_op_instances``.
* Optrace JSON: Chrome trace events (``{"traceEvents": [...]}`` or a bare list)
  with metadata (``ph: "M"``: process/thread names), complete events
  (``ph: "X"``: name, pid, tid, ts, dur, ``args.stackDepth``) and counters
  (``ph: "C"``, e.g. "Memory Bandwidth").
"""
