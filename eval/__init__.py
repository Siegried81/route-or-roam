"""Evaluation harness: labelled questions, scoring, batch runner and report.

Kept separate from the ``rr`` package so the measurement code never depends on
the systems it measures (``rr`` is imported lazily, only by the runner).
"""
