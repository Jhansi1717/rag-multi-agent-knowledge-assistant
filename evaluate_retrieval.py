"""Compatibility entry point for the provenance-gated M1.4 evaluator."""

from evaluation.m14_provenance import main


if __name__ == "__main__":
    raise SystemExit(main())
