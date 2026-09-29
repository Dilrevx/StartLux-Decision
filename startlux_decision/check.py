"""Check that the fast kernels will be used for a model, before starting a server on it.

    python -m startlux_decision.check /path/to/StartLux-Decision-4B

Exits with status 1 when flash-linear-attention or causal-conv1d is missing or not importable; transformers would then
fall back to a plain torch path that is more than ten times slower.
"""
import sys

from .model import fast_kernels_active


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip())
    ok = fast_kernels_active(sys.argv[1])
    print("fast kernels: " + ("active" if ok else "NOT active, pip install flash-linear-attention causal-conv1d"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
