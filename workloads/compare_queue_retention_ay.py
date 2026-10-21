"""Original A-Y requests with queue-aware eviction as the only new policy."""

import compare_prefix_policies as base

base.EVICTION_RESULT_ROOT = "results/queue-retention/ay"

if __name__ == "__main__":
    base.main()
