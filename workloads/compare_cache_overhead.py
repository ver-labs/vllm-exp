"""Allow the asynchronous CPU-profile writer to finish after timing ends."""

import time

from compare_aging_lfu import main

if __name__ == "__main__":
    main()
    time.sleep(0.5)
