## [2026-09-18] — Initial CPU discriminator analysis

**Goal:** Trying to see the descriptors returned from a CPU run.

- **Changes Made:**
  - File / function modified
  - Optimization or fix implemented
- **Scientific / Numerical Checks:**
  - CPU vs. GPU outputs compared: [Yes / No / Match within tolerance]
  - Discrepancies noted:
    - The CPU program runs, but it does not yet reproduce the stored classifications:
      - High-level atom separation: working
      - Contact-region detection: working
      - Interface/perimeter separation: currently broken
      - Surface/bulk separation: close, but eight atoms differ  
- **Data:**
    -N/A
- **Blockers & Next Steps:**
  - Current blocker: shapely & alphashape perimeter and interface detection issues
  - Immediate next task: create a test environment to isolate and fix the blocker most likely with alpha shape/shapely library versions.


## [2026-09-28] — Fixing blocker on shapely and alphashape library versions

**Goal:** Isolate and fix the blocker most likely with alpha shape/shapely library versions.

- **Changes Made:**
  - File / function modified
  - Optimization or fix implemented
- **Notes:**
  - CPU workflow runs.
  - Shapely 2 caused the interface/perimeter failure.
  - Shapely 1.8 restores the exact 35 interface / 24 perimeter split.

- **Data:**
  - N/A
- **Blockers & Next Steps:**
  - -We still need a modern long-term fix for Shapely 2; the older environment is a compatibility   experiment, not the final solution.
  - Current blocker: shapely & alphashape perimeter and interface detection issues
  - Immediate next task: create a test environment to isolate and fix the blocker most likely with alpha shape/shapely library versions.
