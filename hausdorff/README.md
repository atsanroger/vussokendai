# hausdorff/

Positivity test for the [-1, 1] support hypothesis (Möbius/DWF with M5 > 1) on the same
`RBCQCD/corr_Z2_*` files that `AllClass.py` reads. Needs only numpy.

    cd <dir that contains RBCQCD/>
    python hausdorff/hankel_scan.py                # full scan (2 s)
    python hausdorff/hankel_test_proposal.py --ens 48 --chan A1

- `RESULTS.md`        what was found on 2026-10-23, and what is still missing
- `OLD_CODE_NOTES.md` points to check in `AllClass.py` before trusting a |Vus| from it
- `RESULTS_SAFE_VUS.md` certified bounds on the inclusive rate (safe_vus.py, vus_interval.py) and old-vs-new comparison

The data are not in this repository.
