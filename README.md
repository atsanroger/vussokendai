# inclusive

Inclusive tau -> X_s nu: the original Chebyshev code (`AllClass.py`, `args.py`, `my_header.py`, `TauDecay.pdf`, from
`vussokendai`) merged on 2026-10-23 with the notes and the new certified-bounds work.

| path | what |
|---|---|
| `AllClass.py`, `args.py`, `my_header.py`, `TauDecay.pdf` | the original code and write-up (unchanged) |
| `hausdorff/` | Hausdorff positivity scan, certified bounds, old-vs-new comparison, `paper/hausdorff_vus.pdf` |
| `notes/` | the 2026-10-21 literature notes (`REPORT*.md`) |
| `RBCQCD` | symlink to the correlator data (`../BetterData/RBCQCD`, not in git); run the scripts from this directory |

Quick start (from this directory): `python hausdorff/hankel_scan.py`, `python hausdorff/vus_certified.py`.
The data are not in the repository.

---


# The inclusive decay calculation 

 **AllClass.py** is the main executable file. 
 At the bottom of the script, you can select the functions you want to execute by running the following command:

```bash
python3 AllClass.py --Debug --Fit --SavePic --T 12 --EO None
```

## Parameters

- **EO**: Options are `Even`, `Odd`, or `None`.
- **T**: Specifies the length of various polynomials.
- **Fit**: Selects whether to use **Chebyshev fitting**. If not used, the correlation function is added directly.
- **t0** : Timeslicing offset, default is 0.5
  
## CUDA Support

If your environment does not support CUDA( require 11 or above), please:

1. Remove **cupy** from **my_header.py**.
2. Change **cp** to **np** in **AllClass.py**.

## File import

- Change the path inside the ReadFile function in the class LatticeFermion in file **AllClass.py**.
- For RBCQCD's data, put the correlator on directry RBCQCD should work.
