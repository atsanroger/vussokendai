
# Project Title

There are three Python files in this project, with **AllClass.py** being the main executable file. At the bottom of the script, you can select the functions you want to execute by running the following command:

```bash
python3 AllClass.py --Debug --Fit --SavePic --T 12 --EO None
```

## Parameters

- **EO**: Options are `Even`, `Odd`, or `None`.
- **T**: Specifies the length of various polynomials.
- **Fit**: Selects whether to use **Chebyshev fitting**. If not used, the correlation function is added directly.

## CUDA Support

If your environment does not support CUDA, please:

1. Remove **cupy** from **my_header.py**.
2. Change **cp** to **np** in **AllClass.py**.
