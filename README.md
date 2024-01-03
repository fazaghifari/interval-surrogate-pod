# Interval POD-PCE

Code for our paper "Interval Reduced Order Surrogate Modelling Framework for Uncertainty Quantification", to be presented in AIAA SciTech 2024, 8 January 2024.

Dataset for conducting the experiments can be found [here](https://kuleuven-my.sharepoint.com/:f:/g/personal/adam_faza_kuleuven_be/EqGF39FteuVGn2n7qQCf6TgBlxfvbElY4s3BVnO7bFvmHg?e=244eHY)

In our proposed framework, we integrate POD for interval data with PCE for interval observations. Firstly, we employ interval POD to obtain an optimally reduced-order basis from the full-order snapshot. Then, we approximate this reduced-order basis using a non-intrusive interval PCE method. Allowing non-scalar data, such as intervals, is advantageous as it takes into account more information in the physical system modelling.

## Simple Instructions
There are three notebooks available to conduct the experiments:
* `bridge_problem.ipynb`
* `heatcond.ipynb`
* `heatcond_modified.ipynb`

You should be able to directly run the code after installing the package dependencies and put the [interval_cases](https://kuleuven-my.sharepoint.com/:f:/g/personal/adam_faza_kuleuven_be/EqGF39FteuVGn2n7qQCf6TgBlxfvbElY4s3BVnO7bFvmHg?e=244eHY) folder inside `data/`. 

`surrogate.py` contains functions required for performing training and prediction. It is important to note that this file is tailored to our specific use case. If you need to run your own problem, you may need to modify or create it yourself.

### Paper Authors
* Ghifari Adam Faza -- PhD Researcher, KU Leuven
* Keivan Shariatmadar -- Senior Researcher, KU Leuven
* Hans Hallez -- Associate Professor, KU Leuven
* David Moens -- Professor, KU Leuven