Installation
============

First, we need to set up the environment. The easiest way to install the dependencies is through
conda_ or mamba_. We could use the `conda/environment.yml` file in the GQE main repo.

.. code-block:: bash

   conda env create -f conda/environment.yml
   conda activate gqe

Alternatively, we can use the container.

.. _conda: https://docs.conda.io/projects/miniconda/en/latest
.. _mamba: https://mamba.readthedocs.io/en/latest/index.html

Then, to build and install GQE-Python, run

.. code-block:: bash

    pip install <gqe-python-project-root>

Note that since the GQE repo is internal, we have to set up a valid SSH key on Gitlab to authenticate.
