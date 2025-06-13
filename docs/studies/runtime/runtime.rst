Runtime
========

Demo of the runtime as a function of the number of knots.
All runs use the same data and same number of iterations -- the only difference is the number of knots used in the model.

.. figure:: plots/mcmc_runtimes.png
   :width: 100%
   :align: center

   Runtime as a function of the number of knots.


Source Code
-----------

.. literalinclude:: record_runtime.py
   :language: python
   :caption: `record_runtime.py` - script for runtime collection
