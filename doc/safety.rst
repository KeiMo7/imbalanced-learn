.. _safety:

==============================
Resampling safety and auditing
==============================

.. currentmodule:: imblearn.safety

This page is the reference for the opt-in safeguards against resampling data
leakage. A narrative guide with examples is available in
:ref:`common_pitfalls`.

Guards
======

.. autosummary::
   :toctree: generated/

   ResamplingGuard
   GuardedSampler

.. autoclass:: ResamplingGuard
   :members:

.. autoclass:: GuardedSampler
   :members:

Provenance and ratio reports
============================

.. autosummary::
   :toctree: generated/

   ResamplingProvenance
   SamplingRatioReport
   check_sampling_ratio

.. autoclass:: ResamplingProvenance
   :members:

.. autoclass:: SamplingRatioReport
   :members:

.. autofunction:: check_sampling_ratio

Exceptions
==========

.. autosummary::
   :toctree: generated/

   ResamplingLeakageError
   InfeasibleSamplingTargetError

.. autoclass:: ResamplingLeakageError
   :members:

.. autoclass:: InfeasibleSamplingTargetError
   :members:
