"""Utilities for docstring in imbalanced-learn."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT


class Substitution:
    """Decorate a function's or a class' docstring to perform string
    substitution on it.

    This decorator should be robust even if obj.__doc__ is None
    (for example, if -OO was passed to the interpreter)
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, obj):
        if obj.__doc__:
            obj.__doc__ = obj.__doc__.format(**self.params)
        return obj


_random_state_docstring = """random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.
    """.rstrip()

_n_jobs_docstring = """n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.
    """.rstrip()

_weight_strategy_docstring = """weight_strategy : {"inherit", "nearest", "cluster"} or callable, \
        default="inherit"
        Rule used to derive the weight of the synthetic samples when
        ``sample_weight`` is provided to ``fit_resample``:

        - ``"inherit"``: each synthetic sample copies the weight of the sample
          it is generated from;
        - ``"nearest"``: the weight is linearly interpolated between the base
          sample and the selected nearest neighbour using the same step as the
          features, mimicking a weighted replication of the endpoints;
        - ``"cluster"``: each synthetic sample receives the average weight of
          the generation pool (the minority class for SMOTE);
        - callable: ``callable(base_weight, neighbor_weight, steps)`` returning
          an array of weights for the generated samples.

        The rule is only applied when weights are passed; the unweighted path
        is unchanged.

        .. versionadded:: 0.15
    """.rstrip()
