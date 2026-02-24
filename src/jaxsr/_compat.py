"""
Scikit-learn estimator protocol mixin for JAXSR.

Implements ``get_params`` / ``set_params`` so that JAXSR estimators work with
sklearn meta-tools (``clone``, ``cross_val_score``, ``GridSearchCV``,
``Pipeline``) without requiring scikit-learn as a hard dependency.
"""

from __future__ import annotations

import inspect


class _FallbackTargetTags:
    """Minimal target tags fallback."""

    required = True
    one_d_labels = False
    two_d_labels = False
    single_output = True
    multi_output = False
    positive_only = False


class _FallbackInputTags:
    """Minimal input tags fallback."""

    one_d_array = False
    two_d_array = True
    three_d_array = False
    sparse = False
    categorical = False
    string = False
    dict = False
    positive_only = False
    allow_nan = False
    pairwise = False


class _FallbackTags:
    """Minimal tags object for when sklearn is not installed."""

    estimator_type = None
    target_tags = _FallbackTargetTags()
    input_tags = _FallbackInputTags()
    classifier_tags = None
    regressor_tags = None
    transformer_tags = None
    array_api_support = False
    no_validation = False
    non_deterministic = False


class _SklearnCompatMixin:
    """
    Mixin that implements the scikit-learn estimator protocol.

    Any class that inherits from this mixin and stores its ``__init__``
    parameters as same-named instance attributes will be compatible with
    ``sklearn.base.clone()``, ``cross_val_score``, ``GridSearchCV``, etc.
    """

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool
            If True, return parameters of nested sub-estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        sig = inspect.signature(self.__init__)
        params = {}
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            value = getattr(self, name, p.default)
            params[name] = value

        if deep:
            nested = {}
            for name, value in params.items():
                if hasattr(value, "get_params"):
                    for k, v in value.get_params(deep=True).items():
                        nested[f"{name}__{k}"] = v
            params.update(nested)

        return params

    def set_params(self, **params) -> _SklearnCompatMixin:
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters. Supports nested parameters using
            double-underscore syntax (e.g. ``estimator__max_terms=8``).

        Returns
        -------
        self : estimator instance

        Raises
        ------
        ValueError
            If any parameter name is not valid for this estimator.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        # Split into nested and direct params
        nested: dict[str, dict[str, object]] = {}
        for key, value in params.items():
            if "__" in key:
                prefix, sub_key = key.split("__", 1)
                if prefix not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {prefix!r} for {type(self).__name__}. "
                        f"Valid parameters: {sorted(self.get_params(deep=False))}."
                    )
                nested.setdefault(prefix, {})[sub_key] = value
            else:
                if key not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {key!r} for {type(self).__name__}. "
                        f"Valid parameters: {sorted(self.get_params(deep=False))}."
                    )
                setattr(self, key, value)

        for prefix, sub_params in nested.items():
            sub_estimator = getattr(self, prefix)
            sub_estimator.set_params(**sub_params)

        return self

    def __repr__(self) -> str:
        """Sklearn-style repr: ``ClassName(param=val, ...)``."""
        params = self.get_params(deep=False)
        sig = inspect.signature(self.__init__)
        parts = []
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            value = params.get(name, p.default)
            if value is not p.default:
                parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __sklearn_tags__(self):
        """
        Return sklearn tags for estimator compatibility.

        Required by scikit-learn >= 1.6 for ``cross_val_score``,
        ``GridSearchCV``, ``Pipeline``, etc.
        """
        try:
            from sklearn.utils._tags import Tags, TargetTags

            return Tags(
                estimator_type=None,
                target_tags=TargetTags(required=True),
            )
        except ImportError:
            return _FallbackTags()

    def __sklearn_is_fitted__(self) -> bool:
        """Check if the estimator is fitted (sklearn protocol)."""
        return getattr(self, "_is_fitted", False)
