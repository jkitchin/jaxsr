"""
DOE Study Persistence for JAXSR.

Provides the DOEStudy class for managing and serializing complete Design of Experiments
workflows. A study bundles the experimental design, observed data, fitted models,
and iteration history into a single shareable file (.jaxsr ZIP archive).

Typical async workflow::

    # Day 1: create design
    study = DOEStudy(
        name="catalyst_optimization",
        factor_names=["temperature", "pressure", "catalyst"],
        bounds=[(300, 500), (1, 10), (0, 2)],
        feature_types=["continuous", "continuous", "categorical"],
        categories={2: ["A", "B", "C"]},
    )
    study.create_design(method="latin_hypercube", n_points=20)
    study.save("catalyst.jaxsr")

    # Day 2: add lab results and fit
    study = DOEStudy.load("catalyst.jaxsr")
    study.add_observations(X_measured, y_measured)
    study.fit(max_terms=5)
    study.save("catalyst.jaxsr")

    # Day 3: suggest next experiments
    study = DOEStudy.load("catalyst.jaxsr")
    next_points = study.suggest_next(n_points=5)
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .basis import BasisLibrary
from .regressor import SymbolicRegressor
from .selection import SelectionResult

# Schema version for forward/backward compatibility
_SCHEMA_VERSION = "1.0.0"


def _json_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class Iteration:
    """
    Record of a single iteration in the DOE workflow.

    Parameters
    ----------
    round_number : int
        Sequential round number (1-indexed).
    timestamp : str
        ISO 8601 timestamp of when this iteration was recorded.
    n_points_added : int
        Number of new observations added in this round.
    model_expression : str or None
        Fitted model expression after this round, if a fit was performed.
    model_metrics : dict or None
        Model metrics (mse, aic, bic, r2, etc.) after this round.
    notes : str
        Optional user notes for this iteration.
    """

    round_number: int
    timestamp: str
    n_points_added: int
    model_expression: str | None = None
    model_metrics: dict | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "round_number": self.round_number,
            "timestamp": self.timestamp,
            "n_points_added": self.n_points_added,
            "model_expression": self.model_expression,
            "model_metrics": self.model_metrics,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Iteration:
        """
        Deserialize from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with iteration fields.

        Returns
        -------
        iteration : Iteration
            Deserialized iteration record.
        """
        return cls(
            round_number=data["round_number"],
            timestamp=data["timestamp"],
            n_points_added=data["n_points_added"],
            model_expression=data.get("model_expression"),
            model_metrics=data.get("model_metrics"),
            notes=data.get("notes", ""),
        )


@dataclass
class DOEStudy:
    """
    Persistent container for a Design of Experiments workflow.

    Bundles experimental design configuration, observed data, fitted models,
    and iteration history into a single object that can be saved to and loaded
    from a ``.jaxsr`` ZIP archive.

    Parameters
    ----------
    name : str
        Human-readable study name.
    factor_names : list of str
        Names for each factor/feature.
    bounds : list of tuple of (float, float)
        Lower and upper bounds for each factor.
    feature_types : list of str or None
        ``"continuous"`` or ``"categorical"`` for each factor. Defaults to all
        continuous if not provided.
    categories : dict mapping int to list, or None
        For categorical factors, maps factor index to the list of valid levels.
    description : str
        Optional longer description of the study purpose.

    Examples
    --------
    >>> study = DOEStudy(
    ...     name="yield_optimization",
    ...     factor_names=["temperature", "pressure"],
    ...     bounds=[(300, 500), (1, 10)],
    ... )
    >>> study.create_design(method="latin_hypercube", n_points=20)
    >>> study.save("yield_study.jaxsr")
    """

    name: str
    factor_names: list[str]
    bounds: list[tuple[float, float]]
    feature_types: list[str] | None = None
    categories: dict[int, list] | None = None
    description: str = ""

    # Design state
    _design_config: dict = field(default_factory=dict, repr=False)
    _X_design: np.ndarray | None = field(default=None, repr=False)

    # Observation state
    _X_observed: np.ndarray | None = field(default=None, repr=False)
    _y_observed: np.ndarray | None = field(default=None, repr=False)
    _observation_status: list[str] = field(default_factory=list, repr=False)

    # Model state
    _model: SymbolicRegressor | None = field(default=None, repr=False)
    _library_config: dict | None = field(default=None, repr=False)
    _model_config: dict | None = field(default=None, repr=False)
    _result_dict: dict | None = field(default=None, repr=False)

    # History
    _iterations: list[Iteration] = field(default_factory=list, repr=False)

    # Metadata
    _meta: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        n = len(self.factor_names)
        if len(self.bounds) != n:
            raise ValueError(
                f"Length of bounds ({len(self.bounds)}) must match " f"factor_names ({n})"
            )
        if self.feature_types is not None and len(self.feature_types) != n:
            raise ValueError(
                f"Length of feature_types ({len(self.feature_types)}) must match "
                f"factor_names ({n})"
            )
        if self.feature_types is not None:
            for i, ft in enumerate(self.feature_types):
                if ft not in ("continuous", "categorical"):
                    raise ValueError(
                        f"feature_types[{i}] must be 'continuous' or 'categorical', " f"got {ft!r}"
                    )
        if self.categories is not None:
            for idx, levels in self.categories.items():
                if not isinstance(levels, list) or len(levels) == 0:
                    raise ValueError(f"categories[{idx}] must be a non-empty list, got {levels!r}")

        now = datetime.now(timezone.utc).isoformat()
        if not self._meta:
            self._meta = {
                "schema_version": _SCHEMA_VERSION,
                "jaxsr_version": _get_jaxsr_version(),
                "created": now,
                "modified": now,
            }

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def n_factors(self) -> int:
        """Number of factors in the study."""
        return len(self.factor_names)

    @property
    def design_points(self) -> np.ndarray | None:
        """Design matrix, or None if no design has been created."""
        return self._X_design

    @property
    def X(self) -> np.ndarray | None:
        """Observed feature matrix, or None if no observations added."""
        return self._X_observed

    @property
    def y(self) -> np.ndarray | None:
        """Observed response vector, or None if no observations added."""
        return self._y_observed

    @property
    def n_observations(self) -> int:
        """Number of observations collected so far."""
        if self._y_observed is None:
            return 0
        return len(self._y_observed)

    @property
    def is_fitted(self) -> bool:
        """Whether a model has been fitted."""
        return self._model is not None and self._model._is_fitted

    @property
    def model(self) -> SymbolicRegressor | None:
        """The fitted SymbolicRegressor, or None if not yet fitted."""
        return self._model

    @property
    def iterations(self) -> list[Iteration]:
        """List of iteration records."""
        return list(self._iterations)

    @property
    def meta(self) -> dict:
        """Study metadata (schema_version, jaxsr_version, created, modified)."""
        return dict(self._meta)

    @property
    def pending_points(self) -> np.ndarray | None:
        """
        Design points that have not yet been observed.

        Returns None if no design has been created. Returns an empty array
        if all design points have been observed.
        """
        if self._X_design is None:
            return None
        if not self._observation_status:
            return self._X_design.copy()
        pending_mask = np.array([s == "pending" for s in self._observation_status], dtype=bool)
        return self._X_design[pending_mask]

    @property
    def completed_points(self) -> np.ndarray | None:
        """
        Design points that have been observed.

        Returns None if no design has been created.
        """
        if self._X_design is None:
            return None
        if not self._observation_status:
            return np.empty((0, self.n_factors))
        completed_mask = np.array([s == "completed" for s in self._observation_status], dtype=bool)
        return self._X_design[completed_mask]

    # -------------------------------------------------------------------------
    # Design creation
    # -------------------------------------------------------------------------

    def create_design(
        self,
        method: str = "latin_hypercube",
        n_points: int = 20,
        random_state: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate an experimental design.

        Parameters
        ----------
        method : str
            Sampling method. One of ``"latin_hypercube"``, ``"sobol"``,
            ``"halton"``, ``"grid"``, ``"factorial"``, ``"ccd"``,
            ``"box_behnken"``.
        n_points : int
            Number of design points (ignored for factorial/ccd/box_behnken
            which determine their own size).
        random_state : int or None
            Random seed for reproducibility.
        **kwargs
            Additional keyword arguments passed to the sampling function
            or RSM design generator.

        Returns
        -------
        X_design : numpy.ndarray
            Design matrix of shape ``(n_points, n_factors)``.

        Raises
        ------
        ValueError
            If ``method`` is not recognized.
        """
        from .sampling import (
            grid_sample,
            halton_sample,
            latin_hypercube_sample,
            sobol_sample,
        )

        discrete_dims = None
        if self.categories is not None:
            # Map string category levels to numeric indices for sampling.
            # The sampler works in index space; we store the mapping for
            # converting back to labels in display/export.
            discrete_dims = {
                idx: list(range(len(levels))) for idx, levels in self.categories.items()
            }

        rsm_methods = {"factorial", "ccd", "box_behnken", "fractional_factorial"}
        sampling_methods = {"latin_hypercube", "sobol", "halton", "grid"}

        if method in sampling_methods:
            sampler_map = {
                "latin_hypercube": latin_hypercube_sample,
                "sobol": sobol_sample,
                "halton": halton_sample,
                "grid": grid_sample,
            }
            sampler_fn = sampler_map[method]
            if method == "grid":
                X = np.asarray(
                    sampler_fn(
                        n_per_dim=kwargs.pop("n_per_dim", n_points),
                        bounds=self.bounds,
                        discrete_dims=discrete_dims,
                        **kwargs,
                    )
                )
            else:
                X = np.asarray(
                    sampler_fn(
                        n_samples=n_points,
                        bounds=self.bounds,
                        random_state=random_state,
                        discrete_dims=discrete_dims,
                        **kwargs,
                    )
                )
        elif method in rsm_methods:
            from .rsm import ResponseSurface

            rs = ResponseSurface(
                n_factors=self.n_factors,
                bounds=self.bounds,
                factor_names=self.factor_names,
                feature_types=self.feature_types,
                categories=self.categories,
            )
            if method == "factorial":
                X = rs.factorial(levels=kwargs.pop("levels", 2), **kwargs)
            elif method == "ccd":
                X = rs.ccd(
                    alpha=kwargs.pop("alpha", "rotatable"),
                    center_points=kwargs.pop("center_points", 1),
                    **kwargs,
                )
            elif method == "box_behnken":
                X = rs.box_behnken(center_points=kwargs.pop("center_points", 1), **kwargs)
            elif method == "fractional_factorial":
                X = rs.fractional_factorial(resolution=kwargs.pop("resolution", 3), **kwargs)
        else:
            raise ValueError(
                f"Unknown design method {method!r}. Choose from: "
                f"{sorted(sampling_methods | rsm_methods)}"
            )

        self._X_design = np.asarray(X)
        self._observation_status = ["pending"] * len(self._X_design)
        self._design_config = {
            "method": method,
            "n_points": n_points,
            "random_state": random_state,
            **kwargs,
        }
        self._touch_modified()
        return self._X_design.copy()

    # -------------------------------------------------------------------------
    # Observation management
    # -------------------------------------------------------------------------

    def add_observations(
        self,
        X: np.ndarray,
        y: np.ndarray,
        record_iteration: bool = True,
        notes: str = "",
    ) -> None:
        """
        Add experimental observations.

        New data points are appended to the study's observation arrays.
        If a design exists, matching design points are marked as completed.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape ``(n_new, n_factors)``.
        y : numpy.ndarray
            Response vector of shape ``(n_new,)``.
        record_iteration : bool
            Whether to record this as a new iteration in the history.
        notes : str
            Optional notes for this batch of observations.

        Raises
        ------
        ValueError
            If shapes are inconsistent.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)

        if X.shape[1] != self.n_factors:
            raise ValueError(f"X has {X.shape[1]} columns but study has {self.n_factors} factors")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} elements")

        # Append to observation arrays
        if self._X_observed is None:
            self._X_observed = X
            self._y_observed = y
        else:
            self._X_observed = np.vstack([self._X_observed, X])
            self._y_observed = np.concatenate([self._y_observed, y])

        # Mark matching design points as completed
        if self._X_design is not None:
            for x_row in X:
                for idx, (status, design_row) in enumerate(
                    zip(self._observation_status, self._X_design, strict=False)
                ):
                    if status == "pending" and np.allclose(x_row, design_row, atol=1e-10):
                        self._observation_status[idx] = "completed"
                        break

        # Record iteration
        if record_iteration:
            metrics = None
            expr = None
            if self.is_fitted:
                expr = self._model.expression_
                metrics = {
                    "mse": float(self._model._result.mse),
                    "aic": float(self._model._result.aic),
                    "bic": float(self._model._result.bic),
                }
            self._iterations.append(
                Iteration(
                    round_number=len(self._iterations) + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    n_points_added=len(y),
                    model_expression=expr,
                    model_metrics=metrics,
                    notes=notes,
                )
            )

        self._touch_modified()

    # -------------------------------------------------------------------------
    # Model fitting
    # -------------------------------------------------------------------------

    def fit(
        self,
        max_terms: int = 10,
        strategy: str = "greedy_forward",
        information_criterion: str = "aicc",
        basis_config: dict | None = None,
        **kwargs,
    ) -> SymbolicRegressor:
        """
        Fit a symbolic regression model to the current observations.

        Parameters
        ----------
        max_terms : int
            Maximum number of terms in the model.
        strategy : str
            Selection strategy (``"greedy_forward"``, ``"greedy_backward"``,
            ``"exhaustive"``, ``"lasso_path"``).
        information_criterion : str
            Criterion for model selection (``"aic"``, ``"aicc"``, ``"bic"``).
        basis_config : dict or None
            If provided, passed as keyword arguments to ``BasisLibrary.build_default()``.
            For example, ``{"max_degree": 3, "include_transcendental": True}``.
        **kwargs
            Additional keyword arguments passed to ``SymbolicRegressor``.

        Returns
        -------
        model : SymbolicRegressor
            The fitted model.

        Raises
        ------
        RuntimeError
            If no observations have been added yet.
        """
        if self._X_observed is None or self._y_observed is None:
            raise RuntimeError("No observations available. Call add_observations() first.")

        # Build basis library
        basis_kwargs = basis_config or {}
        library = BasisLibrary(
            n_features=self.n_factors,
            feature_names=self.factor_names,
            feature_bounds=self.bounds,
            feature_types=self.feature_types,
            categories=self.categories,
        ).build_default(**basis_kwargs)

        model = SymbolicRegressor(
            basis_library=library,
            max_terms=max_terms,
            strategy=strategy,
            information_criterion=information_criterion,
            **kwargs,
        )
        model.fit(self._X_observed, self._y_observed)

        self._model = model
        self._library_config = library.to_dict()
        self._model_config = {
            "max_terms": max_terms,
            "strategy": strategy,
            "information_criterion": information_criterion,
            **kwargs,
        }
        self._result_dict = model._result.to_dict()

        # Update the latest iteration with model info if one exists
        if self._iterations:
            last = self._iterations[-1]
            last.model_expression = model.expression_
            last.model_metrics = {
                "mse": float(model._result.mse),
                "aic": float(model._result.aic),
                "bic": float(model._result.bic),
                "aicc": float(model._result.aicc),
                "n_terms": model._result.n_terms,
            }

        self._touch_modified()
        return model

    def suggest_next(
        self,
        n_points: int = 5,
        strategy: str = "uncertainty",
        **kwargs,
    ) -> np.ndarray:
        """
        Suggest next experimental points using adaptive sampling.

        Parameters
        ----------
        n_points : int
            Number of points to suggest.
        strategy : str
            Sampling strategy (``"uncertainty"``, ``"error"``, ``"leverage"``,
            ``"gradient"``, ``"space_filling"``, ``"random"``).
        **kwargs
            Additional keyword arguments passed to ``AdaptiveSampler``.

        Returns
        -------
        next_points : numpy.ndarray
            Suggested points of shape ``(n_points, n_factors)``.

        Raises
        ------
        RuntimeError
            If no model has been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("No fitted model. Call fit() first.")

        from .sampling import AdaptiveSampler

        discrete_dims = None
        if self.categories:
            discrete_dims = {
                idx: list(range(len(levels))) for idx, levels in self.categories.items()
            }

        sampler = AdaptiveSampler(
            model=self._model,
            bounds=self.bounds,
            strategy=strategy,
            batch_size=n_points,
            discrete_dims=discrete_dims,
            **kwargs,
        )

        exclude = self._X_observed
        result = sampler.suggest(n_points=n_points, exclude_points=exclude)
        return np.asarray(result.points)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Save the study to a ``.jaxsr`` ZIP archive.

        The archive contains JSON metadata and NumPy binary arrays:

        - ``meta.json`` — schema version, timestamps, jaxsr version
        - ``study.json`` — factor config, design config, model config, iterations
        - ``X_design.npy`` — design matrix (if created)
        - ``X_observed.npy`` — observed features (if any)
        - ``y_observed.npy`` — observed responses (if any)

        Parameters
        ----------
        filepath : str
            Path to write the archive. The ``.jaxsr`` extension is recommended.

        Raises
        ------
        ValueError
            If the study has no meaningful content to save.
        """
        self._touch_modified()

        # Build study JSON (everything except large arrays)
        study_data = {
            "name": self.name,
            "description": self.description,
            "factor_names": self.factor_names,
            "bounds": [list(b) for b in self.bounds],
            "feature_types": self.feature_types,
            "categories": (
                {str(k): v for k, v in self.categories.items()} if self.categories else None
            ),
            "design_config": self._design_config,
            "observation_status": self._observation_status,
            "library_config": self._library_config,
            "model_config": self._model_config,
            "result": self._result_dict,
            "iterations": [it.to_dict() for it in self._iterations],
        }

        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            # Meta
            zf.writestr(
                "meta.json",
                json.dumps(self._meta, indent=2, default=_json_serializable),
            )

            # Study config + history
            zf.writestr(
                "study.json",
                json.dumps(study_data, indent=2, default=_json_serializable),
            )

            # Arrays as .npy
            if self._X_design is not None:
                zf.writestr("X_design.npy", _array_to_bytes(self._X_design))
            if self._X_observed is not None:
                zf.writestr("X_observed.npy", _array_to_bytes(self._X_observed))
            if self._y_observed is not None:
                zf.writestr("y_observed.npy", _array_to_bytes(self._y_observed))

    @classmethod
    def load(cls, filepath: str) -> DOEStudy:
        """
        Load a study from a ``.jaxsr`` ZIP archive.

        Parameters
        ----------
        filepath : str
            Path to the archive.

        Returns
        -------
        study : DOEStudy
            The loaded study with all state restored.

        Raises
        ------
        ValueError
            If the archive has an incompatible schema version.
        FileNotFoundError
            If the file does not exist.
        """
        import jax.numpy as jnp

        with zipfile.ZipFile(filepath, "r") as zf:
            meta = json.loads(zf.read("meta.json"))

            # Version check
            schema_ver = meta.get("schema_version", "0.0.0")
            major = int(schema_ver.split(".")[0])
            if major > int(_SCHEMA_VERSION.split(".")[0]):
                raise ValueError(
                    f"Archive schema version {schema_ver} is newer than supported "
                    f"version {_SCHEMA_VERSION}. Please upgrade jaxsr."
                )

            study_data = json.loads(zf.read("study.json"))

            # Read arrays
            X_design = None
            X_observed = None
            y_observed = None
            if "X_design.npy" in zf.namelist():
                X_design = _bytes_to_array(zf.read("X_design.npy"))
            if "X_observed.npy" in zf.namelist():
                X_observed = _bytes_to_array(zf.read("X_observed.npy"))
            if "y_observed.npy" in zf.namelist():
                y_observed = _bytes_to_array(zf.read("y_observed.npy"))

        # Reconstruct categories with int keys
        categories = None
        if study_data.get("categories"):
            categories = {int(k): v for k, v in study_data["categories"].items()}

        # Create study
        study = cls(
            name=study_data["name"],
            factor_names=study_data["factor_names"],
            bounds=[tuple(b) for b in study_data["bounds"]],
            feature_types=study_data.get("feature_types"),
            categories=categories,
            description=study_data.get("description", ""),
        )

        # Restore internal state
        study._meta = meta
        study._design_config = study_data.get("design_config", {})
        study._X_design = X_design
        study._observation_status = study_data.get("observation_status", [])
        study._X_observed = X_observed
        study._y_observed = y_observed
        study._library_config = study_data.get("library_config")
        study._model_config = study_data.get("model_config")
        study._result_dict = study_data.get("result")
        study._iterations = [Iteration.from_dict(d) for d in study_data.get("iterations", [])]

        # Reconstruct fitted model if result data exists
        if study._result_dict is not None and study._library_config is not None:
            library = BasisLibrary.from_dict(study._library_config)
            model_kwargs = dict(study._model_config) if study._model_config else {}
            model = SymbolicRegressor(basis_library=library, **model_kwargs)

            result_data = study._result_dict
            parametric_params = result_data.get("parametric_params")
            if parametric_params is not None:
                parametric_params = {int(k): v for k, v in parametric_params.items()}
            model._result = SelectionResult(
                coefficients=jnp.array(result_data["coefficients"]),
                selected_indices=jnp.array(result_data["selected_indices"]),
                selected_names=result_data["selected_names"],
                mse=result_data["mse"],
                complexity=result_data["complexity"],
                aic=result_data["aic"],
                bic=result_data["bic"],
                aicc=result_data["aicc"],
                n_samples=result_data["n_samples"],
                parametric_params=parametric_params,
            )
            model._is_fitted = True

            # Restore training data so predict/suggest work
            if X_observed is not None and y_observed is not None:
                model._X_train = jnp.array(X_observed)
                model._y_train = jnp.array(y_observed)

            study._model = model

        return study

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable summary of the study state.

        Returns
        -------
        summary : str
            Multi-line summary string.
        """
        lines = [
            "=" * 60,
            f"DOE Study: {self.name}",
            "=" * 60,
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        lines.append(f"Factors: {', '.join(self.factor_names)}")
        lines.append(f"Bounds: {self.bounds}")

        if self.feature_types:
            lines.append(f"Feature types: {self.feature_types}")
        if self.categories:
            lines.append(f"Categories: {self.categories}")

        lines.append("")
        if self._X_design is not None:
            n_design = len(self._X_design)
            n_pending = sum(1 for s in self._observation_status if s == "pending")
            n_completed = sum(1 for s in self._observation_status if s == "completed")
            lines.append(
                f"Design: {n_design} points " f"({n_completed} completed, {n_pending} pending)"
            )
            lines.append(f"Design method: {self._design_config.get('method', 'unknown')}")
        else:
            lines.append("Design: not created yet")

        lines.append(f"Observations: {self.n_observations}")

        if self.is_fitted:
            lines.append("")
            lines.append(f"Model: {self._model.expression_}")
            lines.append(f"  MSE: {self._model._result.mse:.6g}")
            lines.append(f"  AIC: {self._model._result.aic:.4f}")
            lines.append(f"  Terms: {self._model._result.n_terms}")
        else:
            lines.append("Model: not fitted yet")

        if self._iterations:
            lines.append("")
            lines.append(f"Iterations: {len(self._iterations)}")
            for it in self._iterations:
                status = f"  Round {it.round_number}: +{it.n_points_added} points"
                if it.model_expression:
                    status += f" → {it.model_expression}"
                if it.notes:
                    status += f" ({it.notes})"
                lines.append(status)

        lines.append("")
        lines.append(f"Created: {self._meta.get('created', 'unknown')}")
        lines.append(f"Modified: {self._meta.get('modified', 'unknown')}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "unfitted"
        return (
            f"DOEStudy(name={self.name!r}, factors={self.n_factors}, "
            f"observations={self.n_observations}, {fitted})"
        )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _touch_modified(self) -> None:
        """Update the modified timestamp."""
        self._meta["modified"] = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Module-level helpers
# =============================================================================


def _get_jaxsr_version() -> str:
    """Get the installed jaxsr version string."""
    try:
        from . import __version__

        return __version__
    except ImportError:
        return "unknown"


def _array_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to .npy format bytes."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def _bytes_to_array(data: bytes) -> np.ndarray:
    """Deserialize bytes in .npy format to a numpy array."""
    buf = io.BytesIO(data)
    return np.load(buf)
