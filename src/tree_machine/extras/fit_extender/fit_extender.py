"""
Extras for training models.
"""
import typing as tp

from tree_machine.types import GroundTruth, Inputs, Predictions


@tp.runtime_checkable
class IsEstimator(tp.Protocol):
    def fit(self, X: Inputs, y: GroundTruth, **fit_params) -> "IsEstimator":
        ...

    def predict(self, X: Inputs) -> Predictions:
        ...


class ExtenderResults(tp.TypedDict):
    X: Inputs
    y: GroundTruth
    attrs: dict[str, object]


@tp.runtime_checkable
class IsExtender(tp.Protocol):
    def fit_(self, X: Inputs, y: GroundTruth, **extend_params) -> ExtenderResults:
        ...

    def predict_(self, X: Inputs) -> Inputs:
        ...


def fit_extend(extended_name: str, Estimator: tp.Type, Extender: tp.Type) -> tp.Type:
    if issubclass(Estimator, IsEstimator):
        if issubclass(Extender, IsExtender) and not issubclass(Extender, IsEstimator):
            if not issubclass(Estimator, Extender):

                class Extended(Estimator, Extender):
                    def fit(self, X: Inputs, y: GroundTruth, **fit_params):
                        captured = {
                            k.split("__")[1]: v
                            for k, v in fit_params.items()
                            if k.startswith(f"{Extender.__name__}__")
                        }

                        not_captured = {
                            k: v
                            for k, v in fit_params.items()
                            if not k.startswith(f"{Extender.__name__}__")
                        }

                        results = Extender.fit_(self, X, y, **captured)
                        for attrib, value in results["attrs"].items():
                            setattr(self, f"{Extender.__name__}__{attrib}", value)

                        return Estimator.fit(
                            self, results["X"], results["y"], **not_captured
                        )

                    def predict(self, X: Inputs):
                        """Extends .predict() for new estimator."""
                        return Estimator.predict(self, Extender.predict_(self, X))

                Extended.__name__ = extended_name
                return Extended

            raise TypeError("Estimator is already extended with this type.")

        raise TypeError("Extender type is not a valid extender implementation.")

    raise TypeError("Estimator does not implement .fit() method.")
