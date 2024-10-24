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
    def pre_fit(self, X: Inputs, y: GroundTruth, **extend_params) -> ExtenderResults:
        ...

    def pre_predict(self, X: Inputs) -> Inputs:
        ...


def fit_extend(extended_name: str, Estimator: tp.Type, Extender: tp.Type) -> tp.Type:
    if issubclass(Estimator, IsEstimator):
        if issubclass(Extender, Extender) and not issubclass(Extender, IsEstimator):
            if not issubclass(Estimator, Extender):

                class Extended(Estimator, Extender):
                    def fit(self, X: Inputs, y: GroundTruth, **fit_params):
                        """Extends .fit() for new estimator."""
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

                        results = Extender.pre_fit(self, X, y, **captured)
                        for attrib, value in results["attrs"].items():
                            setattr(self, attrib, value)

                        return Estimator.fit(self, X, y, **not_captured)

                    def predict(self, X: Inputs):
                        """Extends .predict() for new estimator."""
                        return Estimator.predict(self, Extender.pre_predict(self, X))

                Extended.__name__ = extended_name
                return Extended

            raise TypeError("Estimator is already extended with this type.")

        raise TypeError("Extender type is not a valid extender implementation.")

    raise TypeError("Estimator does not implement .fit() method.")
