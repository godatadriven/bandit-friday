from typing import Dict, Optional, Type

from pandas import DataFrame

from banditfriday.strategies.base_strategy import BaseStrategy
from banditfriday.products import Product


class BinnedStrategy(BaseStrategy):
    def __init__(
        self,
        strategy: Type[BaseStrategy],
        products: Dict[str, Product],
        history: Optional[DataFrame] = None,
        n_bins: int = 2,
    ):
        self.n_bins = n_bins
        self.strategies = [
            [strategy(products=products) for _ in range(n_bins)] for _ in range(n_bins)
        ]
        super().__init__(products=products, history=history)
        self.name = f"{strategy.__name__} - Binned"
        if history is not None:
            self.name += " - with training data"

    def learn_from_history(self, df: DataFrame):
        for (age, wealth), sub_df in (
            df.assign(age=lambda d: d["age"].apply(self._index))
            .assign(wealth=lambda d: d["wealth"].apply(self._index))
            .groupby(["age", "wealth"])
        ):
            self.strategies[age][wealth].learn_from_history(df=sub_df)

    def get_recommendation(self, age: float, wealth: float) -> str:
        return self.strategies[self._index(age)][
            self._index(wealth)
        ].get_recommendation(age, wealth)

    def pass_feedback(
        self, age: float, wealth: float, product_name: str, reward: bool
    ) -> None:
        return self.strategies[self._index(age)][self._index(wealth)].pass_feedback(
            age, wealth, product_name, reward
        )

    def _index(self, parameter: float) -> int:
        return min(self.n_bins - 1, int(parameter * self.n_bins))
