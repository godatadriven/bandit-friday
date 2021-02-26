import operator
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Optional, Type, Tuple

from numpy.random import random, beta
from pandas import DataFrame

from banditfriday.products import Product, ALL_PRODUCTS


class Strategy(metaclass=ABCMeta):
    def __init__(
        self, products: Dict[str, Product], history: Optional[DataFrame] = None
    ):
        self.products = products
        if history is not None:
            self.learn_from_history(history)

    def learn_from_history(self, df: DataFrame):
        """Poor man's update."""
        for _, row in df.iterrows():
            for key, value in row.to_dict().items():
                if key in self.products:
                    self.pass_feedback(
                        age=row["age"],
                        wealth=row["wealth"],
                        reward=value,
                        product_name=key,
                    )

    @abstractmethod
    def get_recommendation(self, age: float, wealth: float) -> str:
        pass

    @abstractmethod
    def pass_feedback(
        self, age: float, wealth: float, product_name: str, reward: bool
    ) -> None:
        pass


class BinnedStrategy(Strategy):
    def __init__(
        self,
        strategy: Type[Strategy],
        products: Dict[str, Product],
        history: Optional[DataFrame] = None,
        n_bins: int = 2,
    ):
        self.n_bins = n_bins
        self.strategies = [
            [strategy(products=products) for _ in range(n_bins)] for _ in range(n_bins)
        ]
        super().__init__(products=products, history=history)

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


class BaselineStrategy(Strategy):
    def __init__(self, products: Dict[str, Product], **kwargs):
        self.products_popularity = {key: 0 for key in products}
        super().__init__(products=products, **kwargs)

    def learn_from_history(self, df: DataFrame):
        for product_name in self.products:
            if product_name in df.columns:
                self.products_popularity[product_name] = df[product_name].sum()

    def get_recommendation(self, age: float, wealth: float) -> str:
        """Recommend the product with the highest popularity"""
        return max(self.products_popularity.items(), key=operator.itemgetter(1))[0]

    def pass_feedback(
        self, age: float, wealth: float, product_name: str, reward: bool
    ) -> None:
        self.products_popularity[product_name] += 1


class ThompsonSampling(Strategy):
    def __init__(self, products: Dict[str, Product], **kwargs):
        self.counts = {key: 0 for key in products}
        self.rewards = {key: 0 for key in products}
        super().__init__(products=products, **kwargs)

    def get_recommendation(self, age: float, wealth: float) -> str:
        draw = {
            arm: beta(self.rewards[arm] + 1, self.counts[arm] - self.rewards[arm] + 1)
            for arm in self.counts.keys()
        }
        return max(draw.items(), key=lambda x: x[1])[0]

    def pass_feedback(
        self, age: float, wealth: float, product_name: str, reward: bool
    ) -> None:
        self.counts[product_name] += 1
        self.rewards[product_name] += int(reward)


def simulate(
    strategies: List[Strategy],
    products: Dict[str, Product] = ALL_PRODUCTS,
    steps: int = 100,
):
    for strategy in strategies:
        total_reward = 0
        for _ in range(steps):
            # sample random age and wealth
            age = random()
            wealth = random()

            # get recommendation from strategy
            product_name = strategy.get_recommendation(age=age, wealth=wealth)

            # pass reward to strategy
            reward = products[product_name].is_bought_by(age=age, wealth=wealth)
            total_reward += reward
            strategy.pass_feedback(
                product_name=product_name, reward=reward, age=age, wealth=wealth
            )
        print(f"{strategy}. Total reward: {total_reward} out of {steps}")
        try:
            print(strategy.products_popularity)
        except:
            pass
