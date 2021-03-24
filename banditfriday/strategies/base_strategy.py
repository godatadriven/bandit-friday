from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

from pandas import DataFrame

from banditfriday.products import Product


class BaseStrategy(metaclass=ABCMeta):
    def __init__(
        self, products: Dict[str, Product], history: Optional[DataFrame] = None
    ):
        self.name = type(self).__name__
        self.products = products
        if history is not None:
            self.learn_from_history(history)
            self.name += " - with training data"

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
