import operator
from typing import Dict

from pandas import DataFrame

from banditfriday.strategies.base_strategy import BaseStrategy
from banditfriday.products import Product


class BaselineStrategy(BaseStrategy):
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
