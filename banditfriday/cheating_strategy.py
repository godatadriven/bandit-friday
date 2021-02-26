from typing import Dict

from numpy import argmax, array

from banditfriday.products import Product
from banditfriday.strategy import Strategy


class CheatingStrategy(Strategy):
    def __init__(self, products: Dict[str, Product], **kwargs):
        self.max_probs = argmax(array([p.matrix for p in products.values()]), axis=0)
        super().__init__(products=products, **kwargs)

    def get_recommendation(self, age: float, wealth: float) -> str:
        product_id = self.max_probs[int(wealth*100), int(age*100)]
        return list(self.products.keys())[product_id]

    def pass_feedback(self, product_name: str, reward: bool) -> None:
        pass
