from typing import Dict

from numpy.random import beta

from banditfriday.strategies.base_strategy import BaseStrategy
from banditfriday.products import Product


class ThompsonSampling(BaseStrategy):
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
