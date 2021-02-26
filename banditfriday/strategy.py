import operator
from abc import ABCMeta, abstractmethod
from typing import List, Dict

from numpy.random import random

from banditfriday.products import Product, ALL_PRODUCTS


class Strategy(metaclass=ABCMeta):
    def __init__(self, products: Dict[str, Product]):
        self.products = products

    @abstractmethod
    def get_recommendation(self, age: float, wealth: float) -> str:
        pass


    @abstractmethod
    def pass_feedback(self, product_name: str, reward: bool) -> None:
        pass


class BaselineStrategy(Strategy):
    def __init__(self, history = None, **kwargs):
        super().__init__(**kwargs)
        self.products_popularity = {key: 0 for key in self.products}
        if history is not None:
            for product_name in self.products:
                if product_name in history.columns:
                    self.products_popularity[product_name] = history[product_name].sum()

    def get_recommendation(self, age: float, wealth: float) -> str:
        """Recommend the product with the highest popularity"""
        return max(self.products_popularity.items(), key=operator.itemgetter(1))[0]

    def pass_feedback(self, product_name: str, reward: bool) -> None:
        self.products_popularity[product_name] += 1


def simulate(strategies: List[Strategy], products: Dict[str, Product] = ALL_PRODUCTS, steps: int = 100):
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
            strategy.pass_feedback(product_name=product_name, reward=reward)
        print(f"{strategy}. Total reward: {total_reward} out of {steps}")
        print(strategy.products_popularity)
