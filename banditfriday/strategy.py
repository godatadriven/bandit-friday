from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.random import random
from banditfriday.products import Product, ALL_PRODUCTS
from typing import List

class Strategy(metaclass=ABCMeta):
    def __init__(self, products: List[Product]):
        self.products = products

    @abstractmethod
    def get_recommendation(self, age: float, wealth: float) -> int:
        pass


    @abstractmethod
    def pass_feedback(self, product_id: int, reward: bool) -> None:
        pass


class BaselineStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.products_popularity = [0] * len(self.products)

    def get_recommendation(self, age: float, wealth: float) -> int:
        return np.argmax(self.products_popularity)

    def pass_feedback(self, product_id: int, reward: bool) -> None:
        self.products_popularity[product_id] += 1


def simulate(strategies: List[Strategy], products: List[Product] = ALL_PRODUCTS, steps: int = 100):
    for strategy in strategies:
        total_reward = 0
        for _ in range(steps):
            # sample random age and wealth
            age = random()
            wealth = random()

            # get recommendation from strategy
            product_id = strategy.get_recommendation(age=age, wealth=wealth)

            # pass reward to strategy
            reward = products[product_id].is_bought_by(age=age, wealth=wealth)
            total_reward += reward
            strategy.pass_feedback(product_id=product_id, reward=reward)
        print(f"{strategy}. Total reward: {total_reward} out of {steps}")
