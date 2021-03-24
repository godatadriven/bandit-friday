from typing import List, Dict

from numpy.random import random

from banditfriday.strategies.base_strategy import BaseStrategy
from banditfriday.products import Product, ALL_PRODUCTS


def simulate(
    strategies: List[BaseStrategy],
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
