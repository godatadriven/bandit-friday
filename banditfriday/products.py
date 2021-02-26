from abc import ABCMeta, abstractmethod
from typing import List

import matplotlib.pyplot as plt
from numpy import array, argmax, linspace, mean
from numpy.random import random
from scipy.stats import expon, norm


class Product(metaclass=ABCMeta):
    def __init__(self, mean_probability: float = 0.1):
        self.norm = None
        self.norm = mean_probability / mean([mean(x) for x in self.matrix])

    def is_bought_by(self, age: float, wealth: float) -> bool:
        return random() < self.p(age, wealth)

    def p(self, age: float, wealth: float) -> float:
        if self.norm is None:
            return self._p(age, wealth)
        probability = self._p(age, wealth) * self.norm
        return min(1, probability)

    @abstractmethod
    def _p(self, age: float, wealth: float) -> float:
        pass

    @property
    def matrix(self) -> List[List[float]]:
        return [
            [self.p(age, wealth) for age in linspace(0, 1, 100)]
            for wealth in linspace(0, 1, 100)
        ]

    def show(self, fig: plt.Figure, ax: plt.Axes) -> None:
        pos = ax.imshow(self.matrix, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(pos, ax=ax)
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel("age")
        ax.set_ylabel("wealth")


class Beer(Product):
    def _p(self, age: float, wealth: float) -> float:
        if age < 0.2:
            return 0
        return 0.3 - age * 0.1 + (1 - wealth) * 0.2


class Diapers(Product):
    def _p(self, age: float, wealth: float) -> float:
        p = norm(loc=age, scale=0.2).pdf(0.3)
        return p if wealth > 0.2 else 0


class Lollipops(Product):
    def _p(self, age: float, wealth: float) -> float:
        p = expon().pdf(age) * (1 - wealth * 0.2)
        return p


class Potatoes(Product):
    def _p(self, age: float, wealth: float) -> float:
        return max(0.0, age - wealth ** 2)


def plot_product_probabilities(*products: Product) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = [ax for row in axes for ax in row]
    for product, ax in zip(products, axes):
        product.show(fig, ax)


def plot_max_probabilities(*products: Product) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = [ax for row in axes for ax in row]
    max_probs = argmax(array([p.matrix for p in products]), axis=0)
    for i, (product, ax) in enumerate(zip(products, axes)):
        ax.imshow(max_probs == i)
        ax.set_title(product.__class__.__name__)


ALL_PRODUCTS = [Beer(), Diapers(), Lollipops(), Potatoes()]
