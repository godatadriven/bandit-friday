from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from numpy import linspace, mean
from typing import List

from scipy.stats import norm


class Product(metaclass=ABCMeta):
    def __init__(self, mean_probability: float = 0.1):
        self.norm = None
        self.norm = mean_probability / mean([mean(x) for x in self.matrix])

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


def plot_product_probabilities(*products: Product) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = [ax for row in axes for ax in row]
    for product, ax in zip(products, axes):
        product.show(fig, ax)
