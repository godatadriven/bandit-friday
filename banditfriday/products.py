from abc import ABCMeta, abstractmethod
from json import dump, loads
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from numpy import array, argmax, linspace, mean
from numpy.random import random
from pkg_resources import resource_string
from scipy.stats import expon, norm, gamma, gengamma


class Product(metaclass=ABCMeta):
    def __init__(self, normalization: Optional[float] = None):
        self.norm = normalization or self.load_normalization()

    @classmethod
    def load_normalization(cls) -> float:
        return loads(resource_string("banditfriday", "norms.json"))[cls.__name__]

    @classmethod
    def compute_normalization(cls, mean_probability: float) -> float:
        p = cls(normalization=0.01)
        return 0.01 * mean_probability / mean([mean(x) for x in p.matrix])

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


class CheapToiletPaper(Product):
    def _p(self, age: float, wealth: float) -> float:
        if wealth < 0.05:
            return 0.1
        return 0


class Diapers(Product):
    def _p(self, age: float, wealth: float) -> float:
        p = norm(loc=age, scale=0.2).pdf(0.3)
        return p if wealth > 0.2 else 0


class Lollipops(Product):
    def _p(self, age: float, wealth: float) -> float:
        p = expon().pdf(age) * (1 - wealth * 0.2)
        return p


class Raspberries(Product):
    def _p(self, age: float, wealth: float) -> float:
        p = gamma.pdf(age * wealth, a=5)
        return p


class Potatoes(Product):
    def _p(self, age: float, wealth: float) -> float:
        return max(0.0, age - wealth ** 2)


class Sushi(Product):
    def _p(self, age: float, wealth: float) -> float:
        return gengamma(a=3, c=-3).pdf(age + 0.4) * wealth ** 2


def plot_product_probabilities(products: Dict[str, Product]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = [ax for row in axes for ax in row]
    for product, ax in zip(products.values(), axes):
        product.show(fig, ax)


def plot_max_probabilities(products: Dict[str, Product]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = [ax for row in axes for ax in row]
    max_probs = argmax(array([p.matrix for p in products.values()]), axis=0)
    for i, (product, ax) in enumerate(zip(products.values(), axes)):
        ax.imshow(max_probs == i)
        ax.set_title(product.__class__.__name__)
        ax.set_xlabel("age")
        ax.set_ylabel("wealth")


def compute_normalizations() -> None:
    results = dict()
    probabilities = {"Beer": 0.3, "CheapToiletPaper": 0.01}
    for product_class in Product.__subclasses__():
        if product_class.__name__ not in results:
            results[product_class.__name__] = product_class.compute_normalization(
                probabilities.get(product_class.__name__, 0.175)
            )
    with open("banditfriday/norms.json", "w") as f:
        dump(results, f)


ALL_PRODUCTS = {product.__name__: product() for product in Product.__subclasses__()}
