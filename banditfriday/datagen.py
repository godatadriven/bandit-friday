from typing import Dict

import numpy as np
import pandas as pd

from banditfriday.products import ALL_PRODUCTS, Product


def generate_product_data(products: Dict[str, Product] = ALL_PRODUCTS, size: int = 250):
    """Generates data based on the defined product probabilities

    Each record represents how a customer would react to each of the products.

    :param products: a dict with all products to include
    :param size: number of records to generate

    """
    data = {}
    data["id"] = list(range(size))
    data["age"] = np.random.random(size=size)
    data["wealth"] = np.random.random(size=size)

    for product_name, product in products.items():
        data[product_name] = [
            product.is_bought_by(x, y) for x, y in zip(data["age"], data["wealth"])
        ]

    return pd.DataFrame(data)
