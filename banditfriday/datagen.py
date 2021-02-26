from banditfriday.products import Diapers, Beer, Lollipops, Raspberries
import numpy as np
import pandas as pd
from typing import Dict
from banditfriday.products import ALL_PRODUCTS, Product


def generate_product_data(products: Dict[str, Product], size: int = 100):

    """
     Generates product data for 4 products and saves it as a csv

    :param products: a dict that includes all needed products as class instances
    :param size: number of customers to generate in this dataset, 100 by default

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


def regenerate_data():
    df = generate_product_data(ALL_PRODUCTS, size=250)
    df.to_csv("banditfriday/products_all.csv", index=False)
    df = generate_product_data(
        {
            name: product
            for name, product in ALL_PRODUCTS.items()
            if name in ("Diapers", "Beer", "Lollipops", "Raspberries")
        },
        size=100,
    )
    df.to_csv("banditfriday/products4.csv", index=False)


# to get some dataset and save it use for example the below:
# df = generate_product_data([Diapers(), Beer(), Lollipops(), Raspberries()], size = 200)
# df.to_csv("products4.csv", index = False)
