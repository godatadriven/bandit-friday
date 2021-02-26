from banditfriday.products import Diapers, Beer, Lollipops, Raspberries
import numpy as np
import pandas as pd

def generate_product_data(product_list: list, size = 100):
    
    """
     Generates product data for 4 products and saves it as a csv

    :param product_list: a list that includes all needed products as class instances
    :param size: number of customers to generate in this dataset, 100 by default
    
    """
    
    product_dict = {type(prod).__name__ : prod for prod in product_list}
    
    data = {}
    data['id'] = list(range(size))
    data['age'] = np.random.random(size = size)
    data['wealth'] = np.random.random(size = size)
    
    for prod_name, prod_instance in product_dict.items():
        
        data[prod_name.lower()] = [prod_instance.is_bought_by(x, y) for x, y in zip(data['age'], data['wealth'])]
        
        
    return pd.DataFrame(data)
    

# to get some dataset and save it use for example the below:
# df = generate_product_data([Diapers(), Beer(), Lollipops(), Raspberries()], size = 200)
# df.to_csv("products4.csv", index = False)