import numpy as np

def map_categories_to_numbers(categories):
    category_mapping = {'C': 0, 'A': 1, 'F': 2}
    if isinstance(categories, np.ndarray):
        return np.array([category_mapping[cat] for cat in categories])
    else:
        return category_mapping[categories]
    
def map_numbers_to_categories(numbers):
    numbers_mapping = {0: 'C', 1: 'A', 2: 'F'}
    if isinstance(numbers, np.ndarray):
        return np.array([numbers_mapping[cat] for cat in numbers])
    else:
        return numbers_mapping[numbers]