from setuptools import setup, find_packages

setup(
    name="banditfriday",
    version="0.42.0",
    description="Bandits on Friday",
    packages=find_packages(include=["banditfriday", "banditfriday.*"]),
    install_requires=["scipy", "numpy", "matplotlib", "pandas"],
    entry_points={
        "console_scripts": [
            "recompute-normalization=banditfriday.products:compute_normalizations"
        ]
    },
)
