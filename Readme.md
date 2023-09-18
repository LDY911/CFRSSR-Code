# CFRSSR

The official implementation of "Independent or Social Driven Decision? A Counterfactual Reinforcement Strategy for Graph-Based Social Recommendation"

## Requirements:

- recbole==1.1.1
- pyg>=2.0.4
- pytorch>=1.7.0
- python>=3.7.0
- numba==0.53.1
- numpy==1.20.3
- scipy==1.6.2
- tensorflow==1.14.0

## Instructions:

### Run DiffNet on Ciao:

1. Run `run_counterfactual_generation.py` and set the parameter thresh to 3, 7 to generate counterfactual data.
2. Run `weighted.py` to generate weighted data.
3. Finally, run the `run_main.py` file to get the results.

### Run MHCN on Douban:

1. Run `run_counterfactual_generation.py` and set the parameter thresh to 2, 1 to generate counterfactual data.
2. Paste the generated `Douban_2, 1_influenced_inter_data_j.csv` file into the `/dataset/Douban` folder.
3. Finally, run `main.py` to get the results.

## Acknowledgements:

We would like to express our gratitude to [Recbole](https://github.com/RUCAIBox/RecBole) and [SELFRec](https://github.com/Coder-Yu/SELFRec) for their outstanding work.

## License

This project is licensed under the terms of the MIT license.
