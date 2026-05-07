# Multimodal Biosensing

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

You can train each stage separately (the encoders, the single-variable prototypes, and multivariable prototypes).
1. `python3 -m src.train.encoding.train --dataset <dataset>` to train the univariate encoders.
2. `python3 -m src.train.single_variable_prototypes.kmeans_silhouette --dataset <dataset>` to compute the optimal number of single-variable prototypes for each variable in your data.
3. `python3 -m src.train.single_variable_prototypes.train --dataset <dataset> --initialize` to initialize the single-variable prototypes with k-means++ and then train them.
4. `python3 -m src.train.multivariable_prototypes.train --dataset <dataset>` to train the multivariable prototypes.

Once you've determined your hyperparameters, you can run the pipeline end-to-end:
`python3 -m src.eval.end_to_end --dataset <dataset> --resamples <resamples>`

Our learned models are located in `src/models` and are organized by dataset. `src/models/comparisons` contains saved models for our ablation studies on our simulated dataset. 

## Evaluation
To evaluate the fully trained model on the test set:
```eval
python -m src.eval.eval --dataset <dataset>
```
