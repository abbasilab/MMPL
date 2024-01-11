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


## Results
| Dataset | Accuracy
| ------- | -------- |
| Simulated | 0.9846 |
| CharacterTrajectories | 0.9656 |
| Epilepsy | 0.9686 |
| BasicMotions | 0.9975 |

## Visualizations
Learned latent spaces & univariate prototypes for simulated dataset:

<img width="548" alt="Screen Shot 2024-01-11 at 12 12 39 AM" src="https://github.com/bhaveshk658/mmbs/assets/59468276/ecbb2511-e47c-44db-bf12-0afc953728bb">

Learned multivariate prototypes:

<img width="715" alt="Screen Shot 2024-01-11 at 12 13 12 AM" src="https://github.com/bhaveshk658/mmbs/assets/59468276/b2794241-c6b6-4667-82e0-33bc379c8df7">

Nearest neighbor projections of learned prototypes:

<img width="702" alt="Screen Shot 2024-01-11 at 12 13 56 AM" src="https://github.com/bhaveshk658/mmbs/assets/59468276/5211b850-f00f-4d18-be92-944c80e39890">
