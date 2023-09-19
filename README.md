# Multimodal Biosensing

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

As our model requires three seperate stages of training, we have provided scripts to trian the model from end-to-end. Furthermore, fully trained models
are available in the `models` folder. We include `state_dicts` for each of the 3 stages in the training process:
- `enc.dat` is the pretrained model after single-variable encoding
- `sv_modules_wrapper.dat` is the model after the single-variable prototype stage
- `multivariable_prototypes.dat` is the model after learning the multivariable prototypes

If you want to train the model yourself on these datasets, you can run:
```train
python -m src.models.<dataset>.end_to_end
```
where <dataset> is the specific dataset you would like to train your model on (e.g. basicmotions).

## Evaluation
We've also included scripts to evaluate the fully trained model. Run:
```eval
python -m src.models.<dataset>.eval
```
to evaluate the model on a dataset's test set.

## Results
| Dataset | Accuracy | Command
| ------- | -------- | ------- |
| BasicMotions | 0.9973 | python -m src.models.basicmotions.eval |
