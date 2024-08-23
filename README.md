# flubendazol-recon
Recognizing the Flubendazol amorphization technique using Tensorflow/Keras

---

## Running the experiment

To run the experiment, clone this repo and perform the following steps:

### Generating the synthetic dataset

```
python3 generate_dataset.py
```

### Random train/validation/test separation

```
python3 separate_train_test.py
```

### Training the model

```
python3 train.py
```

### Testing the results

```
python3 test.py
```
