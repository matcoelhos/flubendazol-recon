# flubendazol-recon
Recognizing the Flubendazol amorphization technique using Tensorflow/Keras

If you use this, please cite us:
```
@article{coelhosilva2024,
  title={Towards a Machine-Learning-Based Application for Amorphous Drug Recognition},
  volume={22}, 
  url={https://latamt.ieeer9.org/index.php/transactions/article/view/8988},
  abstractNote={The amorphous drug structure represents an important feature to be reached in the pharmaceutical field due to its possibility of increasing drug solubility, considering that at least 40% of commercially available   crystalline drugs are poorly soluble in water. However, it is known that the amorphous local structure can vary depending on the amorphization technique used. Therefore, recognizing such variations related to a specific amorphization technique through the pair distribution function (PDF) method, for example, is an important tool for drug characterization concerns. This work presents a method to classify amorphous drugs according to their amorphization techniques and related to the local structure variations using machine learning. We used experimental PDF patterns obtained from low-energy X-rays scattering data to extract information and expanded the data through the Monte Carlo method to create a synthetic dataset. Then, we proposed the evaluation of such a technique using a Deep Neural Network. Based on the results obtained, it is suggested that the proposed technique is suitable for the amorphization technique and local structure recognition task.},
  number={9},
  journal={IEEE Latin America Transactions},
  author={Coelho Silva, Mateus and Castro e Silva, Alcides and T. D. Orlando, Marcos and D. N. Bezzon, Vinicius},
  year={2024},
  month={Aug.},
  pages={755–760}
}
```

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
