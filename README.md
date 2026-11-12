# Digit Recognizer

A small CNN ensemble for the Kaggle [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition (MNIST, 28x28 grayscale digits).

**Public leaderboard score: 0.99632**
Validation accuracy on a 2,100-image holdout: 99.67% for the ensemble (single models: 99.57%, 99.67%, 99.52%).

## How it works

- Three copies of the same CNN, trained with different seeds, with their softmax outputs averaged.
- The net is five 3x3 conv blocks (conv, batch norm, ReLU) with two max-pools, dropout 0.2, global average pooling and one linear layer. About 140k parameters.
- Augmentation runs on the GPU in `train.py`: random rotation (±12°), scale (±10%) and shift (about ±1.7 px), applied per batch with `affine_grid`.
- AdamW with a OneCycle schedule (peak lr 3e-3), 30 epochs, label smoothing 0.05, batch size 128.
- 5% of the training set is held out for validation. The final models train on the other 95%.

Training takes about 90 seconds per model on a free Kaggle GPU.

## Run it

On Kaggle (needs the Kaggle CLI and a token):

```
kaggle kernels push -p .
kaggle kernels output <your-username>/digit-recognizer-cnn-ensemble -p out
kaggle competitions submit -c digit-recognizer -f out/submission.csv -m "cnn ensemble"
```

Edit the `id` in `kernel-metadata.json` to your own username first.

Locally (needs PyTorch, a GPU helps):

```
kaggle competitions download -c digit-recognizer -p data
unzip data/digit-recognizer.zip -d data
python train.py
```

This writes `submission.csv` and `results.json`.

## Files

- `train.py`: data loading, augmentation, model, training, ensembling
- `kernel-metadata.json`: Kaggle kernel config (GPU on, internet off, competition data attached)
- `results.json`: validation accuracies from the submitted run

## History

The first version of this repo was a Colab notebook comparing classical models (KNN, SVM, random forest, XGBoost, PCA) and a few Keras networks. It is still in the git history.

Competition citation: AstroDave and Will Cukierski. Digit Recognizer. https://kaggle.com/competitions/digit-recognizer, 2012. Kaggle.
