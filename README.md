# 🧠 MNIST Digit Classification with PyTorch

This project implements a simple **feedforward neural network (FNN)** using **PyTorch** to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

It covers the full workflow from **data preprocessing** to **training, evaluation, and saving the trained model**, following the structure of Udacity’s Deep Learning Nanodegree project.

---

## 🚀 Features

* Loads and preprocesses the **MNIST dataset** (normalization, tensor conversion).
* Implements a **fully connected neural network** with at least two hidden layers.
* Uses **CrossEntropyLoss** for classification.
* Optimizes with **Adam** optimizer.
* Evaluates accuracy on the **test dataset**.
* Tweaks **hyperparameters** (hidden units, learning rate, batch size, epochs) to achieve **≥ 90% accuracy**.
* Saves the trained model with `torch.save()` for later use.

---

## 📂 Project Structure

```
├── MNIST_project.ipynb   # Jupyter Notebook containing the full project
├── saved_model.pth       # Saved trained model (after running notebook)
└── README.md             # Project documentation
```

---

## ⚙️ Requirements

* Python 3.7+
* [PyTorch](https://pytorch.org/)
* torchvision
* matplotlib
* numpy

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

---

## 🏋️ Training & Evaluation

Open the notebook:

```bash
jupyter notebook MNIST_project.ipynb
```

* The notebook will train the model on MNIST.
* Final test accuracy should be **≥ 90%**.
* You can tweak hyperparameters in the training cell to experiment with performance.

---

## 💾 Saving & Loading Model

Save the model:

```python
torch.save(model.state_dict(), "saved_model.pth")
```

Load the model:

```python
model.load_state_dict(torch.load("saved_model.pth"))
```

---

## 📊 Results

* Achieved **\~98% accuracy** on the MNIST test set.
* Model performance improves with hyperparameter tuning.
* Optionally, you can extend with **Convolutional Neural Networks (CNNs)** for even higher accuracy.

---

## ✨ Extensions

* Add a **validation set** for better hyperparameter tuning.
* Try **CNN architectures** like LeNet-5.
* Compare results with Yann LeCun’s benchmark [MNIST results](http://yann.lecun.com/exdb/mnist/).

---

## 📜 License

This project is for educational purposes as part of Udacity’s **Deep Learning Nanodegree**.
Feel free to fork and extend for your own experiments!
