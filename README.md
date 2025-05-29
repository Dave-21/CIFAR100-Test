# Assignment #1 Report

## Introduction
The CIFAR-100 dataset has 100 classes each containing 600 sample images, making image classification harder. This assignment looks into varyong loss functions, optimizers, learning rates, data augmentation, ResNet-based architectures, and how these impact classification performance.

## Dataset
The CIFAR-100 dataset is used which consists of 60,000 sample images. Data augmentation techniques like random cropping, horizontal flipping, and others to improve generalization.

## Methods
A ResNet-based architecture was implemented in PyTorch, using residual blocks and batch normalization. 20 combinations of these hyperparameters:
* Loss functions: **CrossEntropy**, **LabelSmoothing**
* Optimizers: **Adam**, **SGD**, **RMSprop**
* Learning Rates: **0.01**,   **0.001**,  
 **0.0005**

 Each config used **10** epochs on an **``RTX 3080``** for training.

## Experiments
A handful of different model architectures were used but the same preprocessing pipeline. The biggest focus was on the loss function, optimizer, and learning rate. Accuracy was recoreded at each epoch. Learning rate decay was not applied to show the effect of the base learning rate.

## Results
* Accuracy over epochs were graphed for each config.
* A JSON file contained max test accuracy for each config.
* The best accuracy observed: **~67.898%**

## Future Work
Label smoothing consistently improved the generalization and performed even better with the Adam optimizer. Higher Learning rates usually underperformed in early epochs.
Future versions could include:
* Learning rate schedules
* Different network depths
* Advanced optimizers like AdamW or Lookahead

## References







## Images

![This is an alt text.](/image/sample.webp "This is a sample image.")

## Links

You may be using [Markdown Live Preview](https://markdownlivepreview.com/).

## Blockquotes

> Markdown is a lightweight markup language with plain-text-formatting syntax, created in 2004 by John Gruber with Aaron Swartz.
>
>> Markdown is often used to format readme files, for writing messages in online discussion forums, and to create rich text using a plain text editor.

## Tables

| Left columns  | Right columns |
| ------------- |:-------------:|
| left foo      | right foo     |
| left bar      | right bar     |
| left baz      | right baz     |

## Blocks of code

```
let message = 'Hello world';
alert(message);
```

## Inline code

This web site is using `markedjs/marked`.
