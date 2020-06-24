# Knowledge Distillation

This experiment started with a strong intuition that, specifying a hard probability of 100 percent to one class and 0% probability to other classes, would adversely affect the learning of shared features of each class and result in redution of overall accuracy of a deep learning classification model, when two classes share a set of common features. To reduce this loss of accuracy, the idea was to use the soft class probabilities predicted by a large model with higher accuracy to train a simpler model and compare the results to the results obtained by training using class labels. The results showed that the accuracy of simpler models improved significantly while using results from a pretrained large network, instead of using class labels from the dataset. However, the approach was not anything new, and it was already a well researched area of deep learning and is callled Knowledge Distillation<sup>[1] [2] [3]</sup>.

## Architecture

![Architecture](/images/knowledge_distillation_current.png "Title")

In this approach, the MSE loss is calculated on the logits before the softmax funtion. However the [Kullback-Leibler divergence Loss](https://pytorch.org/docs/master/generated/torch.nn.KLDivLoss.html) can be used on the probability distribution after passing the result through a Softmax function.

This code base can be used for continuing experiments with Knowledge distillation. It is a simple framework for experimenting with your own loss functions in a teacher-student scenario for image classsification. You can train both teacher and student network using the framework and monitor training using Tensorboard <sup>[8]</sup>.

## Code base

- Framework - PyTorch <sup>[5]</sup> and PyTorch-Lightning <sup>[6]</sup>
- Image resolution - 224
- Datasets - Any classification dataset that supports the resolution e.g. Imagenette, Cats Vs Dogs etc. Adjust the number of classes in command line arguments
- Most of the code should be self explanatory. Check commandline arguments for default options.
- An nvidia docker dockerfile with necessary dependenciess is provided for creating your own docker image. Please map application and dataset volumes to train using docker.
- Please refer to PyTorch Lightning <sup>[6]</sup> for saving and loading a checkpoint if needed in your training. This experiment was stopped prematurely due to the existence of other frameworks offering similar functionality for experimentation <sup>[4]</sup>.

## Usage

- Install Nvidia Docker
- Build docker image using the given dockerfile
- Download datasets e.g. Fast AI Imagenette dataset <sup>[7]</sup>
- Set the required command line parameters inside `commandline_args.txt`
- Run `experiment.py` with required command line parameters inside the docker after mapping code and dataset directories to the docker container
- Set `train_teacher` flag to true for training the teacher network
- Set `distill` flag to true for training student with knowledge distillation enabled
- Make sure the teacher network training is completed before enabling distilling and training the student
- Monitor training progress using Tensorboard
- Default set of parameters are given in `commandline_args.txt`. The flags specified inside `commandline_args.txt` are automatically loaded during runtime

## Results

The following graphs gives the validation accuracy of the same student model on the same dataset when trained with and without knowledge distillation. The higher accuracy graph shows the distillation result.

![Result 1](/images/result1.png "Title")
![Result 1](/images/result2.png "Title")

## References

[1]. C. Buciluˇa, R. Caruana, and A. Niculescu-Mizil. Model compression. In Proceedings of the
12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD
’06, pages 535–541, New York, NY, USA, 2006. ACM.

[2]. E. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network.
arXiv:1503.02531v1, Mar. 2015.

[3]. Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, Hassan Ghasemzadeh:
Improved Knowledge Distillation via Teacher Assistant. AAAI 2020: 5191-5198

[4]. Neta Zmora ,Guy Jacob, Lev Zlotnik, Bar Elharar, Gal NovikNeural: Network Distiller: A Python Package For DNN Compression Research, Oct. 2019

[1]: https://dl.acm.org/doi/10.1145/1150402.1150464 "Model compression"
[2]: https://arxiv.org/abs/1503.02531 "Distilling the Knowledge in a Neural Network"
[3]: https://arxiv.org/abs/1902.03393 "Improved Knowledge Distillation via Teacher Assistant"
[4]: https://arxiv.org/abs/1910.12232 "Neural Network Distiller: A Python Package For DNN Compression Research"
[5]: https://pytorch.org/ "PyTorch"
[6]: https://github.com/PyTorchLightning/pytorch-lightning "PyTorch Lightning"
[7]: https://github.com/fastai/imagenette "Fast AI Imagenette dataset"
[8]: https://www.tensorflow.org/tensorboard "Tensorboard"
