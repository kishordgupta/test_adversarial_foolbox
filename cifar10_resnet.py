#pip install foolbox 

import foolbox
import numpy as np
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
# instantiate model (supports PyTorch, Keras, TensorFlow (Graph and Eager), JAX, MXNet and many more)
model = models.alexnet(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

# get a batch of images and labels and print the accuracy
images, labels = foolbox.utils.samples(dataset='cifar10', batchsize=16, data_format='channels_first', bounds=(0, 1))
print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))
# -> 0.9375

# apply the attack
attack = foolbox.attacks.BasicIterativeMethod(fmodel)
adversarials = attack(images, labels)
# if the i'th image is misclassfied without a perturbation, then adversarials[i] will be the same as images[i]
# if the attack fails to find an adversarial for the i'th image, then adversarials[i] will all be np.nan

# Foolbox guarantees that all returned adversarials are in fact in adversarials
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
# -> 0.0

adversarial = np.array(adversarials)
for i in range(adversarial.shape[0]):
  plt.imsave('./bim/'+str(i)+'.png',adversarial[i].transpose((1,2,0)))

attack = foolbox.attacks.AdamL1BasicIterativeAttack(fmodel)
adversarials = attack(images, labels)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
adversarial = np.array(adversarials)
for i in range(adversarial.shape[0]):
  plt.imsave('./abim/'+str(i)+'.png',adversarial[i].transpose((1,2,0)))
  
attack = foolbox.attacks.AdamL2BasicIterativeAttack(fmodel)
adversarials = attack(images, labels)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
adversarial = np.array(adversarials)
for i in range(adversarial.shape[0]):
  plt.imsave('./abiml2/'+str(i)+'.png',adversarial[i].transpose((1,2,0)))


attack = foolbox.attacks.PGD(fmodel)
adversarials = attack(images, labels)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
adversarial = np.array(adversarials)
for i in range(adversarial.shape[0]):
  plt.imsave('./PGD/'+str(i)+'.png',adversarial[i].transpose((1,2,0)))

attack = foolbox.attacks.AdamPGD(fmodel)
adversarials = attack(images, labels)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
adversarial = np.array(adversarials)
for i in range(adversarial.shape[0]):
  plt.imsave('./APGD/'+str(i)+'.png',adversarial[i].transpose((1,2,0)))


attack = foolbox.attacks.RandomPGD(fmodel)
adversarials = attack(images, labels)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
adversarial = np.array(adversarials)
for i in range(adversarial.shape[0]):
  plt.imsave('./RPGD/'+str(i)+'.png',adversarial[i].transpose((1,2,0)))

attack = foolbox.attacks.MomentumIterativeAttack(fmodel)
adversarials = attack(images, labels)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
adversarial = np.array(adversarials)
for i in range(adversarial.shape[0]):
  plt.imsave('./MBIM/'+str(i)+'.png',adversarial[i].transpose((1,2,0)))

