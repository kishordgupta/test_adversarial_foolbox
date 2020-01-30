pip install --upgrade foolbox-native

import foolbox.ext.native as fbn
import torchvision.models as models
import torchvision
import torch
import os

# instantiate a model
model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = fbn.models.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

# get data and test the model
images, labels = fbn.utils.samples(fmodel, dataset='imagenet', batchsize=16)
print(fbn.utils.accuracy(fmodel, images, labels))
# -> 0.9375

# apply the attack

# if torch.cuda.is_available():
#   device = torch.device('cuda')
# else:
#   device = torch.device('cpu')
attack = fbn.attacks.LinfinityBasicIterativeAttack(fmodel)
adversarial = attack(images, labels, epsilon=0.03, step_size=0.005)  # L-inf norm
print(fbn.utils.accuracy(fmodel, adversarial, labels))

# temp = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
#                                        torchvision.transforms.Resize((224,224)),
#                                        torchvision.transforms.ToTensor(),
#                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])])



# for i in range(16):
#   adv_tsor = torch.from_numpy(np.array(adversarial)[i]).to(device)
#   adv_tsor = temp(adv_tsor)
# # adv_tsor = normalize(adv_tsor)
#   adv_tsor.unsqueeze_(0)
#   # _,adv_pred,_,adv_pred_prob,origProb_in_adv = self.getPredictionInfo(adv_tsor,label)
#   # print('Prediction from adversary', int(adv_pred.cpu()),float(adv_pred_prob[0].cpu()))

#   # adv_label = np.argmax(fmodel.predictions(adversarial))       # foolbox prediction 

#   # print('AdvInfoFromFoolbox',adv_label)

#               #####3 saving adversary image#####
#   # save_name = str(img_name[:img_name.find('.')])+str('_')+str(int(round(unp_pred_prob[0])))+str('_')+str(int(adv_pred.cpu()))+str('_')+str(int(round(adv_pred_prob[0])))+str('_')+str(origProb_in_adv)+str('.png')
#   save_name = str(i)
#   save_path = os.path.join("as",save_name)

#   imsave(save_path,adversarial.transpose((1,2,0)))         # saves in 0-255 format

import numpy as np
adversarial = np.array(adversarial)
import matplotlib.pyplot as plt

for i in range(adversarial.shape[0]):
  plt.imsave('./as/'+str(i)+'.png',adversarial[i].transpose((1,2,0)))

import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.imshow(adversarial[0].transpose((1,2,0)))
