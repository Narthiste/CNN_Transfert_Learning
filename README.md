# CNN_Transfert_Learning

Le transfert learning, utilisé en Deep Learning, sert principalement à gagner du temps de calcul en gardant l'architecture d'un modèle précédent et en l'appliquant à un nouveau modèle. Par exemple, une architecture CNN dédiée à la reconnaissance de voitures pourra être utilisée pour la reconnaissance de camions. 

## 1. Qu'est ce que le Transfert Learning? 

On peut distinguer trois types de Transfer Learning:

**1) Apprentissage par transfert inductif (Inductive Transfer Learning)**

**2) Apprentissage par transfert non supervisé (Unsupervised Transfer Learning)**

**3) Apprentissage par transfert transductif (Transductive Transfer Learning)**

On peut distinguer deux types de stratégies d'emploi de Transfer Learning:

![image](https://user-images.githubusercontent.com/95342035/160565223-8cd87081-cecc-4485-8c76-6484d68e0a90.png)

**1) Utilisation de modèles pré-entraînés comme extracteurs de features**

On réutilise simplement un réseau pré-entraîné sans sa couche finale. 

**2) Ajustement de modèles pré-trainés:**

Technique plus complexe: on remplace la dernière couche, mais on modifie également d'autres hyperparamètres. 

Pour la reconnaissance d'images, plusieurs modèles sont retenus, notamment VGG-16 et VGG-19

## 2. Introduction à VGG16 et VGG19

### 2.1 VGG16

![image](https://user-images.githubusercontent.com/95342035/160566489-02239abe-4b9f-41d6-a3bb-4726df12cd63.png)

![image](https://user-images.githubusercontent.com/95342035/160568131-79d60c2b-5699-4a14-94b4-8ae72bf48c2b.png)

### 2.2 VGG19

L'architecture VGG19 est similiaire à celle du VGG16, seulement on y ajoute trois couches de convolutions supplémentaires. 

![image](https://user-images.githubusercontent.com/95342035/160567091-00c6595d-463e-4df4-80b5-02087c75240c.png)

*Sources pour la réalisation du readme:*
- https://datascientest.com/transfer-learning 
- https://towardsdatascience.com/inductive-vs-transductive-learning-e608e786f7d
- https://www.kaggle.com/code/shivamb/cnn-architectures-vgg-resnet-inception-tl avec implémentation des modèles sous Keras. 
- https://neurohive.io/en/popular-networks/vgg16/
- https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py modèle de VGG16 sur Tensorflow
