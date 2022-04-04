# CNN_Transfert_Learning

Le transfert learning, est beaucoup utilisé en Deep Learning. En effet, on coviendra qu'il n'est pas judicieux d'entraîner des DNN "from scratch". L'idée est donc de reprendre un réseaux de neurones existant qui accomplit une tâche similaire à celui qu'on cherche à construire, puis de réutiliser les couches basses de ce dernier. C'est ce qu'on nomme *transfert learning*. 

Deux avantages: 
1) On accélère considérablement l'entraînement du modèle
2) Ce modèle requiert significativement moins de données.

![image](https://user-images.githubusercontent.com/95342035/161541143-3c9b3d03-47d6-41b9-b106-e48a13bda003.png)


Pour la reconnaissance d'images, plusieurs modèles sont retenus, notamment VGG-16 et VGG-19

## 2. Introduction à VGG16 et VGG19

### 2.1 VGG16

![image](https://user-images.githubusercontent.com/95342035/160566489-02239abe-4b9f-41d6-a3bb-4726df12cd63.png)

![image](https://user-images.githubusercontent.com/95342035/160568131-79d60c2b-5699-4a14-94b4-8ae72bf48c2b.png)

### 2.2 VGG19

L'architecture VGG19 est similiaire à celle du VGG16, seulement on y ajoute trois couches de convolutions supplémentaires. 

![image](https://user-images.githubusercontent.com/95342035/160567091-00c6595d-463e-4df4-80b5-02087c75240c.png)

## 3. Transfer Learning par la pratique: Reconnaissance d'image

### 3.1: Base de données, analyse et préparation 
*Pour des soucis de rapidité d'exécution, notre travail ne sera pas réalisé sur Jupyter Notebook, mais Google Collab, qui nous permet d'utiliser un GPU pour la phase de convolution*

Tout d'abord, on commence par importer nos bibliothèques qui nous servirons pour le brief. Quelques manipulations sont nécéssaires pour se relier au Google Drive. 

![image](https://user-images.githubusercontent.com/95342035/161544312-db740942-9bc4-42d3-8328-3a8404bef86e.png)

Dès lors que l'environnement est en place, on peut s'occuper du traitement des images. 

![image](https://user-images.githubusercontent.com/95342035/161544527-1ab03d11-53db-4ec0-8dc0-f07df5c82789.png)

![image](https://user-images.githubusercontent.com/95342035/161544619-bec5752c-4a52-4ba6-ae5b-567b6bd5ea1a.png)

![image](https://user-images.githubusercontent.com/95342035/161544694-ccf78205-0d77-41dc-8695-1b0b9e8c1796.png)

![image](https://user-images.githubusercontent.com/95342035/161544788-1ac3dbaf-1ce9-432b-94bc-60397c0adc38.png)

### 3.2: Architecture CNN sur Tensorflow

Avant de passer à la phase de convolution, on applique une "data augmentation" au jeu de données, tout simplement pour avoir plus de données à entraîner pour notre modèle. 

![image](https://user-images.githubusercontent.com/95342035/161544861-2fb8316a-2dba-4ba3-b03a-76ee28179f17.png)

![image](https://user-images.githubusercontent.com/95342035/161545233-1f58b88c-330b-4d5b-b653-98b39165708b.png)

*Configuration du modèle*

![image](https://user-images.githubusercontent.com/95342035/161545347-3c3768b1-6fc3-4ed7-aed9-332b3f15d7a8.png)

Notre modèle est donc le suivant, il peut être executé. 

history = model.fit(datagen.flow(X_train, y_train_categ), epochs = EPOCHS, callbacks= [model_checkpoint_callback], validation_data = (X_val, y_val_categ))

On trace nos courbes d'accuracy et d'erreur: 

![image](https://user-images.githubusercontent.com/95342035/161545687-83f8f89e-365d-448a-9ba6-33820a2aff20.png)

![image](https://user-images.githubusercontent.com/95342035/161545738-d7dfb6ba-233e-4dc1-bc09-c499f7c9e3c7.png)

Enfin, on prépare la matrice de confusion: 

![image](https://user-images.githubusercontent.com/95342035/161545815-0965e522-8de4-4ab6-8694-57d6b4441a13.png)

![image](https://user-images.githubusercontent.com/95342035/161546495-a0ccb4c2-5eb8-449a-bee4-f50b10f9053d.png)


...Et on l'affiche: 

![image](https://user-images.githubusercontent.com/95342035/161545939-d24b3b52-96dd-4778-a89d-7e0f4aa4e107.png)

Pour finir, on enregistre notre modèle au format ".h5" (1,4Go)

![image](https://user-images.githubusercontent.com/95342035/161546052-a4ff5af8-114d-4ed3-a0d8-4734b9ebe8dd.png)


### 3.3: Application

*J'ai passé le test par images et suis directement passé au test webcam*

Pour la partie application, le code de l'application est joint au github. 


*Sources pour la réalisation du readme:*
- Hands on Machine Learning, Aurélien Géron
- https://datascientest.com/transfer-learning 
- https://towardsdatascience.com/inductive-vs-transductive-learning-e608e786f7d
- https://www.kaggle.com/code/shivamb/cnn-architectures-vgg-resnet-inception-tl avec implémentation des modèles sous Keras. 
- https://neurohive.io/en/popular-networks/vgg16/
- https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py modèle de VGG16 sur Tensorflow
