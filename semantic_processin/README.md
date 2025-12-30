
 # Welcome to Semantic Processing Module

## TOC:
- How to download files?
- What is where?

### How to download files?
![](images/image1.png)
Click on Code button and then click on Download ZIP
OR
Use `git clone https://github.com/ContentUpgrad/intro_to_neural_networks.git` command on your terminal if git is installed in your machine. 


### What is where?
The folder structure is given below:

![](images/image6.png)

As you can see there are three main folders when you log in:

1. **Distributional Semantics** This is where all the code files regarding distributional semantics sessions are kept
2. **Knowledge Graphs** This is where all the code files regarding Knowledge Graph session are kept
3. **Topic Modelling**This is where all the code files regarding Topic Modelling session are kept

When you click on any folder you will find the code and data folders as shown below:
![](images/image2.png)
You will find all the code files of the session in code folder and data folder will be empty. Please note that you need to follow the instructions given in the segment for downloading data files and keep it in the data folder manually.

#### Distributional Semantics
You will find the following files in the code folder of Distributional Semantics
![](images/image4.png)
The data files required can be found [here](https://drive.google.com/drive/u/0/folders/1KUnMvuufvo0yXS23EaI2EMNaq2lt5Ynh)

#### Knowledge Graphs
You will find the following files in the code folder of Distributional Semantics
![](images/image3.png)
There are no data files required for this session.

#### Topic Modelling
You will find the following files in the code folder of Distributional Semantics
![](images/image5.png)
The data files required can be found [here](https://drive.google.com/drive/u/0/folders/1umS1MgUXyra3KVF-6FsN8krHQ31lXhlX)



### Example Use Caes for the belwo topics:

### Knowledge Graphs (Lesk Algorithm)

1. To find the sense of a word in a particular sentence we use knowledge graphs.
2. To find the relation between two entities we use knowledge graphs.


### Distributional Semantics  (CBOW,Skip-gram are used to create word embeddings which later can be used achieve the below use cases)
1. To determine the reviews or rating based on user comments or reviews we use the distributional semantics techniques.
2. To do sentiment analysis of the tweets or any other social media comments we use distributional semantics techniques.


### Topic Modelling
1. To find the topics discussed in a set of documents we use topic modelling techniques.


### How do we senteniment analysis ?
1. Were we will have cropus of user reviews or comments on a product.
2. Where we will have a labbeled dataset(1 or 0) against each review or comment.
3. So when create a word embedding using word2Vec model or from pre-trained word embedding we will get the semantic relationship of each word in the review or comment. (eg : good, great, excellent will have similar semantic relationship)
4. Then we build a neural network model to classify the reviews or comments based on the semantic relationship of the words in it.
```
glove_model = Sequential([
    Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'),
    embedding_layer,
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
5. We do global average pooling because we create a create embedding for a single document eg:(The quick brown fox jumps over the lazy dog) we will get (1,9,128) shape after embedding layer. So to convert it to (1,128) we do global average pooling.
so by doing this independent of the length of the document or vocbabulary size we will always get a fixed size embedding vector for all documents.
Rather than creating a embedding vector we could simple simple pass the parse BoW vector or tf-idf vector to the neural network model but in that case the semantic relationship between the words will be lost.
6. Finally we will use the model to predict the sentiment of new reviews or comments.

### Understanding Embeddings

When an embedding vector has 128 dimensions, each dimension is a learned latent feature with no explicit human meaning; semantic information is distributed across all dimensions.

When the documents is purley taking about sports and we create 2 dimensions.
And lets assume x-axis as sports and y-axis as person.
cricket = (9,2) and sachin = (8,3)
So the distance between cricket and sachin will be very less as both are related to sports.
Similarly if we take football = (9,1) and obama = (1,9)
So the distance between football and obama will be very high as both are not related to each other
