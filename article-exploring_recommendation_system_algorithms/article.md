
## **Exploring Recommendation System Algorithms: From Classic to Cutting-Edge**

![](https://cdn-images-1.medium.com/max/2000/1*U84sFtkay34IqI_dOAkWDw.jpeg)

## Introduction

  In the past, people generally purchased products recommended to them by their friends or the people they trust. This is how people used to make purchasing decisions when there was doubt about a product. But since the advent of the Internet, we are so used to ordering online and streaming music and movies that we are constantly creating data in the back end. A recommendation engine uses that data and different algorithms to recommend the most relevant items to users. It initially captures the past behavior of a user, and then it recommends items for future purchase or use.

There are many different types of recommendation engines, the following are the most commonly used and widely recognized in various industries, and each of them is explored in this article.

*  **Content-based filtering**

*  **Collaborative-based filtering**

* **Deep learning recommendation system**

* **Graph Based recommendation system**

## Content-Based Filtering

 The content-based filtering method is a recommendation algorithm that suggests items similar to the ones the users have previously selected or shown interest in. It can recommend based on the actual content present in the item. For example, as shown in Figure 2, a new article is recommended based on the text present in the articles

![Figure 1-  Content-Based Filtering](https://cdn-images-1.medium.com/max/2000/1*a4hScSGe2G762WtX78i0xg.png)

Let’s take a peek at how Netflix suggests what you might want to watch. Imagine they have two special lists: one for what you like (profile) and one for all their shows (item). To match these lists, they use a simple trick called “cosine similarity.”, formula is shown in Figure 2.2. It’s like measuring how close two things are. They check how similar your likes are to the details of the shows. If your likes and a show’s details are similar, Netflix suggests it. This helps them recommend stuff you’re likely to enjoy, based on what you’ve liked before.

![Figure 2- Cosine Similarity formula](https://cdn-images-1.medium.com/max/2000/0*SFeeQ43mnIehPPnx)

The major **downside** to this recommendation engine is that all suggestions fall into the same category, and it becomes somewhat monotonous. As the suggestions are based on what the user has seen or liked, we’ll never get new recommendations that the user has not explored in the past.

## Collaborative-Based filtering

 In collaborative-based filtering recommendation engines, a user-to-user similarity is also considered, along with item similarities, to address some of the drawbacks of content-based filtering. Simply put, a collaborative filtering system recommends an item to user A based on the interests of a similar user B. Figure 3.1 shows a simple working mechanism of collaborative-based filtering

![Figure 3.1-  Colaborative-Based filtering](https://cdn-images-1.medium.com/max/2000/1*AsbMg3LOMSsExCIL8XbeVQ.png)

The similarity between users can be calculated again by the technique mentioned earlier. A **user-item matrix** is created individually for each customer(Figure 3.2), which stores the user’s preference for an item.Taking the same example of Netflix’s recommendation engine, the user aspects like previously watched and liked titles, ratings provided (if any) by the user, frequently watched genres, and so on are stored and used to find similar users. Once these similar users are found, the engine recommends titles that the user has not yet watched but users with similar interests have watched and liked. This type of filtering is quite popular because it is only based on a user’s past behavior, and no additional input is required. It’s used by many major companies, including Amazon, Netflix, and American Express.

![Figure 3.2-  User-Item matrix](https://cdn-images-1.medium.com/max/2000/1*OL80PTi135OTquxATAKO-A.png)

## **There are two types of collaborative filtering algorithms:**

 1. **User-User Collaborative Filtering**: This method finds similarities between users and suggests items based on what similar users have liked in the past. It’s effective but demands extensive computations and time due to the need for user-pair information and similarity calculations. It’s resource-intensive, especially for large customer bases, unless a parallel system is in place.

 2. **Item-Item Collaborative Filtering**: Here, instead of similar users, the focus is on similar items. An item similarity matrix is created based on a user’s past selections, and recommendations come from this matrix of similar items. This approach is computationally more efficient because the item similarity matrix stays constant over time. This makes recommendations quicker for new customers.

![Figure 3.3 User-Based vs Item-Based fultering](https://cdn-images-1.medium.com/max/2000/1*5AgFWbQopyi8DPT9qj9cmw.png)

One of the **drawbacks** of this method happens when no ratings are provided for a particular item; then, it can’t be recommended. And reliable recommendations can be tough to get if a user has only rated a few items.

## DeepLearning Recommendation System

Various companies use deep neural networks (DNNs) to enhance the customer experience, especially if it’s unstructured data like images and text. If you’re new to the concept of deep learning, fear not! Here is a beginner-friendly article about [introduction to deep learning](https://medium.com/@Sumeet_Agrawal/introduction-to-deep-learning-4410c5fb5a9a), designed to provide you to the fundamentals without overwhelming you with technical jargon.The following are some types of deep learning–based recommender systems.

* Autoencoder based

* Restricted Boltzmann

![Figure 4.1 Deep Learning Recommender System](https://cdn-images-1.medium.com/max/2000/1*N_yNW8bsSZqDSAaQ1DtdPQ.png)

### Autoencoder Based Recommendation:

*What is it?* Autoencoders, a class of deep learning models, are designed to encode and decode data. In the context of recommendation systems, autoencoders learn to represent users and items in a compressed form, capturing intricate data patterns, let’s explore the elements captured in Figure 4.2. This involves training an autoencoder to reconstruct input data, such as user-item interactions, and using these learned representations for generating recommendations. Autoencoders **excel** at revealing hidden features and relationships that might elude traditional methods.



![Figure 4.2 Autoencoder based Recommendation System](https://cdn-images-1.medium.com/max/2800/1*6EHqmR3UOjhv1T6wOlh0xw.png)

Compared to traditional methods, autoencoders offer a more nuanced understanding of user preferences and item attributes. Their ability to capture latent patterns and hidden connections leads to improved recommendations that align with individual tastes. Unlike some conventional approaches, autoencoders adapt effectively to intricate and evolving user behaviors, resulting in higher recommendation accuracy.

Autoencoders, while powerful, can be computationally demanding. Training deep learning models requires substantial resources, and fine-tuning the architecture can be complex. In comparison to simpler methods, implementing autoencoders might involve a steeper learning curve. Additionally, for extremely sparse datasets, the performance of autoencoders can be hindered, making them less suitable for scenarios with minimal user-item interactions.



### Restricted Boltzmann Machines (RBMs) in Recommendations:

First, let’s start with the Boltzmann machine (BM). BM is a type of unsupervised neural network. Three distinct features characterize BM as illustrated in Figure 4.3.

* No output layers

* No direction between connection

* Each neuron is densely connected to each other, even between input nodes (visible nodes).

When applied to recommendation systems, RBMs are adept at uncovering latent patterns from user-item interactions. They model interactions as binary values, with hidden nodes representing latent factors. RBMs shine in collaborative filtering scenarios, even when faced with sparse data. Their prowess lies in dissecting intricate user preferences and item attributes.

![Figure 4.3 Boltzman Machine](https://cdn-images-1.medium.com/max/2000/1*Y5TK2jJt8nLWqQ8wshlDcg.png)

Compared to straightforward methods, RBMs **excel** in scenarios with sparse data. They adeptly uncover latent factors that contribute to user preferences, leading to enhanced recommendations. RBMs can capture intricate connections that simpler techniques might miss, resulting in more accurate personalized suggestions.

Training RBMs can be computationally intensive, especially when dealing with large datasets. Their probabilistic nature requires careful parameter tuning, and their implementation might demand a deeper understanding of probabilistic modeling compared to more straightforward techniques. For less complex data scenarios, the benefits of RBMs might not be fully realized, making them potentially overkill.

## Graph-Based Recommendation Systems

In graph-based recommendation systems, knowledge graph structures represent relationships between users and items. A knowledge graph is the structure of a network of interconnected datasets, enriched with semantics, an example is related in Figure 5.1. The graph structure, when visualized, has three primary components; nodes, edges, and labels. The edge of the link defines the relationship between two nodes/entities, where each node can be any object, user, item, place, and so forth. The underlying semantics provide an additional dynamic context to the defined relationships, enabling more complex decision-making

![Figure 5.1 User-item Interaction Graph](https://cdn-images-1.medium.com/max/2000/1*VINmk0C4SJCWzGP1-INQnA.png)

### Advantages of Graph-Based Recommendations

Graph-based recommendation systems excel in modeling diverse user-item interactions, providing personalized suggestions by considering network influence, addressing cold start challenges, and revealing unexpected recommendations.

### Limitations of Graph-Based Recommendations:

Despite their benefits, graph-based systems face scalability issues, struggle with sparse data, and rely heavily on data quality. They may also encounter challenges in recommending new items and involve complex model design.

## Conclusion.

Recommender systems have a rich history, originating in the late 1970s with systems like Grundy. Since their commercial introduction in the early 1990s, they have gained prominence due to their financial benefits and time-saving advantages. Notably, Netflix’s recommendation engine stands as a prominent example of their impact. The increasing demand for reliable recommender systems across domains underscores the need for ongoing research and innovation. These systems effectively streamline decision-making, benefiting businesses and users alike. The mentioned book serves as a comprehensive guide to building recommender systems in Python, offering a spectrum of methods for developers and practitioners. Ultimately, this field’s growth relies on continuous exploration and deeper understanding, as highlighted by the book’s contribution to advancing recommendation engine concepts and implementations.

## Source:

[Applied Recommender Systems with Python](https://link.springer.com/book/10.1007/978-1-4842-8954-9#toc)


