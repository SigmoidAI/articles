# Document Vectorization with LTUVectorizer and LTUTransformer.
This article provides a fast introduction in LTUVectorizer and LTUTransformer modules from Molda library. You are going to learn how these algorithms can be implemented and how they work.

## Article Summary

Vectorization is a key method in Natural Language Processing (NLP) that enables computers to efficiently comprehend and interpret human language. Vectorization makes NLP systems more effective and precise by converting text input into numerical vectors that algorithms can process, understand, and extract valuable language-based information from.
A well-known vectorization formula is TF-IDF vectorization, but this article comes to show another vectorizer called LTU.
## LTU formula
LTU transformation is calculated using the following formula:
\omega _{ij} = \frac{(log(f_{ij}) + 1)log(\frac{N}{n_{j}})}{0.8 + 0.2 \times f_{j} \times \frac{j}{f_{j}}}
where f_{if} denotes the target term frequency in a particular document, f_{j} the total document frequency, n_{j} the number of documents containing the target term, N the total number of documets.

## Getting Started

Follow these steps to get started with LTUVectorizer and LTUTransformer:

1. **Prepare Your Environment**: Before you begin, make sure you have a virtual environment set up for your project. If not, create and activate a virtual environment:

    ```sh
    python -m venv myenv
    source myenv/bin/activate      # On Windows: .\myenv\Scripts\activate
    ```

2. **Install Requirements**: Inside your virtual environment, install the required packages:

    ```sh
    pip install molda numpy
    ```

3. **Import the methods**: Use the following code to import the module:

    ```python3
    from molda import LTUVectorizer
    ```
The difference between Vectorizer and Transformer is the input. The Transformer uses a count matrix as input while the Vectorizer some text.

4. **Get data**: To test the methods we need some sample documents. Here is an example:

    ```python3
document = [
    '''I love my dog
    and I like to pet him.''',
    'I love my cat!',
    "I like to pet my cat.",
    "I enjoy playing with both, cats and dogs."
]

    ```

5. **Instantiate the vectorizer**: With just 2 lines of code we get a sparse matrix from the document:
```python3
vectorizer = LTUVectorizer()
sparse_matrix = vectorizer.fit_transform(document)
```
The output is a LTU weighted sparse matrix that can be easily transformed in a simple matrix using numpy library:
```python3
print(sparse_matrix)

(0, 7) 1.4242701385551153
(0, 11)    1.122910936582831
(0, 8) 1.122910936582831
(0, 0) 1.122910936582831
(0, 4) 1.4242701385551153
(0, 10)    0.9090931800308315
(0, 9) 1.122910936582831
(1, 2) 1.122910936582831
(1, 10)    0.9090931800308315
(1, 9) 1.122910936582831
(2, 13)    1.4242701385551153
(2, 2) 1.122910936582831
(2, 11)    1.122910936582831
(2, 8) 1.122910936582831
(2, 10)    0.9090931800308315
(3, 5) 1.4242701385551153
(3, 3) 1.4242701385551153
(3, 1) 1.4242701385551153
(3, 14)    1.4242701385551153
(3, 12)    1.4242701385551153
(3, 6) 1.4242701385551153
(3, 0) 1.122910936582831

```
## Next Steps

Now that you've successfully created the sparse matrix, feel free to play with the parameters of the constructor. Also you can incorporate it in a pipeline with scikit-learn modules.

For more information you can visit the github page: https://github.com/SigmoidAI/molda or reach our community.
Happy coding!
