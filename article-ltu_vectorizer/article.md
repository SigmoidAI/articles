# Document Vectorization with LTUVectorizer and LTUTransformer

This article provides a fast introduction to the LTUVectorizer and LTUTransformer modules from the Molda library. You will learn how these algorithms can be implemented and how they work.

## Article Summary

Vectorization is a key method in Natural Language Processing (NLP) that enables computers to efficiently comprehend and interpret human language. By converting text input into numerical vectors, vectorization makes NLP systems more effective and precise. This allows algorithms to process, understand, and extract valuable language-based information. A well-known vectorization formula is TF-IDF vectorization, but this article introduces another vectorizer called LTU.

## LTU Formula

LTU transformation is calculated using the following formula:

$$
\omega _{ij} = \frac{(\log(f_{ij}) + 1)\log\left(\frac{N}{n_{j}}\right)}{0.8 + 0.2 \times f_{j} \times \frac{j}{f_{j}}}
$$

where $f_{ij}$ denotes the target term frequency in a particular document, $f_{j}$ the total document frequency, $n_{j}$ the number of documents containing the target term, and $N$ the total number of documents.

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

3. **Import the Methods**: Use the following code to import the module:

    ```python3
    from molda import LTUVectorizer
    ```

   The difference between Vectorizer and Transformer is the input. The Transformer uses a count matrix as input, while the Vectorizer uses some text.

4. **Get Data**: To test the methods we need some sample documents. Here is an example:

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

6. **Parameters**: The main 3 parameters of the Vectorizer are stop_words, smooth_idf and sublinear_tf:

- stop_words : {'english'}, list, default=None

    This parameter will remove all the english stop words from the resulting tokens. Only english is supported.


```python3
vectorizer_stop_words = LTUVectorizer(stop_words="english")
sparse_matrix_stop_words = vectorizer_stop_words.fit_transform(document)
print(sparse_matrix_stop_words)

  (0, 7)	1.1159507448271522
  (0, 5)	1.1159507448271522
  (0, 2)	1.4154420178615919
  (0, 6)	1.1159507448271522
  (1, 0)	1.1159507448271522
  (1, 6)	1.1159507448271522
  (2, 0)	1.1159507448271522
  (2, 7)	1.1159507448271522
  (2, 5)	1.1159507448271522
  (3, 3)	1.4154420178615919
  (3, 1)	1.4154420178615919
  (3, 8)	1.4154420178615919
  (3, 4)	1.4154420178615919
```

- smooth_idf : bool, default=True

    Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.

```python3
vectorizer_smooth_idf = LTUVectorizer(smooth_idf=False)
sparse_matrix_smooth_idf = vectorizer_smooth_idf.fit_transform(document)
print(sparse_matrix_smooth_idf)

  (0, 7)	1.7735971602918106
  (0, 11)	1.258420201767527
  (0, 8)	1.258420201767527
  (0, 0)	1.258420201767527
  (0, 4)	1.7735971602918106
  (0, 10)	0.9570609997952424
  (0, 9)	1.258420201767527
  (1, 2)	1.258420201767527
  (1, 10)	0.9570609997952424
  (1, 9)	1.258420201767527
  (2, 13)	1.7735971602918106
  (2, 2)	1.258420201767527
  (2, 11)	1.258420201767527
  (2, 8)	1.258420201767527
  (2, 10)	0.9570609997952424
  (3, 5)	1.7735971602918106
  (3, 3)	1.7735971602918106
  (3, 1)	1.7735971602918106
  (3, 14)	1.7735971602918106
  (3, 12)	1.7735971602918106
  (3, 6)	1.7735971602918106
  (3, 0)	1.258420201767527
```

- sublinear_tf : bool, default=False

    Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)

```python3
vectorizer_sublinear_tf = LTUVectorizer(sublinear_tf=True)
sparse_matrix_sublinear_tf = vectorizer_sublinear_tf.fit_transform(document)
print(sparse_matrix_sublinear_tf)

  (0, 7)	1.3476661267766539
  (0, 11)	1.062515453813376
  (0, 8)	1.062515453813376
  (0, 0)	1.062515453813376
  (0, 4)	1.3476661267766539
  (0, 10)	0.8601978316094646
  (0, 9)	1.062515453813376
  (1, 2)	1.062515453813376
  (1, 10)	0.8601978316094646
  (1, 9)	1.062515453813376
  (2, 13)	1.3476661267766539
  (2, 2)	1.062515453813376
  (2, 11)	1.062515453813376
  (2, 8)	1.062515453813376
  (2, 10)	0.8601978316094646
  (3, 5)	1.3476661267766539
  (3, 3)	1.3476661267766539
  (3, 1)	1.3476661267766539
  (3, 14)	1.3476661267766539
  (3, 12)	1.3476661267766539
  (3, 6)	1.3476661267766539
  (3, 0)	1.062515453813376
```


## Next Steps

Now that you've successfully created the sparse matrix, feel free to play with the parameters of the constructor. You can also incorporate it in a pipeline with scikit-learn modules.

For more information, visit the GitHub page: [https://github.com/SigmoidAI/molda](https://github.com/SigmoidAI/molda) or reach out to our community.

Happy coding!

