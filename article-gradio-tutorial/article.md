
# Gradio: Simplifying GUI Creation for Machine Learning

## Abstract

The integration of Machine Learning models into user-friendly applications plays an important role in enhancing usability and accessibility. The creation of a Graphical User Interface is a major step into deploying the models, because it allows interaction and data input.

Gradio is an open-source package that bridges this gap for Python Developers. This paper explores the functionalities of Gradio, demonstrating how it enables the construction of GUIs with minimal coding effort. It answers the question: “How can one easily create a GUI for Machine Learning models without extensive application development background?”

Through specific examples, this article aims to demonstrate the straightforward process of building GUIs with Gradio.

**Keywords:** Application development, Gradio, Graphical User Interface (GUI), Machine Learning, Model deployment, Python, User-friendly applications

## Introduction

When working with Machine Learning models, there is a variety of models and options to choose from. Depending on the data and the goal one wants to achieve, the user might choose a model for classification or regression, one oriented on Natural Language Processing or Computer Vision. Yet, one more important step that ensures the usability of models is the creation of a Graphical User Interface.

A graphical user interface (GUI) is a digital interface in which a user interacts with graphical components such as icons, buttons, and menus. Such an interface can be obtained by creating a computer, mobile, or web application which would include the Machine Learning model. However such products require knowing additional programming languages and frameworks. To address this challenge and to ensure that individuals who code in Python can easily construct a GUI, Gradio has been developed.

## What is Gradio?

Gradio is an open-source Python package that allows programmers to quickly build a demo or web application for their machine learning model, API, or any arbitrary Python function. One of the most exciting parts is that they can then share a link to their demo or web application in just a few seconds using Gradio’s built-in sharing features¹.

![](https://miro.medium.com/v2/resize:fit:1050/0*iXS3S4G8e-V-3r7T)

**Fig.1**  Gradio (Source:  [https://pypi.org/project/gradio/](https://pypi.org/project/gradio/))

## **Why choose Gradio?**

Gradio supports various applications, from showcasing machine learning models to debugging and deploying data science workflows. It integrates with the Python ecosystem, providing real-time feedback and becoming an ideal choice for Python developers seeking to enhance their projects with user-friendly GUIs.

## **Get started**

Note: For the tutorial, I will use Google Colab.

**Install Gradio**

Create a new code block and run:

    !pip install gradio

Next, import Gradio:

    import gradio as gr

Now we are ready to proceed further!

**How to create your first interfaces?**

Gradio offers the flexibility to handle different types of input and output, including text, images, audio, and video.

For our initial application, we will use the text functionality. We will create a new code block where we will write:

    def greet(name):  
        return "Hello, "+ name + "!"  
      
    iface = gr.Interface(  
        fn=greet,  
        inputs=gr.Textbox(info="Input your name", placeholder="Name", label="Name"),  
        outputs=gr.Textbox(label="Message"),  
    )  
      
    iface.launch()

The  `gr.Interface`  method is used to create a user interface for the  `greet`  function. It accepts several parameters: the function itself (`greet`), the type of input elements to present (`gr.Textbox`  in our scenario, whose value is sent as a parameter to the  `greet`  function), and the type of output to display (`gr.Textbox`  here, which displays the value returned by the  `greet`  function).

The final product should look like this:

![](https://miro.medium.com/v2/resize:fit:840/1*8FJc53PsDRuUfgPokAqGwg.gif)

**Fig.2**  The output of the greet function

Next, we will create another application that takes input from the user and returns a sentence based on the input.

    def build_sentence(season, weather, temperature, enjoy_weather):  
        sentence = f"In {season}, with {weather} weather and {temperature} degrees Celsius outside, you {'enjoy' if enjoy_weather == 'Yes' else 'do not enjoy'} the weather."  
        return sentence  
      
    iface = gr.Interface(  
        fn=build_sentence,  
        inputs=[  
            gr.Dropdown(choices=['Winter', 'Spring', 'Summer', 'Autumn'], info="Select the season", label="Season"),  
            gr.Dropdown(choices=['freezing', 'warm', 'hot', 'cold'], info="Select the weather", label="Weather"),  
            gr.Slider(minimum=-30, maximum=60, step=2, info="How many degrees Celsius are outside?", label="Temperature"),  
            gr.Radio(['Yes', 'No'], info="Do you enjoy the weather?", label="Emotion")  
        ],  
        outputs=gr.Textbox(label="Sentence")  
    )  
      
    iface.launch()

In this code, we create an interface for a function named  `build_sentence`  that constructs a sentence about weather conditions based on input parameters such as season, weather type, temperature, and whether the user enjoys the weather or not.

The interface includes dropdown menus,  `gr.Dropdown`, for selecting the season (Winter, Spring, Summer, Autumn) and the weather type (freezing, warm, hot, cold). Additionally, a slider,  `gr.Slider`, allows users to specify the temperature in degrees Celsius, ranging from -30 to 60 in steps of 2. This interface also provides radio buttons,  `gr.Radio`, to indicate whether the user enjoys the weather (Yes or No).

All these inputs are sent as parameters to the  `build_sentence`  function which creates a string that is returned to the interface and displayed as output.

The final product looks like this:

![](https://miro.medium.com/v2/resize:fit:840/1*itNsX6T952vvo8aBCSTirA.gif)

**Fig.3**  The output of the make_sentence function

For more examples, you can refer to the documentation of Gradio, available  [here](https://www.gradio.app/guides/quickstart).

**How to use Gradio for Machine Learning?**

As mentioned above, Gradio is a great tool to create GUI for Machine Learning tools. To exemplify this, let’s work on a real dataset.

For this project, I will use the Mushroom Classification dataset².

Import the necessary libraries and load the dataset:

    # Import libraries  
    import numpy as np  
    import pandas as pd  
    import matplotlib.pyplot as plt  
    import seaborn  as sns  
    import gradio as gr  
    import pickle

    # Load dataset  
    dataset = pd.read_csv('/content/mushrooms.csv')  
    dataset.head()

![](https://miro.medium.com/v2/resize:fit:1050/1*xV35YcBWWHef9Srk_O5b3A.png)

**Fig.4**  The head of the dataset

The dataset consists of 23 columns:

-   **class:** edible = e, poisonous = p;
-   **cap-shape:**  bell = b, conical = c, convex = x, flat = f, knobbed = k, sunken = s;
-   **cap-surface:**  fibrous = f, grooves = g, scaly = y, smooth = s;
-   **cap-color:**  brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y;
-   **bruises:**  bruises = t, no = f;
-   **odor:**  almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s;
-   **gill-attachment:**  attached = a, descending = d, free = f, notched = n;
-   **gill-spacing:**  close = c, crowded = w, distant = d;
-   **gill-size:** broad = b, narrow = n;
-   **gill-color:**  black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w, yellow = y;
-   **stalk-shape:** enlarging = e, tapering = t;
-   **stalk-root:**  bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?;
-   **stalk-surface-above-ring:**  fibrous = f, scaly = y, silky = k, smooth = s;
-   **stalk-surface-below-ring:**  fibrous = f, scaly = y, silky = k, smooth = s;
-   **stalk-color-above-ring:** brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y;
-   **stalk-color-below-ring:** brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y;
-   **veil-type:**  partial = p, universal = u;
-   **veil-color:**  brown = n, orange = o, white = w, yellow = y;
-   **ring-number:**  none = n, one = o, two = t;
-   **ring-type:** cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z;
-   **spore-print-color:** black = k, brown = n, buff = b, chocolate = h, green = r, orange = o, purple = u, white = w, yellow = y;
-   **population:** abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y;
-   **habitat:** grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d.

I am interested in predicting whether a mushroom is edible or not.

I will skip some steps such as checking for NaN values or duplicate rows as this is outside the scope of this paper, but the full code is available on GitHub³.

I can get the statistical summary of this dataset using:

    dataset.describe()

![](https://miro.medium.com/v2/resize:fit:1050/1*vhfFn47M51ji_Wn2vTZraA.png)

**Fig.5**  Statistical summary of the dataset

However, for large datasets, this is hard to read. We can improve this by creating an interface in Gradio that allows analyzing each column separately.

    # Define a function to generate summary statistics for a specified column  
    def summary_statistics(column_name):  
        summary = dataset[column_name].describe()  # Generate descriptive statistics for the selected column  
        return summary.to_dict()  # Convert the pandas Series to a dictionary and return it  

  

    # Get a list of column names from the dataset  
    column_names = dataset.columns.tolist()  

  

    # Set up a Gradio interface  
    iface = gr.Interface(  
        fn=summary_statistics,  # Function to call when inputs change  
        inputs=gr.Dropdown(choices=column_names, label="Column", info="Select Column"),  # Dropdown to select a column  
        outputs=gr.JSON(label="Summary Statistics"),  # Output field to display summary statistics as JSON  
        live=True  # Allow live updates while selecting different columns  
    )  

  

    # Launch the Gradio interface  
    iface.launch()

This code creates a tool to analyze data columns interactively. It defines a function  `summary_statistic`  that computes and displays descriptive statistics for a selected column in a dataset. Using  `inputs`, it sets up a dropdown menu where users can choose a column to analyze. When selected, the function computes statistics and displays them in a user-friendly JSON format. The interface looks like this:

![](https://miro.medium.com/v2/resize:fit:1050/1*g9wjTgVYLOh_U147XphGrg.png)

**Fig.6**  Statistical summary of each column using Gradio

For the univariate analysis, we can create for each column individually a plot. For example, we will create a histogram for the class column:

    sns.histplot(dataset['class'])  
    plt.xlabel("Class")  
    plt.ylabel("Count")  
    plt.title("Distribution of Mushrooms by Class")  
    plt.show()

![](https://miro.medium.com/v2/resize:fit:1050/1*YGLssnxoRF3SyK5wkKxJ8w.png)

**Fig.7**  Distribution of the mushrooms by class

But it would take a bit of time to plot individually for each column such a histogram. Gradio can help us to create an interface to visualize each column in a more time-efficient way.

    # Define a function to plot a histogram for a specified column  
    def plot_column(column_name):  
        plt.figure(figsize=(10, 6)) # Set up the figure size  
        sns.histplot(dataset[column_name], kde=False) # Plot the histogram using seaborn  
        plt.title(f'Distribution of Mushrooms by {column_name}') # Set the plot title  
        plt.xlabel(column_name) # Set the label for the x-axis  
        plt.ylabel('Count') # Set the label for the y-axis  
        plt.xticks(rotation=90) # Rotate x-axis labels for better readability  
        plt.tight_layout() # Ensure the plot layout is tight  
        return plt.gcf() # Return the current figure  

  

    # Get a list of column names from the dataset  
    column_names = dataset.columns.tolist()  
      
    # Set up a Gradio interface  
    iface = gr.Interface(  
        fn=plot_column, # Function to call when inputs change  
        inputs=gr.Dropdown(choices=column_names, label="Select Column"),  # Dropdown to select a column  
        outputs=gr.Plot(), # Output field to display the plot  
        live=True # Allow live updates while selecting different columns  
    )  
      
    # Launch the Gradio interface  
    iface.launch()

This code defines a function  `plot_column`  that uses seaborn (`sns`) to plot histograms based on the selected column chosen as input at  `inputs`. The Gradio interface allows users to choose a column from a dropdown menu and see its histogram instantly. The final result looks like this:

![](https://miro.medium.com/v2/resize:fit:1050/1*kdEwUduRDoefyNvUuLrioQ.png)

**Fig.8**  Plotting histograms using Gradio

![](https://miro.medium.com/v2/resize:fit:1050/1*WHlS-AjVioLUk4wRq40M_w.png)

**Fig.9**  Plotting histograms using Gradio

For the multivariate analysis, we can also add a new input for the hue, which will help us visualize better specific characteristics of this dataset.

    # Define a function to plot a count plot with hue  
    def plot_count(column1, hue_column):  
        plt.figure(figsize=(12, 6)) # Set up the figure size  
        sns.countplot(x=dataset[column1], hue=dataset[hue_column]) # Plot countplot using seaborn  
        plt.title(f'Count Plot of {column1} with Hue {hue_column}') # Set the plot title  
        plt.xticks(rotation=45) # Rotate x-axis labels for better readability  
        plt.tight_layout() # Ensure the plot layout is tight  
        return plt.gcf() # Return the current figure  

  

    # Get a list of column names from the dataset  
    column_names = dataset.columns.tolist()  
      
    # Set up a Gradio interface  
    iface = gr.Interface(  
        fn=plot_count, # Function to call when inputs change  
        inputs=[gr.Dropdown(choices=column_names, label="Column 1"), # Dropdown to select the first column  
                gr.Dropdown(choices=column_names, label="Hue Column")], # Dropdown to select the hue column  
        outputs=gr.Plot(), # Output field to display the plot  
        live=True # Allow live updates while selecting different columns  
    )  

  

    # Launch the Gradio interface  
    iface.launch()

This code sets up an interactive tool to plot count plots with a specified column (`Column 1`) on the x-axis and another column (`Hue Column`) as a distinguishing factor (hue) using seaborn (`sns`). Users select columns from dropdown menus in the Gradio interface to visualize how categories in  `Column 1`  are distributed based on categories in  `Hue Column`. The interface has the following aspect:

![](https://miro.medium.com/v2/resize:fit:1050/1*AMb3yJEYCkJrWCombB7KtQ.png)

**Fig.10**  Plotting countplots using Gradio

![](https://miro.medium.com/v2/resize:fit:1050/1*TI4cX3tyeNUDQPmMOj_WqA.png)

**Fig.11**  Plotting countplots using Gradio

Now that the visualization part is over, it’s time to build our model.

We will prepare the data by splitting it into the X and y set, where X contains all columns except ‘class’ and y contains only the target column — ‘class’.

    # Separate features (X) and target variable (y) from the dataset  
    X = dataset.drop(['class'], axis=1) # X contains all columns except 'class'  
    y = dataset['class'] # y contains only the 'class' column
    
    Since the dataset contains categorical data, we will apply One-Hot Encoding on X and LabelEncoder() on y.
    
    # Convert categorical variables into dummy variables  
    X = pd.get_dummies(X)  
    X.head()

    from sklearn.preprocessing import LabelEncoder   
    encoder = LabelEncoder()  # Create an instance of LabelEncoder  
    y = encoder.fit_transform(y) # Transform the target variable (y) into numerical labels

We will save the column names of data frame X into a file. These names will serve as reference points during the interface creation process.

    with open('column_names.txt', 'w') as file:  
        file.write('\n'.join(X.columns))

Split the dataset into the Train set and Test set.

    from sklearn.model_selection import train_test_split    

    # Split the dataset into training and testing sets  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

To predict the values, we will use the Logistic Regression model from the Scikit-learn library.

    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression()  # Create an instance of LogisticRegression  
    # Train the Logistic Regression model  
    model.fit(X_train, y_train)

To be able to use the interface without fitting the model and the encoder each time, we will save them into pickle files and we will import them when necessary.

    pickle_out = open("classifier.pkl", mode="wb") # Open a file named "classifier.pkl" in write-binary mode ('wb')  
    pickle.dump(model, pickle_out) # Serialize (pickle) the trained model (model) and write it to the file  
    pickle_out.close() # Close the pickle file
    
    pickle_out = open("encoder.pkl", mode="wb") # Open a file named "encoder.pkl" in write-binary mode ('wb')  
    pickle.dump(encoder, pickle_out) # Serialize (pickle) the encoder object (encoder) and write it to the file  
    pickle_out.close() # Close the pickle file

Now we have all the necessary tools to build our interface.

Load all the necessary files and define the columns:

    # Load the model  
    with open("/content/classifier.pkl", "rb") as pickle_in:  
        model = pickle.load(pickle_in)  
      

    # Load the encoder  
    with open("/content/encoder.pkl", "rb") as pickle_in:  
        encoder = pickle.load(pickle_in)  
      
    # Load dataset  
    dataset = pd.read_csv('/content/mushrooms.csv')  
      
    # Define column names from the original dataset for the interface  
    columns = dataset.columns.drop('class').tolist()  
      
    # Define column names based on the one-hot encoded dataset  
    with open('/content/column_names.txt', 'r') as file:  
        columns_fit = file.read().splitlines()

The script first loads the pre-trained machine learning model and the encoder, which were saved in the`classifier.pkl`  and  `encoder.pkl`  files respectively. It then loads the mushroom dataset into a Pandas DataFrame and defines column names based on both the original dataset and the one-hot encoded format used during training, retrieved from the  `column_names.txt`  file.

    # Function to preprocess input data  
    def preprocess_input(input_data):  
        input_dict = {col: 0 for col in columns_fit}     # Initialize an input dictionary with all columns set to 0  
        index = 0  
        for element in input_data: # Iterate through each element in input_data and set corresponding keys to 1 in input_dict  
            input_dict[f"{columns[index]}_{element}"] = 1  
            index += 1  
        return pd.DataFrame([input_dict]) # Create a DataFrame from the input dictionary  
      
    # Function to predict class  
    def predict_class(*inputs):  
        inputs_list = []  
        for input in inputs: # Gather all inputs into a list  
            inputs_list.append(input)  
        input_data = preprocess_input(inputs_list) # Preprocess the input data  
        prediction = model.predict(input_data) # Predict using the trained model  
        predicted_class = encoder.inverse_transform(prediction)[0] # Convert predicted label index back to the original class label  
        return predicted_class  

  

    # Define Gradio interface  
    iface = gr.Interface(  
        fn=predict_class, # Function to be executed when inputs are provided  
        inputs=[gr.Dropdown(label=col, choices=dataset[col].unique().tolist()) for col in columns], # Dropdowns for selecting input values  
        outputs=gr.Textbox(label="Predicted Class"), # Textbox to display the predicted class  
        title='\U0001F344 Mushroom Edibility Prediction', # Title of the Gradio interface  
        description='Predict if a mushroom is edible or poisonous based on its characteristics.' # Description of the Gradio interface  
    )  
      
    # Launch Gradio interface  
    iface.launch() 

To make predictions, the script includes a function  `preprocess_input`  that preprocesses user-selected mushroom characteristics received as input from  `inputs`  into a format compatible with the model and respects the one-hot encoded format on which the model was trained.

Another function,  `predict_class`, predicts whether the mushroom is edible or poisonous based on the data frame received from the  `preprocess_input`  function, using the loaded model, and provides a human-readable prediction using the  `inverse_transform()`  method of the encoder. This result is returned to the Gradio interface which displays it as output. The final result looks like this:

![](https://miro.medium.com/v2/resize:fit:1050/1*mzG84rLj6xHyRCIjNoPz0g.png)

**Conclusion**

In this project, we successfully implemented interactive interfaces using Gradio to enhance data handling and user interaction in Machine Learning. These interfaces streamline data management tasks, significantly improving accessibility and efficiency. Gradio’s intuitive features not only simplify complex processes but also highlight its capability to save time

[1]  [https://www.gradio.app/guides/quickstart](https://www.gradio.app/guides/quickstart)

[2]  [https://www.kaggle.com/datasets/uciml/mushroom-classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)

[3]  [https://github.com/MihaelaCatan04/Gradio-Tutorial.git](https://github.com/MihaelaCatan04/Gradio-Tutorial.git)



