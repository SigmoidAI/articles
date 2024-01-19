## Speech Recognition

![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/f7cb8f77-d19b-4b3e-b744-f014bf562a49)

Speech recognition is not just about the algorithms we apply to get a list of words from an audio file. It is a complex topic that includes several key ideas and in the following article I will cover the most important one.


## Key words:

Amplitude, Frequency, Vocal Cords, Phoneme.

- Amplitude is the maximum extent of a vibration or oscillation, measured from the position of equilibrium.
- Frequency represnts the number of times something happens within a particular period, or the fact of something happening often or a large number or times.
- The vocal cords (also called vocal folds) are two bands of smooth muscle tissue found in the larynx (voice box). The vocal cords vibrate and air passes through the cords from the lungs to produce the sound of your voice.
- A phoneme is any of the perceptually distinct units of sound in a specified language that distinguish one word from another, for example p, b, d, and t in the English words pad, pat, bad, and bat.

## *How do sounds appear?*
(The material played in the following video can be found below it)

https://github.com/Tudor-Gavriliuc/articles/assets/120340209/ace079b1-5faf-49a5-ace8-ae34c738d3e5

// Voice and speech are fundamental to human communication, produced by a sophisticated interplay of bodily functions. The unique sound of one's voice originates from the vocal folds in the larynx, or voice box, which vibrate when air from the lungs is forced through them. This process begins with breathing: the diaphragm contracts to pull air into the lungs, then relaxes to push it out, setting the stage for voice production. When speaking, this airflow causes the vocal folds to come together and vibrate, creating sound. In contrast, speech is the transformation of these sounds into distinct, recognizable patterns through the coordinated movement of muscles in the head, throat, chest, mouth, nasal cavity, and abdomen. Together, these intricate processes allow us to convey thoughts, feelings, and ideas, showcasing the complexity and importance of voice and speech in our daily lives. //
 
The primary source of speech sounds in most languages is the human vocal tract. It consists of the lungs, larynx (voice box), pharynx, oral cavity (mouth), and nasal cavity. Different speech sounds are produced by manipulating airflow, vocal cord vibrations, and the shape of the vocal tract.

![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/e90c018d-8cb5-428a-9708-1018c35703fa)


## *From analog to digital representation*

The first step in analog-to-digital conversion is sampling, where the continuous analog speech signal is converted into discrete samples taken at regular intervals. The sampling rate, such as 8 kHz or 16 kHz, determines how frequently these samples are taken, representing the original analog signal more accurately in the digital domain.

![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/0e2d6582-2d96-4fa2-bb53-c9ad0f36347c)

After sampling the analog signal, the next step is quantization. This involves converting each sampled value from the continuous amplitude range into a discrete digital value using binary numbers. The bit depth determines the number of bits used to represent each sample, with higher bit depths providing more precision but requiring more storage and resources. Common bit depths in speech recognition are 8, 16, and 24 bits. The digital values are then encoded into formats like PCM, used widely in speech recognition. The result is a digital signal composed of discrete samples, which can be stored or processed by speech recognition algorithms.
https://youtu.be/JARmZM4FH8Q

## *Fast Fourier Transform (FFT)*
(The material played in the following video can be found below it)

Speech signals are time-varying, and the information about the sounds and phonemes is often encoded in the frequency domain. FFT allows us to transform the speech signal from the time domain to the frequency domain, revealing the specific frequencies and their magnitudes present in the signal.

https://github.com/Tudor-Gavriliuc/articles/assets/120340209/2ffbe853-b0d8-4d0d-8ccb-a44526fb4f4f

// This song guessing game utilizes the mathematical concept of the Fourier transform to analyze and reconstruct songs. A WAV file of a song, essentially a 1D array representing speaker displacement over time, serves as the starting point. The Fourier transform then decomposes this wave into its constituent sine waves, each with its own frequency and amplitude, allowing visualization of the song's frequency spectrum. For discrete audio files, a Discrete Fourier transform is used, resulting in a bar chart-like frequency spectrum. The game progressively reveals the song by sorting these frequencies by amplitude and gradually reintroducing them, starting with the most influential ones. This process reshapes the wave closer to the original song with each step, making the song more recognizable. //

## *Feature Extraction*

The crucial speech recognition process of feature extraction includes breaking down the digital voice signal into a collection of significant and representative features. These characteristics act as inputs to voice recognition algorithms, enabling them to efficiently recognize and distinguish between various spoken words or phonemes. The following steps are commonly involved in the feature extraction process:

Preprocessing: The digital voice signal may go through some preprocessing stages to improve its quality and reduce noise before feature extraction. Filtering, normalizing, and silence elimination are common preprocessing methods.

Frame blocking is the process of breaking up a continuous digital speech output into smaller, overlapping frames. A brief segment of the signal, often lasting 20 to 30 milliseconds, makes up each frame. When frames overlap, the flow of information is maintained.

[Speech Recognition — Feature Extraction MFCC & PLP](https://jonathan-hui.medium.com/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9)

![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/1b80cdd5-e7cd-42f4-bab3-5fd1e8101208)

## *Acoustic Modeling*
**Input**
A speech signal.
**Output**
A recognized sequence of phones (the basic unit of speech).


[A Basic Introduction to Speech Recognition (Hidden Markov Model & Neural Networks)](https://www.youtube.com/watch?v=U0XtE4_QLXI) (Below you can find the material played in this video)

// Hannes and Rodolphe created a workshop on speech recognition, diving into the technology behind it without prior expertise. Their video, aimed at simplifying the concept, starts by explaining the conversion of human-produced analog sounds into digital data through microphones. This process involves capturing sound waves, chopping them into blocks, and digitizing them, visualizing them using spectrograms that display frequency, intensity, and time. The core of speech recognition is decoding these digital sounds into meaningful language, focusing on the smallest sound units called phonemes and their variations, allophones. The video covers two main techniques used in speech recognition software: the Hidden Markov Model, which reconstructs speech by analyzing phoneme sequences using statistical probabilities, and Neural Networks, which mimic brain function and adapt over time to recognize speech patterns, including unique voices and emotions. Combining these two techniques forms a hybrid model, leveraging their respective strengths to enhance speech recognition. //

In voice recognition systems, acoustic models are a particular kind of machine-learning model. The model is trained to identify the acoustic features of human speech and provide an appropriate text transcription. Huge datasets of spoken words, phrases, and sentences are often used to train acoustic models. To “map” the sounds of human speech, these datasets are used.

![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/c7036b6b-e697-4411-b291-17c2e3294c69)

## *Hidden Markov Model ( HMM)*

A Hidden Markov Model (HMM) is a mathematical model that deals with sequences of observations, where each observation is tied to a hidden state. The model uses probabilities to transition between states and emit observations. It’s commonly used in tasks like speech recognition or DNA sequence analysis. The HMM’s goal is to find the best sequence of hidden states that explains the observed data. It learns from training data and is used for decoding sequences in various applications.

![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/01e14069-c487-4cd3-a31b-e6af7c60c27c)

## *Hidden Markov Model ( HMM)*

```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
  
# Define the state space
states = ["Word0", "Word1", "Word2", "Word3"]
n_states = len(states)
  
# Define the observation space
observations = ["Loud", "Soft"]
n_observations = len(observations)
  
# Define the initial state distribution
start_probability = np.array([0.8, 0.1, 0.1, 0.0])
  
+transition_probability = np.array([[0.7, 0.2, 0.1, 0.0],
                                    [0.0, 0.6, 0.4, 0.0],
                                    [0.0, 0.0, 0.6, 0.4],
                                    [0.0, 0.0, 0.0, 1.0]])
  
# Define the observation likelihoods
emission_probability = np.array([[0.7, 0.3],
                                  [0.4, 0.6],
                                  [0.6, 0.4],
                                  [0.3, 0.7]])
  
# Fit the model
model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability
  
# Define the sequence of observations
observations_sequence = np.array([0, 0, 0, 0, 1, 1, 0, 1]).reshape(-1, 1)
  
# Predict the most likely hidden states
hidden_states = model.predict(observations_sequence)
print("Most likely hidden states:", hidden_states)
  
# Plot the results
sns.set_style("darkgrid")
plt.plot(hidden_states, '-o', label="Hidden State")
plt.legend()
plt.show()

```
![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/b5fedd8b-593d-46b3-8307-005a8719219a)
- In our case (“Speech recognition”) observation is the audio file that we take as input, and the states are the [phonemes,words] we need to find out.

## *Language Modeling*

- A language model is a probabilistic model that assigns probabilities to any sequence of words.
- A good language model has to assign high probabilities to well performed sentences.
- 
  **Statistical Approach (N-grams):**

- Traditional language models, like N-gram models, estimate the likelihood of a word based on its previous N-1 words.
- The model calculates probabilities of word sequences based on the frequency of co-occurrences in the training data.
- For example, a bigram model predicts the next word based on the previous word, while a trigram model considers the previous two words.

  ![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/597f009c-eaed-4e4c-9b11-1eb52fdcbd4b)

## *Decoding*
- Combine the outputs of the acoustic model and the language model to decode the most likely transcription for the given audio.
- Apply algorithms like Viterbi or beam search to find the best sequence of words that match the audio features.

  *The code below shows how the Viterbi algorithm works in the context of speech recognition.*

```
# Define possible observations and states
obs = ("sound1", "sound2", "sound3")
states = ("Word1", "Word2")

# Define initial state probabilities
start_p = {"Word1": 0.6, "Word2": 0.4}

# Define transition probabilities between states
trans_p = {
    "Word1": {"Word1": 0.7, "Word2": 0.3},
    "Word2": {"Word1": 0.4, "Word2": 0.6},
}

# Define emission probabilities of observations from states
emit_p = {
    "Word1": {"sound1": 0.5, "sound2": 0.4, "sound3": 0.1},
    "Word2": {"sound1": 0.1, "sound2": 0.3, "sound3": 0.6},
}

# Define the Viterbi algorithm function
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # Initialize Viterbi matrix

    # Initialization for t = 0
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st] * emit_p[st][obs[t]]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st] * emit_p[st][obs[t]]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    # Print the Viterbi matrix
    for line in dptable(V):
        print(line)

    # Backtrack to find the most likely sequence of states
    opt = []
    max_prob = 0.0
    best_st = None
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    # Print the result
    print("The steps of states are " + " ".join(opt) + " with the highest probability of %s" % max_prob)

# Define a function to print the Viterbi matrix as a table
def dptable(V):
    yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state]["prob"]) for v in V)

# Call the Viterbi function with defined parameters
viterbi(obs, states, start_p, trans_p, emit_p)
```
![image](https://github.com/Tudor-Gavriliuc/articles/assets/120340209/2905115f-f9f9-472f-82df-7887d43b3706)

## *A simple approach of speech recognition*

```
# Import required libraries
from gtts import gTTS  # For text-to-speech synthesis
import os
from os import path
from deep_translator import GoogleTranslator  # For text translation
import speech_recognition as sp  # For speech recognition

# Define a class named 'pevervodcik'
class pevervodcik:
    # Constructor with 'language' as a parameter
    def __init__(self, language):
        self.language = language

    # Method to capture audio input using a microphone
    def get_audio(self):
        mic = sp.Microphone(device_index=0)  # Microphone initialization
        r = sp.Recognizer()  # Speech recognizer initialization
        with mic as source:
            print("Spuneti acum")  # Prompt for user to speak
            audio = r.listen(source)  # Listen for audio input
        try:
            c = r.recognize_google(audio, language="ro-RO")  # Recognize spoken text using Google's API
        except:
            print("Something go wrong")  # Error message if recognition fails
        return c  # Return recognized text

    # Method to translate the provided text
    def translate(self, c):
        # Translate the text using GoogleTranslator with the target language specified in the constructor
        translated = GoogleTranslator(source='auto', target=self.language).translate(c)
        print(translated)  # Print translated text
        return translated  # Return translated text

    # Method to convert translated text to voice and play it
    def voice(self, trans):
        # Convert translated text to speech using gTTS (Google Text-to-Speech)
        myobj = gTTS(text=trans, lang=self.language, slow=False)
        myobj.save("welcome.mp3")  # Save the generated speech as an audio file
        os.system("welcome.mp3")  # Play the saved audio file

# Create an instance of 'pevervodcik' with target language 'en' (English)
Ion = pevervodcik('en')
# Capture audio input from the user
c = Ion.get_audio()
# Translate the captured text
trans = Ion.translate(c)
# Convert the translated text to speech and play it
Ion.voice(trans)

# Note: Instead of using a microphone, the code can also be modified to read from an existing audio file.
```
In the code above we have created a simple translator based on speech recognition algorithm. In our class ,,perevodcik,, we have 3 methods.

First method is responsible for recognizing the speech and saving it as an audio file.

Second method is responsible for translating the audio file according to the required language and saving it in an audio new file using GoogleTranslator library.

The last method plays the audio file using gTTS library.

## Conclusion

In conclusion, speech recognition stands as a remarkable testament to the ever-evolving potential of human-technology symbiosis. Its transformative impact transcends mere convenience, delving deep into the realms of accessibility, communication, and innovation. What was once a futuristic concept has woven itself into the fabric of our daily lives, enriching the way we interact with devices, unlocking new horizons of productivity, and fostering inclusion.
