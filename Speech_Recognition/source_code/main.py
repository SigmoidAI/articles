

from sklearn import hmm

# Define the HMM model
model = hmm.MultinomialHMM(n_components=2, n_iter=100)

# Transition probabilities
model.startprob_ = [0.5, 0.5]  # Equal initial probabilities
model.transmat_ = [[0.7, 0.3],  # Transition from sunny to rainy and vice versa
                   [0.4, 0.6]]

# Emission probabilities (carrying an umbrella or not)
model.emissionprob_ = [[0.9, 0.1],  # High chance of carrying umbrella when rainy
                       [0.2, 0.8]]  # Low chance of carrying umbrella when sunny

# Generate training data
observations = [[0, 1, 0, 0, 1, 0]]  # 0 for no umbrella, 1 for umbrella

# Train the HMM
model.fit(observations)

# Predict hidden states for new observations
new_observations = [[0, 1, 0, 1, 1, 0]]
hidden_states = model.predict(new_observations)

# Print predicted hidden states
print("Predicted Hidden States:", hidden_states)
