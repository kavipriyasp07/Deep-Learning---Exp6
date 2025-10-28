# EXPERIMENT 06: Implement a Simple LSTM using TensorFlow/Keras

## **DL- Developing a Deep Learning Model for NER using LSTM**

## **AIM**

To develop an LSTM-based model for recognizing the named entities in the text.

## **THEORY**

**Neural Network Model**

<img width="500" height="471" alt="image" src="https://github.com/user-attachments/assets/f1561c2b-7b24-4cc6-950b-53f4ed3b1977" />

## **DESIGN STEPS**

**STEP 1:** Data Preprocessing

  - Load the dataset (ner_dataset.csv) using pandas.

  - Fill missing values using forward fill (.ffill() method).

  - Extract unique words and tags from the dataset and create mappings (word2idx, tag2idx).

**STEP 2:** Sentence Grouping

  - Combine words, their POS tags, and entity tags into complete sentences using groupby("Sentence #").

  - Each sentence becomes a list of (word, POS, tag) tuples to preserve word-level tagging structure.

**STEP 3:** Token Indexing and Padding

  - Convert each word and tag into their corresponding integer indices using the mappings.

  - Apply padding (using Keras pad_sequences) to make all sequences equal in length (e.g., max_len = 50).

  - Split data into training and testing sets using train_test_split.

**STEP 4:** Model Construction

  - Define an Embedding layer to convert word indices into dense vectors.

  - Apply SpatialDropout1D for regularization.

  - Use a Bidirectional LSTM layer to capture contextual information from both directions.

  - Add a TimeDistributed Dense layer with a softmax activation to predict entity tags at each word position.

**STEP 5:** Model Compilation and Training

  - Compile the model with Adam optimizer and sparse_categorical_crossentropy loss.

  - Train the model for multiple epochs (e.g., 3) with the training data and validate using the test set.

**STEP 6:** Evaluation and Prediction

  - Plot training vs. validation accuracy and loss to monitor learning.

  - Predict tags for a sample sentence from the test set.
    
  - Compare the true tags and predicted tags word by word to evaluate model performance.

## **PROGRAM**


**Name:** KAVIPRIYA SP

**Register Number:** 2305002011

```python

import matplotlib.pyplot as plt, pandas as pd, numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers, Model

# Load + preprocess
data = pd.read_csv("ner_dataset.csv", encoding="latin1").ffill()  # ✅ replaces deprecated fillna(method='ffill')
print("Unique words:", data['Word'].nunique(), "| Unique tags:", data['Tag'].nunique())

words, tags = list(data['Word'].unique()) + ["ENDPAD"], list(data['Tag'].unique())
word2idx, tag2idx = {w:i+1 for i,w in enumerate(words)}, {t:i for i,t in enumerate(tags)}

# Group sentences safely
sents = data.groupby("Sentence #", group_keys=False).apply(
    lambda s:[(w,p,t) for w,p,t in zip(s.Word,s.POS,s.Tag)]
).tolist()

# Sequence preparation
max_len = 50
X = sequence.pad_sequences([[word2idx[w[0]] for w in s] for s in sents],
                           maxlen=max_len,padding="post",value=len(words)-1)
y = sequence.pad_sequences([[tag2idx[w[2]] for w in s] for s in sents],
                           maxlen=max_len,padding="post",value=tag2idx["O"])

# ✅ Convert labels to integer array
X, y = np.array(X, dtype="int32"), np.array(y, dtype="int32")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)

# Model
inp = layers.Input(shape=(max_len,))
x = layers.Embedding(len(words), 50, input_length=max_len)(inp)
x = layers.SpatialDropout1D(0.13)(x)
x = layers.Bidirectional(layers.LSTM(250, return_sequences=True, recurrent_dropout=0.13))(x)
out = layers.TimeDistributed(layers.Dense(len(tags), activation="softmax"))(x)

model = Model(inp, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(Xtr, ytr, validation_data=(Xte, yte), batch_size=45, epochs=3)

# Metrics plot
hist = pd.DataFrame(model.history.history)
hist[['accuracy','val_accuracy']].plot(); hist[['loss','val_loss']].plot()

# Sample prediction
i = 20
p = np.argmax(model.predict(np.array([Xte[i]])), axis=-1)[0]
print("{:15}{:5}\t{}".format("Word", "True", "Pred")); print("-"*30)
for w,t,pd_ in zip(Xte[i], yte[i], p):
    print("{:15}{}\t{}".format(words[w-1], tags[t], tags[pd_]))


````




## **OUTPUT**

## TRAINING LOSS, VALIDATION LOSS VS ITERATION PLOT:

<img width="565" height="418" alt="image" src="https://github.com/user-attachments/assets/aca3d86c-ed07-4aa7-a9f8-e1c3405c1d16" />

<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/22164c2d-981c-4a96-a6ed-eddb88682f22" />

## SAMPLE TEXT PREDICTION:

<img width="319" height="700" alt="image" src="https://github.com/user-attachments/assets/68785715-2788-4751-ab71-0523146b484d" />


## **RESULT**

Thus, an LSTM-based model for recognizing the named entities in the text is successfully developed.
