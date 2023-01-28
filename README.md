# Introduction to NLP

Introduction to Natural Language Processing using Tensorflow

Natural Language Processing problems often referred to as Sequence problems (going from one sequence to another).

Natural Language is a broad term but can be considered it to cover any of the following:
- Text (such as that contained in an email, blog post, book, Tweet)
- Speech (a conversation you have with a doctor, voice commands you give to a smart speaker)

If you're building an email application, you might want to scan incoming emails to see if they're spam or not a spam (classification).

If you're trying to analyse customer feedback complaints, you might want to discover which section of your business they're for.

Both of these types of data are often referred to as sequences (a sentence is a sequence of words). So a common term you'll come across in NLP problem is called seq2seq, in other words, finding information in one sequence to produce another sequence (e.g converting a speech command to a sequence of text-based steps).

To get hands-on with NLP in tensorflow, we're going to practice the steps we've used previously but this time with text data:

***Text -> turn into numbers -> build a model -> train the model to find patterns -> use patterns (make predictions)***

> **Resource** : [A Simple Introduction to Natural Language Processing](https://becominghuman.ai/a-simple-introduction-to-natural-language-processing-ea66a1747b32)

## Things I've learned

- Downloading a text dataset
- Visualizing text data
- Converting text into numbers using tokenization
- Turning our tokenized text into an embedding
- Modelling a text dataset
  - Starting with a baseline (TF-IDF)
  - Building several deep learning text models
    - Dense, LSTM, GRU, Conv1D, Transfer learning
- Comparing the performance of each our models
- Combining our models into an ensemble
- Saving and loading a trained model
- Find the most wrong prediction
