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

      Text -> turn into numbers -> build a model -> train the model to find patterns -> use patterns (make predictions)

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

## Exercises

- [ ] Rebuild, compile and train model_1, model_2 and model_5 using the [Keras Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) instead of the Functional API.
- [ ] Retrain the baseline model with 10% of the training data. How does perform compared to the Universal Sentence Encoder model with 10% of the training data?
- [ ] Try fine-tuning the TF Hub Universal Sentence Encoder model by setting training=True when instantiating it as a Keras layer.

```
# We can use this encoding layer in place of our text_vectorizer and embedding layer
sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=True) # turn training on to fine-tune the TensorFlow Hub model
```
- [ ] Retrain the best model you've got so far on the whole training set (no validation split). Then use this trained model to make predictions on the test dataset and format the predictions into the same format as the sample_submission.csv file from Kaggle (see the Files tab in Colab for what the sample_submission.csv file looks like). Once you've done this, [make a submission to the Kaggle competition](https://www.kaggle.com/c/nlp-getting-started/data), how did your model perform?
- [ ] Combine the ensemble predictions using the majority vote (mode), how does this perform compare to averaging the prediction probabilities of each model?
- [ ] Make a confusion matrix with the best performing model's predictions on the validation set and the validation ground truth labels.

## Resources
- [Natural Language Processing with TensorFlow by Mr D.Bourke](https://dev.mrdbourke.com/tensorflow-deep-learning/08_introduction_to_nlp_in_tensorflow/)
