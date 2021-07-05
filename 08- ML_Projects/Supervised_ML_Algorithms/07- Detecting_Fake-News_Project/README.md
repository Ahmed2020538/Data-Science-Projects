# Fake NEWS PROBLEM ::
****************************
  * Do you trust all the news you hear from social media?

  * All news are not real, right?

  * How will you detect fake news?

  * The answer is Python. By practicing this advanced python project of detecting fake news, you will easily make a difference between real and fake news.

  * Before moving ahead in this machine learning project, get aware of the terms related to it like fake news, tfidfvectorizer, PassiveAggressive Classifier.


# What is Fake News?
****************************
  * A type of yellow journalism, fake news encapsulates pieces of news that may be hoaxes and is generally spread through social media and other online media. This is often done to further or impose certain ideas and is often achieved with political agendas. Such news items may contain false and/or exaggerated claims, and may end up being viralized by algorithms, and users may end up in a filter bubble.

# What is a TfidfVectorizer?
********************************
  * TF (Term Frequency): The number of times a word appears in a document is its Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.

  * IDF (Inverse Document Frequency): Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus.

  * The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

# # What is a Support vector Classifier?
**************************************************

  * Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

  * The advantages of support vector machines are:

  * Effective in high dimensional spaces.

  * Still effective in cases where number of dimensions is greater than the number of samples.

  * Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

  * Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

  * The disadvantages of support vector machines include:

  * If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

  * SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

  * The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.

# What is a PassiveAggressiveClassifier?
******************************************
  * Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

  * Detecting Fake News with Python
To build a model to accurately classify a piece of news as REAL or FAKE.

  * About Detecting Fake News with Python
This advanced python project of detecting fake news deals with fake and real news. Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.

# The fake news Dataset
****************************************
  * The dataset we’ll use for this python project- we’ll call it news.csv. This dataset has a shape of 7796×4. The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE. The dataset takes up 29.2MB of space and you can download it here.
