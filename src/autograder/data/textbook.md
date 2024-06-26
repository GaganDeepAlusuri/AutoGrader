### PART I

**The Fundamentals of**

**Machine Learning**

**CHAPTER 1**

**The Machine Learning Landscape**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 1 in the final
release of the book.
```

When most people hear “Machine Learning,” they picture a robot: a dependable but‐
ler or a deadly Terminator depending on who you ask. But Machine Learning is not
just a futuristic fantasy, it’s already here. In fact, it has been around for decades in
some specialized applications, such as Optical Character Recognition (OCR). But the
first ML application that really became mainstream, improving the lives of hundreds
of millions of people, took over the world back in the 1990s: it was the spam filter.
Not exactly a self-aware Skynet, but it does technically qualify as Machine Learning
(it has actually learned so well that you seldom need to flag an email as spam any‐
more). It was followed by hundreds of ML applications that now quietly power hun‐
dreds of products and features that you use regularly, from better recommendations
to voice search.

Where does Machine Learning start and where does it end? What exactly does it
mean for a machine to learn something? If I download a copy of Wikipedia, has my
computer really “learned” something? Is it suddenly smarter? In this chapter we will
start by clarifying what Machine Learning is and why you may want to use it.

Then, before we set out to explore the Machine Learning continent, we will take a
look at the map and learn about the main regions and the most notable landmarks:
supervised versus unsupervised learning, online versus batch learning, instance-
based versus model-based learning. Then we will look at the workflow of a typical ML
project, discuss the main challenges you may face, and cover how to evaluate and
fine-tune a Machine Learning system.

```
3
```

This chapter introduces a lot of fundamental concepts (and jargon) that every data
scientist should know by heart. It will be a high-level overview (the only chapter
without much code), all rather simple, but you should make sure everything is
crystal-clear to you before continuing to the rest of the book. So grab a coffee and let’s
get started!

```
If you already know all the Machine Learning basics, you may want
to skip directly to Chapter 2. If you are not sure, try to answer all
the questions listed at the end of the chapter before moving on.
```

### What Is Machine Learning?

Machine Learning is the science (and art) of programming computers so they can
learn from data.

Here is a slightly more general definition:

```
[Machine Learning is the] field of study that gives computers the ability to learn
without being explicitly programmed.
—Arthur Samuel, 1959
```

And a more engineering-oriented one:

```
A computer program is said to learn from experience E with respect to some task T
and some performance measure P, if its performance on T, as measured by P, improves
with experience E.
—Tom Mitchell, 1997
```

For example, your spam filter is a Machine Learning program that can learn to flag
spam given examples of spam emails (e.g., flagged by users) and examples of regular
(nonspam, also called “ham”) emails. The examples that the system uses to learn are
called the training set. Each training example is called a training instance (or sample).
In this case, the task T is to flag spam for new emails, the experience E is the training
data, and the performance measure P needs to be defined; for example, you can use
the ratio of correctly classified emails. This particular performance measure is called
accuracy and it is often used in classification tasks.

If you just download a copy of Wikipedia, your computer has a lot more data, but it is
not suddenly better at any task. Thus, it is not Machine Learning.

### Why Use Machine Learning?

Consider how you would write a spam filter using traditional programming techni‐
ques (Figure 1-1):

**4 | Chapter 1: The Machine Learning Landscape**

1. First you would look at what spam typically looks like. You might notice that
   some words or phrases (such as “4U,” “credit card,” “free,” and “amazing”) tend to
   come up a lot in the subject. Perhaps you would also notice a few other patterns
   in the sender’s name, the email’s body, and so on.
2. You would write a detection algorithm for each of the patterns that you noticed,
   and your program would flag emails as spam if a number of these patterns are
   detected.
3. You would test your program, and repeat steps 1 and 2 until it is good enough.

Figure 1-1. The traditional approach

Since the problem is not trivial, your program will likely become a long list of com‐
plex rules—pretty hard to maintain.

In contrast, a spam filter based on Machine Learning techniques automatically learns
which words and phrases are good predictors of spam by detecting unusually fre‐
quent patterns of words in the spam examples compared to the ham examples
(Figure 1-2). The program is much shorter, easier to maintain, and most likely more
accurate.

```
Why Use Machine Learning? | 5
```

Figure 1-2. Machine Learning approach

Moreover, if spammers notice that all their emails containing “4U” are blocked, they
might start writing “For U” instead. A spam filter using traditional programming
techniques would need to be updated to flag “For U” emails. If spammers keep work‐
ing around your spam filter, you will need to keep writing new rules forever.

In contrast, a spam filter based on Machine Learning techniques automatically noti‐
ces that “For U” has become unusually frequent in spam flagged by users, and it starts
flagging them without your intervention (Figure 1-3).

Figure 1-3. Automatically adapting to change

Another area where Machine Learning shines is for problems that either are too com‐
plex for traditional approaches or have no known algorithm. For example, consider
speech recognition: say you want to start simple and write a program capable of dis‐
tinguishing the words “one” and “two.” You might notice that the word “two” starts
with a high-pitch sound (“T”), so you could hardcode an algorithm that measures
high-pitch sound intensity and use that to distinguish ones and twos. Obviously this
technique will not scale to thousands of words spoken by millions of very different

**6 | Chapter 1: The Machine Learning Landscape**

people in noisy environments and in dozens of languages. The best solution (at least
today) is to write an algorithm that learns by itself, given many example recordings
for each word.

Finally, Machine Learning can help humans learn (Figure 1-4): ML algorithms can be
inspected to see what they have learned (although for some algorithms this can be
tricky). For instance, once the spam filter has been trained on enough spam, it can
easily be inspected to reveal the list of words and combinations of words that it
believes are the best predictors of spam. Sometimes this will reveal unsuspected cor‐
relations or new trends, and thereby lead to a better understanding of the problem.

Applying ML techniques to dig into large amounts of data can help discover patterns
that were not immediately apparent. This is called data mining.

Figure 1-4. Machine Learning can help humans learn

To summarize, Machine Learning is great for:

- Problems for which existing solutions require a lot of hand-tuning or long lists of
  rules: one Machine Learning algorithm can often simplify code and perform bet‐
  ter.
- Complex problems for which there is no good solution at all using a traditional
  approach: the best Machine Learning techniques can find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data.

```
Why Use Machine Learning? | 7
```

### Types of Machine Learning Systems

There are so many different types of Machine Learning systems that it is useful to
classify them in broad categories based on:

- Whether or not they are trained with human supervision (supervised, unsuper‐
  vised, semisupervised, and Reinforcement Learning)
- Whether or not they can learn incrementally on the fly (online versus batch
  learning)
- Whether they work by simply comparing new data points to known data points,
  or instead detect patterns in the training data and build a predictive model, much
  like scientists do (instance-based versus model-based learning)

These criteria are not exclusive; you can combine them in any way you like. For
example, a state-of-the-art spam filter may learn on the fly using a deep neural net‐
work model trained using examples of spam and ham; this makes it an online, model-
based, supervised learning system.

Let’s look at each of these criteria a bit more closely.

#### Supervised/Unsupervised Learning

Machine Learning systems can be classified according to the amount and type of
supervision they get during training. There are four major categories: supervised
learning, unsupervised learning, semisupervised learning, and Reinforcement Learn‐
ing.

**Supervised learning**

In supervised learning, the training data you feed to the algorithm includes the desired
solutions, called labels (Figure 1-5).

Figure 1-5. A labeled training set for supervised learning (e.g., spam classification)

**8 | Chapter 1: The Machine Learning Landscape**

```
1 Fun fact: this odd-sounding name is a statistics term introduced by Francis Galton while he was studying the
fact that the children of tall people tend to be shorter than their parents. Since children were shorter, he called
this regression to the mean. This name was then applied to the methods he used to analyze correlations
between variables.
```

A typical supervised learning task is classification. The spam filter is a good example
of this: it is trained with many example emails along with their class (spam or ham),
and it must learn how to classify new emails.

Another typical task is to predict a target numeric value, such as the price of a car,
given a set of features (mileage, age, brand, etc.) called predictors. This sort of task is
called regression ( Figure 1-6).^1 To train the system, you need to give it many examples
of cars, including both their predictors and their labels (i.e., their prices).

```
In Machine Learning an attribute is a data type (e.g., “Mileage”),
while a feature has several meanings depending on the context, but
generally means an attribute plus its value (e.g., “Mileage =
15,000”). Many people use the words attribute and feature inter‐
changeably, though.
```

Figure 1-6. Regression

Note that some regression algorithms can be used for classification as well, and vice
versa. For example, Logistic Regression is commonly used for classification, as it can
output a value that corresponds to the probability of belonging to a given class (e.g.,
20% chance of being spam).

```
Types of Machine Learning Systems | 9
```

```
2 Some neural network architectures can be unsupervised, such as autoencoders and restricted Boltzmann
machines. They can also be semisupervised, such as in deep belief networks and unsupervised pretraining.
```

Here are some of the most important supervised learning algorithms (covered in this
book):

- k-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVMs)
- Decision Trees and Random Forests
- Neural networks^2

**Unsupervised learning**

In unsupervised learning, as you might guess, the training data is unlabeled
(Figure 1-7). The system tries to learn without a teacher.

Figure 1-7. An unlabeled training set for unsupervised learning

Here are some of the most important unsupervised learning algorithms (most of
these are covered in Chapter 8 and Chapter 9):

- Clustering
  —K-Means
  —DBSCAN
  —Hierarchical Cluster Analysis (HCA)
- Anomaly detection and novelty detection
  —One-class SVM
  —Isolation Forest

**10 | Chapter 1: The Machine Learning Landscape**

- Visualization and dimensionality reduction
  —Principal Component Analysis (PCA)
  —Kernel PCA
  —Locally-Linear Embedding (LLE)
  —t-distributed Stochastic Neighbor Embedding (t-SNE)
- Association rule learning
  —Apriori
  —Eclat

For example, say you have a lot of data about your blog’s visitors. You may want to
run a clustering algorithm to try to detect groups of similar visitors (Figure 1-8). At
no point do you tell the algorithm which group a visitor belongs to: it finds those
connections without your help. For example, it might notice that 40% of your visitors
are males who love comic books and generally read your blog in the evening, while
20% are young sci-fi lovers who visit during the weekends, and so on. If you use a
hierarchical clustering algorithm, it may also subdivide each group into smaller
groups. This may help you target your posts for each group.

Figure 1-8. Clustering

Visualization algorithms are also good examples of unsupervised learning algorithms:
you feed them a lot of complex and unlabeled data, and they output a 2D or 3D rep‐
resentation of your data that can easily be plotted (Figure 1-9). These algorithms try
to preserve as much structure as they can (e.g., trying to keep separate clusters in the
input space from overlapping in the visualization), so you can understand how the
data is organized and perhaps identify unsuspected patterns.

```
Types of Machine Learning Systems | 11
```

```
3 Notice how animals are rather well separated from vehicles, how horses are close to deer but far from birds,
and so on. Figure reproduced with permission from Socher, Ganjoo, Manning, and Ng (2013), “T-SNE visual‐
ization of the semantic word space.”
```

Figure 1-9. Example of a t-SNE visualization highlighting semantic clusters^3

A related task is dimensionality reduction, in which the goal is to simplify the data
without losing too much information. One way to do this is to merge several correla‐
ted features into one. For example, a car’s mileage may be very correlated with its age,
so the dimensionality reduction algorithm will merge them into one feature that rep‐
resents the car’s wear and tear. This is called feature extraction.

```
It is often a good idea to try to reduce the dimension of your train‐
ing data using a dimensionality reduction algorithm before you
feed it to another Machine Learning algorithm (such as a super‐
vised learning algorithm). It will run much faster, the data will take
up less disk and memory space, and in some cases it may also per‐
form better.
```

Yet another important unsupervised task is anomaly detection—for example, detect‐
ing unusual credit card transactions to prevent fraud, catching manufacturing defects,
or automatically removing outliers from a dataset before feeding it to another learn‐
ing algorithm. The system is shown mostly normal instances during training, so it
learns to recognize them and when it sees a new instance it can tell whether it looks

**12 | Chapter 1: The Machine Learning Landscape**

```
4 That’s when the system works perfectly. In practice it often creates a few clusters per person, and sometimes
mixes up two people who look alike, so you need to provide a few labels per person and manually clean up
some clusters.
```

like a normal one or whether it is likely an anomaly (see Figure 1-10). A very similar
task is novelty detection: the difference is that novelty detection algorithms expect to
see only normal data during training, while anomaly detection algorithms are usually
more tolerant, they can often perform well even with a small percentage of outliers in
the training set.

Figure 1-10. Anomaly detection

Finally, another common unsupervised task is association rule learning, in which the
goal is to dig into large amounts of data and discover interesting relations between
attributes. For example, suppose you own a supermarket. Running an association rule
on your sales logs may reveal that people who purchase barbecue sauce and potato
chips also tend to buy steak. Thus, you may want to place these items close to each
other.

**Semisupervised learning**

Some algorithms can deal with partially labeled training data, usually a lot of unla‐
beled data and a little bit of labeled data. This is called semisupervised learning
(Figure 1-11).

Some photo-hosting services, such as Google Photos, are good examples of this. Once
you upload all your family photos to the service, it automatically recognizes that the
same person A shows up in photos 1, 5, and 11, while another person B shows up in
photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all
the system needs is for you to tell it who these people are. Just one label per person,^4
and it is able to name everyone in every photo, which is useful for searching photos.

```
Types of Machine Learning Systems | 13
```

Figure 1-11. Semisupervised learning

Most semisupervised learning algorithms are combinations of unsupervised and
supervised algorithms. For example, deep belief networks (DBNs) are based on unsu‐
pervised components called restricted Boltzmann machines (RBMs) stacked on top of
one another. RBMs are trained sequentially in an unsupervised manner, and then the
whole system is fine-tuned using supervised learning techniques.

**Reinforcement Learning**

Reinforcement Learning is a very different beast. The learning system, called an agent
in this context, can observe the environment, select and perform actions, and get
rewards in return (or penalties in the form of negative rewards, as in Figure 1-12). It
must then learn by itself what is the best strategy, called a policy, to get the most
reward over time. A policy defines what action the agent should choose when it is in a
given situation.

**14 | Chapter 1: The Machine Learning Landscape**

Figure 1-12. Reinforcement Learning

For example, many robots implement Reinforcement Learning algorithms to learn
how to walk. DeepMind’s AlphaGo program is also a good example of Reinforcement
Learning: it made the headlines in May 2017 when it beat the world champion Ke Jie
at the game of Go. It learned its winning policy by analyzing millions of games, and
then playing many games against itself. Note that learning was turned off during the
games against the champion; AlphaGo was just applying the policy it had learned.

#### Batch and Online Learning

Another criterion used to classify Machine Learning systems is whether or not the
system can learn incrementally from a stream of incoming data.

**Batch learning**

In batch learning, the system is incapable of learning incrementally: it must be trained
using all the available data. This will generally take a lot of time and computing
resources, so it is typically done offline. First the system is trained, and then it is
launched into production and runs without learning anymore; it just applies what it
has learned. This is called offline learning.

If you want a batch learning system to know about new data (such as a new type of
spam), you need to train a new version of the system from scratch on the full dataset
(not just the new data, but also the old data), then stop the old system and replace it
with the new one.

Fortunately, the whole process of training, evaluating, and launching a Machine
Learning system can be automated fairly easily (as shown in Figure 1-3), so even a

```
Types of Machine Learning Systems | 15
```

batch learning system can adapt to change. Simply update the data and train a new
version of the system from scratch as often as needed.

This solution is simple and often works fine, but training using the full set of data can
take many hours, so you would typically train a new system only every 24 hours or
even just weekly. If your system needs to adapt to rapidly changing data (e.g., to pre‐
dict stock prices), then you need a more reactive solution.

Also, training on the full set of data requires a lot of computing resources (CPU,
memory space, disk space, disk I/O, network I/O, etc.). If you have a lot of data and
you automate your system to train from scratch every day, it will end up costing you a
lot of money. If the amount of data is huge, it may even be impossible to use a batch
learning algorithm.

Finally, if your system needs to be able to learn autonomously and it has limited
resources (e.g., a smartphone application or a rover on Mars), then carrying around
large amounts of training data and taking up a lot of resources to train for hours
every day is a showstopper.

Fortunately, a better option in all these cases is to use algorithms that are capable of
learning incrementally.

**Online learning**

In online learning, you train the system incrementally by feeding it data instances
sequentially, either individually or by small groups called mini-batches. Each learning
step is fast and cheap, so the system can learn about new data on the fly, as it arrives
(see Figure 1-13).

Figure 1-13. Online learning

Online learning is great for systems that receive data as a continuous flow (e.g., stock
prices) and need to adapt to change rapidly or autonomously. It is also a good option

**16 | Chapter 1: The Machine Learning Landscape**

if you have limited computing resources: once an online learning system has learned
about new data instances, it does not need them anymore, so you can discard them
(unless you want to be able to roll back to a previous state and “replay” the data). This
can save a huge amount of space.

Online learning algorithms can also be used to train systems on huge datasets that
cannot fit in one machine’s main memory (this is called out-of-core learning). The
algorithm loads part of the data, runs a training step on that data, and repeats the
process until it has run on all of the data (see Figure 1-14).

```
Out-of-core learning is usually done offline (i.e., not on the live
system), so online learning can be a confusing name. Think of it as
incremental learning.
```

Figure 1-14. Using online learning to handle huge datasets

One important parameter of online learning systems is how fast they should adapt to
changing data: this is called the learning rate. If you set a high learning rate, then your
system will rapidly adapt to new data, but it will also tend to quickly forget the old
data (you don’t want a spam filter to flag only the latest kinds of spam it was shown).
Conversely, if you set a low learning rate, the system will have more inertia; that is, it
will learn more slowly, but it will also be less sensitive to noise in the new data or to
sequences of nonrepresentative data points (outliers).

A big challenge with online learning is that if bad data is fed to the system, the sys‐
tem’s performance will gradually decline. If we are talking about a live system, your
clients will notice. For example, bad data could come from a malfunctioning sensor
on a robot, or from someone spamming a search engine to try to rank high in search

```
Types of Machine Learning Systems | 17
```

results. To reduce this risk, you need to monitor your system closely and promptly
switch learning off (and possibly revert to a previously working state) if you detect a
drop in performance. You may also want to monitor the input data and react to
abnormal data (e.g., using an anomaly detection algorithm).

#### Instance-Based Versus Model-Based Learning

One more way to categorize Machine Learning systems is by how they generalize.
Most Machine Learning tasks are about making predictions. This means that given a
number of training examples, the system needs to be able to generalize to examples it
has never seen before. Having a good performance measure on the training data is
good, but insufficient; the true goal is to perform well on new instances.

There are two main approaches to generalization: instance-based learning and
model-based learning.

**Instance-based learning**

Possibly the most trivial form of learning is simply to learn by heart. If you were to
create a spam filter this way, it would just flag all emails that are identical to emails
that have already been flagged by users—not the worst solution, but certainly not the
best.

Instead of just flagging emails that are identical to known spam emails, your spam
filter could be programmed to also flag emails that are very similar to known spam
emails. This requires a measure of similarity between two emails. A (very basic) simi‐
larity measure between two emails could be to count the number of words they have
in common. The system would flag an email as spam if it has many words in com‐
mon with a known spam email.

This is called instance-based learning: the system learns the examples by heart, then
generalizes to new cases by comparing them to the learned examples (or a subset of
them), using a similarity measure. For example, in Figure 1-15 the new instance
would be classified as a triangle because the majority of the most similar instances
belong to that class.

**18 | Chapter 1: The Machine Learning Landscape**

Figure 1-15. Instance-based learning

**Model-based learning**

Another way to generalize from a set of examples is to build a model of these exam‐
ples, then use that model to make predictions. This is called model-based learning
(Figure 1-16).

Figure 1-16. Model-based learning

For example, suppose you want to know if money makes people happy, so you down‐
load the Better Life Index data from the OECD’s website as well as stats about GDP
per capita from the IMF’s website. Then you join the tables and sort by GDP per cap‐
ita. Table 1-1 shows an excerpt of what you get.

```
Types of Machine Learning Systems | 19
```

```
5 By convention, the Greek letter θ (theta) is frequently used to represent model parameters.
```

Table 1-1. Does money make people happier?

```
Country GDP per capita (USD) Life satisfaction
Hungary 12,240 4.9
Korea 27,195 5.8
France 37,675 6.5
Australia 50,962 7.3
United States 55,805 7.2
```

Let’s plot the data for a few random countries (Figure 1-17).

Figure 1-17. Do you see a trend here?

There does seem to be a trend here! Although the data is noisy (i.e., partly random), it
looks like life satisfaction goes up more or less linearly as the country’s GDP per cap‐
ita increases. So you decide to model life satisfaction as a linear function of GDP per
capita. This step is called model selection: you selected a linear model of life satisfac‐
tion with just one attribute, GDP per capita (Equation 1-1).

```
Equation 1-1. A simple linear model
life_satisfaction =θ 0 +θ 1 × GDP_per_capita
```

This model has two model parameters, θ 0 and θ 1.^5 By tweaking these parameters, you
can make your model represent any linear function, as shown in Figure 1-18.

**20 | Chapter 1: The Machine Learning Landscape**

Figure 1-18. A few possible linear models

Before you can use your model, you need to define the parameter values θ 0 and θ 1.
How can you know which values will make your model perform best? To answer this
question, you need to specify a performance measure. You can either define a utility
function (or fitness function) that measures how good your model is, or you can define
a cost function that measures how bad it is. For linear regression problems, people
typically use a cost function that measures the distance between the linear model’s
predictions and the training examples; the objective is to minimize this distance.

This is where the Linear Regression algorithm comes in: you feed it your training
examples and it finds the parameters that make the linear model fit best to your data.
This is called training the model. In our case the algorithm finds that the optimal
parameter values are θ 0 = 4.85 and θ 1 = 4.91 × 10–5.

Now the model fits the training data as closely as possible (for a linear model), as you
can see in Figure 1-19.

Figure 1-19. The linear model that fits the training data best

```
Types of Machine Learning Systems | 21
```

```
6 The prepare_country_stats() function’s definition is not shown here (see this chapter’s Jupyter notebook if
you want all the gory details). It’s just boring Pandas code that joins the life satisfaction data from the OECD
with the GDP per capita data from the IMF.
7 It’s okay if you don’t understand all the code yet; we will present Scikit-Learn in the following chapters.
```

You are finally ready to run the model to make predictions. For example, say you
want to know how happy Cypriots are, and the OECD data does not have the answer.
Fortunately, you can use your model to make a good prediction: you look up Cyprus’s
GDP per capita, find $22,587, and then apply your model and find that life satisfac‐
tion is likely to be somewhere around 4.85 + 22,587 × 4.91 × 10-5 = 5.96.

To whet your appetite, Example 1-1 shows the Python code that loads the data, pre‐
pares it,^6 creates a scatterplot for visualization, and then trains a linear model and
makes a prediction.^7

Example 1-1. Training and running a linear model using Scikit-Learn

**import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model**

_# Load the data_
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter=' **\t** ',
encoding='latin1', na_values="n/a")

_# Prepare the data_
country*stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c*[country*stats["GDP per capita"]]
y = np.c*[country_stats["Life satisfaction"]]

_# Visualize the data_
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

_# Select a linear model_
model = sklearn.linear_model.LinearRegression()

_# Train the model_
model.fit(X, y)

_# Make a prediction for Cyprus_
X*new = [[22587]] *# Cyprus' GDP per capita*
**print** (model.predict(X_new)) *# outputs [[5.96242338]]\_

**22 | Chapter 1: The Machine Learning Landscape**

```
If you had used an instance-based learning algorithm instead, you
would have found that Slovenia has the closest GDP per capita to
that of Cyprus ($20,732), and since the OECD data tells us that
Slovenians’ life satisfaction is 5.7, you would have predicted a life
satisfaction of 5.7 for Cyprus. If you zoom out a bit and look at the
two next closest countries, you will find Portugal and Spain with
life satisfactions of 5.1 and 6.5, respectively. Averaging these three
values, you get 5.77, which is pretty close to your model-based pre‐
diction. This simple algorithm is called k-Nearest Neighbors regres‐
sion (in this example, k = 3).
Replacing the Linear Regression model with k-Nearest Neighbors
regression in the previous code is as simple as replacing these two
lines:
import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()
with these two:
import sklearn.neighbors
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
```

If all went well, your model will make good predictions. If not, you may need to use
more attributes (employment rate, health, air pollution, etc.), get more or better qual‐
ity training data, or perhaps select a more powerful model (e.g., a Polynomial Regres‐
sion model).

In summary:

- You studied the data.
- You selected a model.
- You trained it on the training data (i.e., the learning algorithm searched for the
  model parameter values that minimize a cost function).
- Finally, you applied the model to make predictions on new cases (this is called
  inference), hoping that this model will generalize well.

This is what a typical Machine Learning project looks like. In Chapter 2 you will
experience this first-hand by going through an end-to-end project.

We have covered a lot of ground so far: you now know what Machine Learning is
really about, why it is useful, what some of the most common categories of ML sys‐
tems are, and what a typical project workflow looks like. Now let’s look at what can go
wrong in learning and prevent you from making accurate predictions.

```
Types of Machine Learning Systems | 23
```

### Main Challenges of Machine Learning

In short, since your main task is to select a learning algorithm and train it on some
data, the two things that can go wrong are “bad algorithm” and “bad data.” Let’s start
with examples of bad data.

#### Insufficient Quantity of Training Data

For a toddler to learn what an apple is, all it takes is for you to point to an apple and
say “apple” (possibly repeating this procedure a few times). Now the child is able to
recognize apples in all sorts of colors and shapes. Genius.

Machine Learning is not quite there yet; it takes a lot of data for most Machine Learn‐
ing algorithms to work properly. Even for very simple problems you typically need
thousands of examples, and for complex problems such as image or speech recogni‐
tion you may need millions of examples (unless you can reuse parts of an existing
model).

**24 | Chapter 1: The Machine Learning Landscape**

8 For example, knowing whether to write “to,” “two,” or “too” depending on the context.
9 Figure reproduced with permission from Banko and Brill (2001), “Learning Curves for Confusion Set Disam‐
biguation.”
10 “The Unreasonable Effectiveness of Data,” Peter Norvig et al. (2009).

```
The Unreasonable Effectiveness of Data
In a famous paper published in 2001, Microsoft researchers Michele Banko and Eric
Brill showed that very different Machine Learning algorithms, including fairly simple
ones, performed almost identically well on a complex problem of natural language
disambiguation^8 once they were given enough data (as you can see in Figure 1-20).
```

```
Figure 1-20. The importance of data versus algorithms^9
```

```
As the authors put it: “these results suggest that we may want to reconsider the trade-
off between spending time and money on algorithm development versus spending it
on corpus development.”
The idea that data matters more than algorithms for complex problems was further
popularized by Peter Norvig et al. in a paper titled “The Unreasonable Effectiveness
of Data” published in 2009.^10 It should be noted, however, that small- and medium-
sized datasets are still very common, and it is not always easy or cheap to get extra
training data, so don’t abandon algorithms just yet.
```

```
Main Challenges of Machine Learning | 25
```

#### Nonrepresentative Training Data

In order to generalize well, it is crucial that your training data be representative of the
new cases you want to generalize to. This is true whether you use instance-based
learning or model-based learning.

For example, the set of countries we used earlier for training the linear model was not
perfectly representative; a few countries were missing. Figure 1-21 shows what the
data looks like when you add the missing countries.

Figure 1-21. A more representative training sample

If you train a linear model on this data, you get the solid line, while the old model is
represented by the dotted line. As you can see, not only does adding a few missing
countries significantly alter the model, but it makes it clear that such a simple linear
model is probably never going to work well. It seems that very rich countries are not
happier than moderately rich countries (in fact they seem unhappier), and conversely
some poor countries seem happier than many rich countries.

By using a nonrepresentative training set, we trained a model that is unlikely to make
accurate predictions, especially for very poor and very rich countries.

It is crucial to use a training set that is representative of the cases you want to general‐
ize to. This is often harder than it sounds: if the sample is too small, you will have
sampling noise (i.e., nonrepresentative data as a result of chance), but even very large
samples can be nonrepresentative if the sampling method is flawed. This is called
sampling bias.

```
A Famous Example of Sampling Bias
Perhaps the most famous example of sampling bias happened during the US presi‐
dential election in 1936, which pitted Landon against Roosevelt: the Literary Digest
conducted a very large poll, sending mail to about 10 million people. It got 2.4 million
answers, and predicted with high confidence that Landon would get 57% of the votes.
```

**26 | Chapter 1: The Machine Learning Landscape**

```
Instead, Roosevelt won with 62% of the votes. The flaw was in the Literary Digest’s
sampling method:
```

- First, to obtain the addresses to send the polls to, the Literary Digest used tele‐
  phone directories, lists of magazine subscribers, club membership lists, and the
  like. All of these lists tend to favor wealthier people, who are more likely to vote
  Republican (hence Landon).
- Second, less than 25% of the people who received the poll answered. Again, this
  introduces a sampling bias, by ruling out people who don’t care much about poli‐
  tics, people who don’t like the Literary Digest, and other key groups. This is a spe‐
  cial type of sampling bias called nonresponse bias.

```
Here is another example: say you want to build a system to recognize funk music vid‐
eos. One way to build your training set is to search “funk music” on YouTube and use
the resulting videos. But this assumes that YouTube’s search engine returns a set of
videos that are representative of all the funk music videos on YouTube. In reality, the
search results are likely to be biased toward popular artists (and if you live in Brazil
you will get a lot of “funk carioca” videos, which sound nothing like James Brown).
On the other hand, how else can you get a large training set?
```

#### Poor-Quality Data

Obviously, if your training data is full of errors, outliers, and noise (e.g., due to poor-
quality measurements), it will make it harder for the system to detect the underlying
patterns, so your system is less likely to perform well. It is often well worth the effort
to spend time cleaning up your training data. The truth is, most data scientists spend
a significant part of their time doing just that. For example:

- If some instances are clearly outliers, it may help to simply discard them or try to
  fix the errors manually.
- If some instances are missing a few features (e.g., 5% of your customers did not
  specify their age), you must decide whether you want to ignore this attribute alto‐
  gether, ignore these instances, fill in the missing values (e.g., with the median
  age), or train one model with the feature and one model without it, and so on.

#### Irrelevant Features

As the saying goes: garbage in, garbage out. Your system will only be capable of learn‐
ing if the training data contains enough relevant features and not too many irrelevant
ones. A critical part of the success of a Machine Learning project is coming up with a
good set of features to train on. This process, called feature engineering, involves:

```
Main Challenges of Machine Learning | 27
```

- Feature selection: selecting the most useful features to train on among existing
  features.
- Feature extraction: combining existing features to produce a more useful one (as
  we saw earlier, dimensionality reduction algorithms can help).
- Creating new features by gathering new data.

Now that we have looked at many examples of bad data, let’s look at a couple of exam‐
ples of bad algorithms.

#### Overfitting the Training Data

Say you are visiting a foreign country and the taxi driver rips you off. You might be
tempted to say that all taxi drivers in that country are thieves. Overgeneralizing is
something that we humans do all too often, and unfortunately machines can fall into
the same trap if we are not careful. In Machine Learning this is called overfitting: it
means that the model performs well on the training data, but it does not generalize
well.

Figure 1-22 shows an example of a high-degree polynomial life satisfaction model
that strongly overfits the training data. Even though it performs much better on the
training data than the simple linear model, would you really trust its predictions?

Figure 1-22. Overfitting the training data

Complex models such as deep neural networks can detect subtle patterns in the data,
but if the training set is noisy, or if it is too small (which introduces sampling noise),
then the model is likely to detect patterns in the noise itself. Obviously these patterns
will not generalize to new instances. For example, say you feed your life satisfaction
model many more attributes, including uninformative ones such as the country’s
name. In that case, a complex model may detect patterns like the fact that all coun‐
tries in the training data with a w in their name have a life satisfaction greater than 7:
New Zealand (7.3), Norway (7.4), Sweden (7.2), and Switzerland (7.5). How confident

**28 | Chapter 1: The Machine Learning Landscape**

are you that the W-satisfaction rule generalizes to Rwanda or Zimbabwe? Obviously
this pattern occurred in the training data by pure chance, but the model has no way
to tell whether a pattern is real or simply the result of noise in the data.

```
Overfitting happens when the model is too complex relative to the
amount and noisiness of the training data. The possible solutions
are:
```

- To simplify the model by selecting one with fewer parameters
  (e.g., a linear model rather than a high-degree polynomial
  model), by reducing the number of attributes in the training
  data or by constraining the model
- To gather more training data
- To reduce the noise in the training data (e.g., fix data errors
  and remove outliers)

Constraining a model to make it simpler and reduce the risk of overfitting is called
regularization. For example, the linear model we defined earlier has two parameters,
θ 0 and θ 1. This gives the learning algorithm two degrees of freedom to adapt the model
to the training data: it can tweak both the height (θ 0 ) and the slope (θ 1 ) of the line. If
we forced θ 1 = 0, the algorithm would have only one degree of freedom and would
have a much harder time fitting the data properly: all it could do is move the line up
or down to get as close as possible to the training instances, so it would end up
around the mean. A very simple model indeed! If we allow the algorithm to modify θ 1
but we force it to keep it small, then the learning algorithm will effectively have some‐
where in between one and two degrees of freedom. It will produce a simpler model
than with two degrees of freedom, but more complex than with just one. You want to
find the right balance between fitting the training data perfectly and keeping the
model simple enough to ensure that it will generalize well.

Figure 1-23 shows three models: the dotted line represents the original model that
was trained with a few countries missing, the dashed line is our second model trained
with all countries, and the solid line is a linear model trained with the same data as
the first model but with a regularization constraint. You can see that regularization
forced the model to have a smaller slope, which fits a bit less the training data that the
model was trained on, but actually allows it to generalize better to new examples.

```
Main Challenges of Machine Learning | 29
```

Figure 1-23. Regularization reduces the risk of overfitting

The amount of regularization to apply during learning can be controlled by a hyper‐
parameter. A hyperparameter is a parameter of a learning algorithm (not of the
model). As such, it is not affected by the learning algorithm itself; it must be set prior
to training and remains constant during training. If you set the regularization hyper‐
parameter to a very large value, you will get an almost flat model (a slope close to
zero); the learning algorithm will almost certainly not overfit the training data, but it
will be less likely to find a good solution. Tuning hyperparameters is an important
part of building a Machine Learning system (you will see a detailed example in the
next chapter).

#### Underfitting the Training Data

As you might guess, underfitting is the opposite of overfitting: it occurs when your
model is too simple to learn the underlying structure of the data. For example, a lin‐
ear model of life satisfaction is prone to underfit; reality is just more complex than
the model, so its predictions are bound to be inaccurate, even on the training exam‐
ples.

The main options to fix this problem are:

- Selecting a more powerful model, with more parameters
- Feeding better features to the learning algorithm (feature engineering)
- Reducing the constraints on the model (e.g., reducing the regularization hyper‐
  parameter)

#### Stepping Back

By now you already know a lot about Machine Learning. However, we went through
so many concepts that you may be feeling a little lost, so let’s step back and look at the
big picture:

**30 | Chapter 1: The Machine Learning Landscape**

- Machine Learning is about making machines get better at some task by learning
  from data, instead of having to explicitly code rules.
- There are many different types of ML systems: supervised or not, batch or online,
  instance-based or model-based, and so on.
- In a ML project you gather data in a training set, and you feed the training set to
  a learning algorithm. If the algorithm is model-based it tunes some parameters to
  fit the model to the training set (i.e., to make good predictions on the training set
  itself ), and then hopefully it will be able to make good predictions on new cases
  as well. If the algorithm is instance-based, it just learns the examples by heart and
  generalizes to new instances by comparing them to the learned instances using a
  similarity measure.
- The system will not perform well if your training set is too small, or if the data is
  not representative, noisy, or polluted with irrelevant features (garbage in, garbage
  out). Lastly, your model needs to be neither too simple (in which case it will
  underfit) nor too complex (in which case it will overfit).

There’s just one last important topic to cover: once you have trained a model, you
don’t want to just “hope” it generalizes to new cases. You want to evaluate it, and fine-
tune it if necessary. Let’s see how.

### Testing and Validating

The only way to know how well a model will generalize to new cases is to actually try
it out on new cases. One way to do that is to put your model in production and moni‐
tor how well it performs. This works well, but if your model is horribly bad, your
users will complain—not the best idea.

A better option is to split your data into two sets: the training set and the test set. As
these names imply, you train your model using the training set, and you test it using
the test set. The error rate on new cases is called the generalization error (or out-of-
sample error), and by evaluating your model on the test set, you get an estimate of this
error. This value tells you how well your model will perform on instances it has never
seen before.

If the training error is low (i.e., your model makes few mistakes on the training set)
but the generalization error is high, it means that your model is overfitting the train‐
ing data.

```
It is common to use 80% of the data for training and hold out 20%
for testing. However, this depends on the size of the dataset: if it
contains 10 million instances, then holding out 1% means your test
set will contain 100,000 instances: that’s probably more than
enough to get a good estimate of the generalization error.
```

```
Testing and Validating | 31
```

#### Hyperparameter Tuning and Model Selection

So evaluating a model is simple enough: just use a test set. Now suppose you are hesi‐
tating between two models (say a linear model and a polynomial model): how can
you decide? One option is to train both and compare how well they generalize using
the test set.

Now suppose that the linear model generalizes better, but you want to apply some
regularization to avoid overfitting. The question is: how do you choose the value of
the regularization hyperparameter? One option is to train 100 different models using
100 different values for this hyperparameter. Suppose you find the best hyperparame‐
ter value that produces a model with the lowest generalization error, say just 5% error.

So you launch this model into production, but unfortunately it does not perform as
well as expected and produces 15% errors. What just happened?

The problem is that you measured the generalization error multiple times on the test
set, and you adapted the model and hyperparameters to produce the best model for
that particular set. This means that the model is unlikely to perform as well on new
data.

A common solution to this problem is called holdout validation: you simply hold out
part of the training set to evaluate several candidate models and select the best one.
The new heldout set is called the validation set (or sometimes the development set, or
dev set). More specifically, you train multiple models with various hyperparameters
on the reduced training set (i.e., the full training set minus the validation set), and
you select the model that performs best on the validation set. After this holdout vali‐
dation process, you train the best model on the full training set (including the valida‐
tion set), and this gives you the final model. Lastly, you evaluate this final model on
the test set to get an estimate of the generalization error.

This solution usually works quite well. However, if the validation set is too small, then
model evaluations will be imprecise: you may end up selecting a suboptimal model by
mistake. Conversely, if the validation set is too large, then the remaining training set
will be much smaller than the full training set. Why is this bad? Well, since the final
model will be trained on the full training set, it is not ideal to compare candidate
models trained on a much smaller training set. It would be like selecting the fastest
sprinter to participate in a marathon. One way to solve this problem is to perform
repeated cross-validation, using many small validation sets. Each model is evaluated
once per validation set, after it is trained on the rest of the data. By averaging out all
the evaluations of a model, we get a much more accurate measure of its performance.
However, there is a drawback: the training time is multiplied by the number of valida‐
tion sets.

**32 | Chapter 1: The Machine Learning Landscape**

11 “The Lack of A Priori Distinctions Between Learning Algorithms,” D. Wolpert (1996).

#### Data Mismatch

```
In some cases, it is easy to get a large amount of data for training, but it is not per‐
fectly representative of the data that will be used in production. For example, suppose
you want to create a mobile app to take pictures of flowers and automatically deter‐
mine their species. You can easily download millions of pictures of flowers on the
web, but they won’t be perfectly representative of the pictures that will actually be
taken using the app on a mobile device. Perhaps you only have 10,000 representative
pictures (i.e., actually taken with the app). In this case, the most important rule to
remember is that the validation set and the test must be as representative as possible
of the data you expect to use in production, so they should be composed exclusively
of representative pictures: you can shuffle them and put half in the validation set, and
half in the test set (making sure that no duplicates or near-duplicates end up in both
sets). After training your model on the web pictures, if you observe that the perfor‐
mance of your model on the validation set is disappointing, you will not know
whether this is because your model has overfit the training set, or whether this is just
due to the mismatch between the web pictures and the mobile app pictures. One sol‐
ution is to hold out part of the training pictures (from the web) in yet another set that
Andrew Ng calls the train-dev set. After the model is trained (on the training set, not
on the train-dev set), you can evaluate it on the train-dev set: if it performs well, then
the model is not overfitting the training set, so if performs poorly on the validation
set, the problem must come from the data mismatch. You can try to tackle this prob‐
lem by preprocessing the web images to make them look more like the pictures that
will be taken by the mobile app, and then retraining the model. Conversely, if the
model performs poorly on the train-dev set, then the model must have overfit the
training set, so you should try to simplify or regularize the model, get more training
data and clean up the training data, as discussed earlier.
```

```
No Free Lunch Theorem
A model is a simplified version of the observations. The simplifications are meant to
discard the superfluous details that are unlikely to generalize to new instances. How‐
ever, to decide what data to discard and what data to keep, you must make assump‐
tions. For example, a linear model makes the assumption that the data is
fundamentally linear and that the distance between the instances and the straight line
is just noise, which can safely be ignored.
In a famous 1996 paper,^11 David Wolpert demonstrated that if you make absolutely
no assumption about the data, then there is no reason to prefer one model over any
other. This is called the No Free Lunch (NFL) theorem. For some datasets the best
```

```
Testing and Validating | 33
```

```
model is a linear model, while for other datasets it is a neural network. There is no
model that is a priori guaranteed to work better (hence the name of the theorem). The
only way to know for sure which model is best is to evaluate them all. Since this is not
possible, in practice you make some reasonable assumptions about the data and you
evaluate only a few reasonable models. For example, for simple tasks you may evalu‐
ate linear models with various levels of regularization, and for a complex problem you
may evaluate various neural networks.
```

### Exercises

In this chapter we have covered some of the most important concepts in Machine
Learning. In the next chapters we will dive deeper and write more code, but before we
do, make sure you know how to answer the following questions:

1. How would you define Machine Learning?
2. Can you name four types of problems where it shines?
3. What is a labeled training set?
4. What are the two most common supervised tasks?
5. Can you name four common unsupervised tasks?
6. What type of Machine Learning algorithm would you use to allow a robot to
   walk in various unknown terrains?
7. What type of algorithm would you use to segment your customers into multiple
   groups?
8. Would you frame the problem of spam detection as a supervised learning prob‐
   lem or an unsupervised learning problem?
9. What is an online learning system?
10. What is out-of-core learning?
11. What type of learning algorithm relies on a similarity measure to make predic‐
    tions?
12. What is the difference between a model parameter and a learning algorithm’s
    hyperparameter?
13. What do model-based learning algorithms search for? What is the most common
    strategy they use to succeed? How do they make predictions?
14. Can you name four of the main challenges in Machine Learning?
15. If your model performs great on the training data but generalizes poorly to new
    instances, what is happening? Can you name three possible solutions?
16. What is a test set and why would you want to use it?

**34 | Chapter 1: The Machine Learning Landscape**

17. What is the purpose of a validation set?
18. What can go wrong if you tune hyperparameters using the test set?
19. What is repeated cross-validation and why would you prefer it to using a single
    validation set?

Solutions to these exercises are available in ???.

```
Exercises | 35
```

```
1 The example project is completely fictitious; the goal is just to illustrate the main steps of a Machine Learning
project, not to learn anything about the real estate business.
```

**CHAPTER 2**

**End-to-End Machine Learning Project**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 2 in the final
release of the book.
```

In this chapter, you will go through an example project end to end, pretending to be a
recently hired data scientist in a real estate company.^1 Here are the main steps you will
go through:

1. Look at the big picture.
2. Get the data.
3. Discover and visualize the data to gain insights.
4. Prepare the data for Machine Learning algorithms.
5. Select a model and train it.
6. Fine-tune your model.
7. Present your solution.
8. Launch, monitor, and maintain your system.

```
37
```

```
2 The original dataset appeared in R. Kelley Pace and Ronald Barry, “Sparse Spatial Autoregressions,” Statistics
& Probability Letters 33, no. 3 (1997): 291–297.
```

### Working with Real Data

When you are learning about Machine Learning it is best to actually experiment with
real-world data, not just artificial datasets. Fortunately, there are thousands of open
datasets to choose from, ranging across all sorts of domains. Here are a few places
you can look to get data:

- Popular open data repositories:
  —UC Irvine Machine Learning Repository
  —Kaggle datasets
  —Amazon’s AWS datasets
- Meta portals (they list open data repositories):
  —http://dataportals.org/
  —http://opendatamonitor.eu/
  —http://quandl.com/
- Other pages listing many popular open data repositories:
  —Wikipedia’s list of Machine Learning datasets
  —Quora.com question
  —Datasets subreddit

In this chapter we chose the California Housing Prices dataset from the StatLib repos‐
itory^2 (see Figure 2-1). This dataset was based on data from the 1990 California cen‐
sus. It is not exactly recent (you could still afford a nice house in the Bay Area at the
time), but it has many qualities for learning, so we will pretend it is recent data. We
also added a categorical attribute and removed a few features for teaching purposes.

**38 | Chapter 2: End-to-End Machine Learning Project**

Figure 2-1. California housing prices

### Look at the Big Picture

Welcome to Machine Learning Housing Corporation! The first task you are asked to
perform is to build a model of housing prices in California using the California cen‐
sus data. This data has metrics such as the population, median income, median hous‐
ing price, and so on for each block group in California. Block groups are the smallest
geographical unit for which the US Census Bureau publishes sample data (a block
group typically has a population of 600 to 3,000 people). We will just call them “dis‐
tricts” for short.

Your model should learn from this data and be able to predict the median housing
price in any district, given all the other metrics.

```
Since you are a well-organized data scientist, the first thing you do
is to pull out your Machine Learning project checklist. You can
start with the one in ???; it should work reasonably well for most
Machine Learning projects but make sure to adapt it to your needs.
In this chapter we will go through many checklist items, but we will
also skip a few, either because they are self-explanatory or because
they will be discussed in later chapters.
```

#### Frame the Problem

The first question to ask your boss is what exactly is the business objective; building a
model is probably not the end goal. How does the company expect to use and benefit

```
Look at the Big Picture | 39
```

```
3 A piece of information fed to a Machine Learning system is often called a signal in reference to Shannon’s
information theory: you want a high signal/noise ratio.
```

from this model? This is important because it will determine how you frame the
problem, what algorithms you will select, what performance measure you will use to
evaluate your model, and how much effort you should spend tweaking it.

Your boss answers that your model’s output (a prediction of a district’s median hous‐
ing price) will be fed to another Machine Learning system (see Figure 2-2), along
with many other signals.^3 This downstream system will determine whether it is worth
investing in a given area or not. Getting this right is critical, as it directly affects reve‐
nue.

Figure 2-2. A Machine Learning pipeline for real estate investments

```
Pipelines
A sequence of data processing components is called a data pipeline. Pipelines are very
common in Machine Learning systems, since there is a lot of data to manipulate and
many data transformations to apply.
Components typically run asynchronously. Each component pulls in a large amount
of data, processes it, and spits out the result in another data store, and then some time
later the next component in the pipeline pulls this data and spits out its own output,
and so on. Each component is fairly self-contained: the interface between components
is simply the data store. This makes the system quite simple to grasp (with the help of
a data flow graph), and different teams can focus on different components. Moreover,
if a component breaks down, the downstream components can often continue to run
normally (at least for a while) by just using the last output from the broken compo‐
nent. This makes the architecture quite robust.
```

**40 | Chapter 2: End-to-End Machine Learning Project**

```
On the other hand, a broken component can go unnoticed for some time if proper
monitoring is not implemented. The data gets stale and the overall system’s perfor‐
mance drops.
```

The next question to ask is what the current solution looks like (if any). It will often
give you a reference performance, as well as insights on how to solve the problem.
Your boss answers that the district housing prices are currently estimated manually
by experts: a team gathers up-to-date information about a district, and when they
cannot get the median housing price, they estimate it using complex rules.

This is costly and time-consuming, and their estimates are not great; in cases where
they manage to find out the actual median housing price, they often realize that their
estimates were off by more than 20%. This is why the company thinks that it would
be useful to train a model to predict a district’s median housing price given other data
about that district. The census data looks like a great dataset to exploit for this pur‐
pose, since it includes the median housing prices of thousands of districts, as well as
other data.

Okay, with all this information you are now ready to start designing your system.
First, you need to frame the problem: is it supervised, unsupervised, or Reinforce‐
ment Learning? Is it a classification task, a regression task, or something else? Should
you use batch learning or online learning techniques? Before you read on, pause and
try to answer these questions for yourself.

Have you found the answers? Let’s see: it is clearly a typical supervised learning task
since you are given labeled training examples (each instance comes with the expected
output, i.e., the district’s median housing price). Moreover, it is also a typical regres‐
sion task, since you are asked to predict a value. More specifically, this is a multiple
regression problem since the system will use multiple features to make a prediction (it
will use the district’s population, the median income, etc.). It is also a univariate
regression problem since we are only trying to predict a single value for each district.
If we were trying to predict multiple values per district, it would be a multivariate
regression problem. Finally, there is no continuous flow of data coming in the system,
there is no particular need to adjust to changing data rapidly, and the data is small
enough to fit in memory, so plain batch learning should do just fine.

```
If the data was huge, you could either split your batch learning
work across multiple servers (using the MapReduce technique), or
you could use an online learning technique instead.
```

```
Look at the Big Picture | 41
```

#### Select a Performance Measure

Your next step is to select a performance measure. A typical performance measure for
regression problems is the Root Mean Square Error (RMSE). It gives an idea of how
much error the system typically makes in its predictions, with a higher weight for
large errors. Equation 2-1 shows the mathematical formula to compute the RMSE.

```
Equation 2-1. Root Mean Square Error (RMSE)
```

```
RMSE X ,h = m^1 ∑
i= 1
```

```
m
h x i −yi
```

```
2
```

**42 | Chapter 2: End-to-End Machine Learning Project**

4 Recall that the transpose operator flips a column vector into a row vector (and vice versa).

```
Notations
This equation introduces several very common Machine Learning notations that we
will use throughout this book:
```

- m is the number of instances in the dataset you are measuring the RMSE on.
  —For example, if you are evaluating the RMSE on a validation set of 2,000 dis‐
  tricts, then m = 2,000.
- **x** (i) is a vector of all the feature values (excluding the label) of the ith instance in
  the dataset, and y(i) is its label (the desired output value for that instance).
  —For example, if the first district in the dataset is located at longitude –118.29°,
  latitude 33.91°, and it has 1,416 inhabitants with a median income of $38,372,
  and the median house value is $156,400 (ignoring the other features for now),
  then:

```
x^1 =
```

```
−118. 29
33. 91
1, 416
38, 372
```

```
and:
```

```
y^1 = 156, 400
```

- **X** is a matrix containing all the feature values (excluding labels) of all instances in
  the dataset. There is one row per instance and the ith row is equal to the transpose
  of **x** (i), noted ( **x** (i))T.^4
  —For example, if the first district is as just described, then the matrix **X** looks
  like this:

```
X =
```

```
x^1
```

```
T
```

```
x^2
```

```
T
```

```
⋮
x^1999 T
```

```
x^2000 T
```

```
=
```

```
−118. 29 33. 91 1, 416 38, 372
⋮ ⋮ ⋮ ⋮
```

```
Look at the Big Picture | 43
```

- h is your system’s prediction function, also called a hypothesis. When your system
  is given an instance’s feature vector **x** (i), it outputs a predicted value ŷ(i) = h( **x** (i))
  for that instance (ŷ is pronounced “y-hat”).
  —For example, if your system predicts that the median housing price in the first
  district is $158,400, then ŷ(1) = h( **x** (1)) = 158,400. The prediction error for this
  district is ŷ(1) – y(1) = 2,000.
- RMSE( **X** ,h) is the cost function measured on the set of examples using your
  hypothesis h.

```
We use lowercase italic font for scalar values (such as m or y(i)) and function names
(such as h), lowercase bold font for vectors (such as x (i)), and uppercase bold font for
matrices (such as X ).
```

Even though the RMSE is generally the preferred performance measure for regression
tasks, in some contexts you may prefer to use another function. For example, suppose
that there are many outlier districts. In that case, you may consider using the Mean
Absolute Error (also called the Average Absolute Deviation; see Equation 2-2):

```
Equation 2-2. Mean Absolute Error
```

```
MAE X ,h =m^1 ∑
i= 1
```

```
m
h x i −yi
```

Both the RMSE and the MAE are ways to measure the distance between two vectors:
the vector of predictions and the vector of target values. Various distance measures,
or norms, are possible:

- Computing the root of a sum of squares (RMSE) corresponds to the Euclidean
  norm: it is the notion of distance you are familiar with. It is also called the ℓ 2
  norm, noted ∥ · ∥ 2 (or just ∥ · ∥).
- Computing the sum of absolutes (MAE) corresponds to the ℓ 1 norm, noted ∥ · ∥ 1.
  It is sometimes called the Manhattan norm because it measures the distance
  between two points in a city if you can only travel along orthogonal city blocks.
- More generally, the ℓk norm of a vector **v** containing n elements is defined as

```
∥𝐀∥k= v 0 k+ v 1 k+⋯+vnk
```

```
1
k. ℓ 0 just gives the number of non-zero ele‐
ments in the vector, and ℓ∞ gives the maximum absolute value in the vector.
```

- The higher the norm index, the more it focuses on large values and neglects small
  ones. This is why the RMSE is more sensitive to outliers than the MAE. But when

**44 | Chapter 2: End-to-End Machine Learning Project**

```
5 The latest version of Python 3 is recommended. Python 2.7+ may work too, but it is now deprecated, all major
scientific libraries are dropping support for it, so you should migrate to Python 3 as soon as possible.
```

```
outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs
very well and is generally preferred.
```

#### Check the Assumptions

Lastly, it is good practice to list and verify the assumptions that were made so far (by
you or others); this can catch serious issues early on. For example, the district prices
that your system outputs are going to be fed into a downstream Machine Learning
system, and we assume that these prices are going to be used as such. But what if the
downstream system actually converts the prices into categories (e.g., “cheap,”
“medium,” or “expensive”) and then uses those categories instead of the prices them‐
selves? In this case, getting the price perfectly right is not important at all; your sys‐
tem just needs to get the category right. If that’s so, then the problem should have
been framed as a classification task, not a regression task. You don’t want to find this
out after working on a regression system for months.

Fortunately, after talking with the team in charge of the downstream system, you are
confident that they do indeed need the actual prices, not just categories. Great! You’re
all set, the lights are green, and you can start coding now!

### Get the Data

It’s time to get your hands dirty. Don’t hesitate to pick up your laptop and walk
through the following code examples in a Jupyter notebook. The full Jupyter note‐
book is available at https://github.com/ageron/handson-ml2.

#### Create the Workspace

First you will need to have Python installed. It is probably already installed on your
system. If not, you can get it at https://www.python.org/.^5

Next you need to create a workspace directory for your Machine Learning code and
datasets. Open a terminal and type the following commands (after the $ prompts):

```
$ export ML_PATH="$HOME/ml" # You can change the path if you prefer
$ mkdir -p $ML_PATH
```

You will need a number of Python modules: Jupyter, NumPy, Pandas, Matplotlib, and
Scikit-Learn. If you already have Jupyter running with all these modules installed,
you can safely skip to “Download the Data” on page 49. If you don’t have them yet,
there are many ways to install them (and their dependencies). You can use your sys‐

```
Get the Data | 45
```

```
6 We will show the installation steps using pip in a bash shell on a Linux or MacOS system. You may need to
adapt these commands to your own system. On Windows, we recommend installing Anaconda instead.
7 If you want to upgrade pip for all users on your machine rather than just your own user, you should remove
the --user option and make sure you have administrator rights (e.g., by adding sudo before the whole com‐
mand on Linux or MacOSX).
8 Alternative tools include venv (very similar to virtualenv and included in the standard library), virtualenv‐
wrapper (provides extra functionalities on top of virtualenv), pyenv (allows easy switching between Python
versions), and pipenv (a great packaging tool by the same author as the popular requests library, built on top
of pip, virtualenv and more).
```

tem’s packaging system (e.g., apt-get on Ubuntu, or MacPorts or HomeBrew on
MacOS), install a Scientific Python distribution such as Anaconda and use its packag‐
ing system, or just use Python’s own packaging system, pip, which is included by
default with the Python binary installers (since Python 2.7.9).^6 You can check to see if
pip is installed by typing the following command:

```
$ python3 -m pip --version
pip 19.0.2 from [...]/lib/python3.6/site-packages (python 3.6)
```

You should make sure you have a recent version of pip installed. To upgrade the pip
module, type:^7

```
$ python3 -m pip install --user -U pip
Collecting pip
[...]
Successfully installed pip-19.0.2
```

```
Creating an Isolated Environment
If you would like to work in an isolated environment (which is strongly recom‐
mended so you can work on different projects without having conflicting library ver‐
sions), install virtualenv^8 by running the following pip command (again, if you want
virtualenv to be installed for all users on your machine, remove --user and run this
command with administrator rights):
$ python3 -m pip install --user -U virtualenv
Collecting virtualenv
[...]
Successfully installed virtualenv
Now you can create an isolated Python environment by typing:
$ cd $ML_PATH
$ virtualenv env
Using base prefix '[...]'
New python executable in [...]/ml/env/bin/python3.6
Also creating executable in [...]/ml/env/bin/python
Installing setuptools, pip, wheel...done.
```

**46 | Chapter 2: End-to-End Machine Learning Project**

```
9 Note that Jupyter can handle multiple versions of Python, and even many other languages such as R or
Octave.
```

```
Now every time you want to activate this environment, just open a terminal and type:
$ cd $ML_PATH
$ source env/bin/activate # on Linux or MacOSX
$ .\env\Scripts\activate # on Windows
To deactivate this environment, just type deactivate. While the environment is
active, any package you install using pip will be installed in this isolated environment,
and Python will only have access to these packages (if you also want access to the sys‐
tem’s packages, you should create the environment using virtualenv’s --system-site-
packages option). Check out virtualenv’s documentation for more information.
```

Now you can install all the required modules and their dependencies using this sim‐
ple pip command (if you are not using a virtualenv, you will need the --user option
or administrator rights):

```
$ python3 -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn
Collecting jupyter
Downloading jupyter-1.0.0-py2.py3-none-any.whl
Collecting matplotlib
[...]
```

To check your installation, try to import every module like this:

```
$ python3 -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"
```

There should be no output and no error. Now you can fire up Jupyter by typing:

```
$ jupyter notebook
[I 15:24 NotebookApp] Serving notebooks from local directory: [...]/ml
[I 15:24 NotebookApp] 0 active kernels
[I 15:24 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/
[I 15:24 NotebookApp] Use Control-C to stop this server and shut down all
kernels (twice to skip confirmation).
```

A Jupyter server is now running in your terminal, listening to port 8888. You can visit
this server by opening your web browser to [http://localhost:8888/](http://localhost:8888/) (this usually hap‐
pens automatically when the server starts). You should see your empty workspace
directory (containing only the env directory if you followed the preceding virtualenv
instructions).

Now create a new Python notebook by clicking on the New button and selecting the
appropriate Python version^9 (see Figure 2-3).

This does three things: first, it creates a new notebook file called Untitled.ipynb in
your workspace; second, it starts a Jupyter Python kernel to run this notebook; and

```
Get the Data | 47
```

third, it opens this notebook in a new tab. You should start by renaming this note‐
book to “Housing” (this will automatically rename the file to Housing.ipynb) by click‐
ing Untitled and typing the new name.

Figure 2-3. Your workspace in Jupyter

A notebook contains a list of cells. Each cell can contain executable code or formatted
text. Right now the notebook contains only one empty code cell, labeled “In [1]:”. Try
typing **print("Hello world!")** in the cell, and click on the play button (see
Figure 2-4) or press Shift-Enter. This sends the current cell to this notebook’s Python
kernel, which runs it and returns the output. The result is displayed below the cell,
and since we reached the end of the notebook, a new cell is automatically created. Go
through the User Interface Tour from Jupyter’s Help menu to learn the basics.

Figure 2-4. Hello world Python notebook

**48 | Chapter 2: End-to-End Machine Learning Project**

10 You might also need to check legal constraints, such as private fields that should never be copied to unsafe
datastores.
11 In a real project you would save this code in a Python file, but for now you can just write it in your Jupyter
notebook.

#### Download the Data

```
In typical environments your data would be available in a relational database (or
some other common datastore) and spread across multiple tables/documents/files. To
access it, you would first need to get your credentials and access authorizations,^10 and
familiarize yourself with the data schema. In this project, however, things are much
simpler: you will just download a single compressed file, housing.tgz, which contains a
comma-separated value (CSV) file called housing.csv with all the data.
You could use your web browser to download it, and run tar xzf housing.tgz to
decompress the file and extract the CSV file, but it is preferable to create a small func‐
tion to do that. It is useful in particular if data changes regularly, as it allows you to
write a small script that you can run whenever you need to fetch the latest data (or
you can set up a scheduled job to do that automatically at regular intervals). Auto‐
mating the process of fetching the data is also useful if you need to install the dataset
on multiple machines.
Here is the function to fetch the data:^11
import os
import tarfile
from six.moves import urllib
```

```
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
```

```
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
if not os.path.isdir(housing_path):
os.makedirs(housing_path)
tgz_path = os.path.join(housing_path, "housing.tgz")
urllib.request.urlretrieve(housing_url, tgz_path)
housing_tgz = tarfile.open(tgz_path)
housing_tgz.extractall(path=housing_path)
housing_tgz.close()
Now when you call fetch_housing_data(), it creates a datasets/housing directory in
your workspace, downloads the housing.tgz file, and extracts the housing.csv from it in
this directory.
Now let’s load the data using Pandas. Once again you should write a small function to
load the data:
```

```
Get the Data | 49
```

```
import pandas as pd
```

```
def load_housing_data(housing_path=HOUSING_PATH):
csv_path = os.path.join(housing_path, "housing.csv")
return pd.read_csv(csv_path)
```

This function returns a Pandas DataFrame object containing all the data.

#### Take a Quick Look at the Data Structure

Let’s take a look at the top five rows using the DataFrame’s head() method (see
Figure 2-5).

Figure 2-5. Top five rows in the dataset

Each row represents one district. There are 10 attributes (you can see the first 6 in the
screenshot): longitude, latitude, housing_median_age, total_rooms, total_bed
rooms, population, households, median_income, median_house_value, and
ocean_proximity.

The info() method is useful to get a quick description of the data, in particular the
total number of rows, and each attribute’s type and number of non-null values (see
Figure 2-6).

**50 | Chapter 2: End-to-End Machine Learning Project**

Figure 2-6. Housing info

There are 20,640 instances in the dataset, which means that it is fairly small by
Machine Learning standards, but it’s perfect to get started. Notice that the total_bed
rooms attribute has only 20,433 non-null values, meaning that 207 districts are miss‐
ing this feature. We will need to take care of this later.

All attributes are numerical, except the ocean_proximity field. Its type is object, so it
could hold any kind of Python object, but since you loaded this data from a CSV file
you know that it must be a text attribute. When you looked at the top five rows, you
probably noticed that the values in the ocean_proximity column were repetitive,
which means that it is probably a categorical attribute. You can find out what cate‐
gories exist and how many districts belong to each category by using the
value_counts() method:

```
>>> housing["ocean_proximity"].value_counts()
<1H OCEAN 9136
INLAND 6551
NEAR OCEAN 2658
NEAR BAY 2290
ISLAND 5
Name: ocean_proximity, dtype: int64
```

Let’s look at the other fields. The describe() method shows a summary of the
numerical attributes (Figure 2-7).

```
Get the Data | 51
```

12 The standard deviation is generally denoted σ (the Greek letter sigma), and it is the square root of the var‐
iance, which is the average of the squared deviation from the mean. When a feature has a bell-shaped normal
distribution (also called a Gaussian distribution), which is very common, the “68-95-99.7” rule applies: about
68% of the values fall within 1σ of the mean, 95% within 2σ, and 99.7% within 3σ.

```
Figure 2-7. Summary of each numerical attribute
```

```
The count, mean, min, and max rows are self-explanatory. Note that the null values are
ignored (so, for example, count of total_bedrooms is 20,433, not 20,640). The std
row shows the standard deviation, which measures how dispersed the values are.^12
The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indi‐
cates the value below which a given percentage of observations in a group of observa‐
tions falls. For example, 25% of the districts have a housing_median_age lower than
18, while 50% are lower than 29 and 75% are lower than 37. These are often called the
25 th percentile (or 1st quartile), the median, and the 75th percentile (or 3rd quartile).
Another quick way to get a feel of the type of data you are dealing with is to plot a
histogram for each numerical attribute. A histogram shows the number of instances
(on the vertical axis) that have a given value range (on the horizontal axis). You can
either plot this one attribute at a time, or you can call the hist() method on the
whole dataset, and it will plot a histogram for each numerical attribute (see
Figure 2-8). For example, you can see that slightly over 800 districts have a
median_house_value equal to about $100,000.
%matplotlib inline # only in a Jupyter notebook
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
```

```
52 | Chapter 2: End-to-End Machine Learning Project
```

```
The hist() method relies on Matplotlib, which in turn relies on a
user-specified graphical backend to draw on your screen. So before
you can plot anything, you need to specify which backend Matplot‐
lib should use. The simplest option is to use Jupyter’s magic com‐
mand %matplotlib inline. This tells Jupyter to set up Matplotlib
so it uses Jupyter’s own backend. Plots are then rendered within the
notebook itself. Note that calling show() is optional in a Jupyter
notebook, as Jupyter will automatically display plots when a cell is
executed.
```

Figure 2-8. A histogram for each numerical attribute

Notice a few things in these histograms:

1. First, the median income attribute does not look like it is expressed in US dollars
   (USD). After checking with the team that collected the data, you are told that the
   data has been scaled and capped at 15 (actually 15.0001) for higher median
   incomes, and at 0.5 (actually 0.4999) for lower median incomes. The numbers
   represent roughly tens of thousands of dollars (e.g., 3 actually means about
   $30,000). Working with preprocessed attributes is common in Machine Learning,

```
Get the Data | 53
```

```
and it is not necessarily a problem, but you should try to understand how the
data was computed.
```

2. The housing median age and the median house value were also capped. The lat‐
   ter may be a serious problem since it is your target attribute (your labels). Your
   Machine Learning algorithms may learn that prices never go beyond that limit.
   You need to check with your client team (the team that will use your system’s out‐
   put) to see if this is a problem or not. If they tell you that they need precise pre‐
   dictions even beyond $500,000, then you have mainly two options:
   a.Collect proper labels for the districts whose labels were capped.
   b. Remove those districts from the training set (and also from the test set, since
   your system should not be evaluated poorly if it predicts values beyond
   $500,000).
3. These attributes have very different scales. We will discuss this later in this chap‐
   ter when we explore feature scaling.
4. Finally, many histograms are tail heavy: they extend much farther to the right of
   the median than to the left. This may make it a bit harder for some Machine
   Learning algorithms to detect patterns. We will try transforming these attributes
   later on to have more bell-shaped distributions.

Hopefully you now have a better understanding of the kind of data you are dealing
with.

```
Wait! Before you look at the data any further, you need to create a
test set, put it aside, and never look at it.
```

#### Create a Test Set

It may sound strange to voluntarily set aside part of the data at this stage. After all,
you have only taken a quick glance at the data, and surely you should learn a whole
lot more about it before you decide what algorithms to use, right? This is true, but
your brain is an amazing pattern detection system, which means that it is highly
prone to overfitting: if you look at the test set, you may stumble upon some seemingly
interesting pattern in the test data that leads you to select a particular kind of
Machine Learning model. When you estimate the generalization error using the test
set, your estimate will be too optimistic and you will launch a system that will not
perform as well as expected. This is called data snooping bias.

Creating a test set is theoretically quite simple: just pick some instances randomly,
typically 20% of the dataset (or less if your dataset is very large), and set them aside:

**54 | Chapter 2: End-to-End Machine Learning Project**

13 In this book, when a code example contains a mix of code and outputs, as is the case here, it is formatted like
in the Python interpreter, for better readability: the code lines are prefixed with >>> (or ... for indented
blocks), and the outputs have no prefix.
14 You will often see people set the random seed to 42. This number has no special property, other than to be
The Answer to the Ultimate Question of Life, the Universe, and Everything.

```
import numpy as np
```

```
def split_train_test(data, test_ratio):
shuffled_indices = np.random.permutation(len(data))
test_set_size = int(len(data) * test_ratio)
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]
return data.iloc[train_indices], data.iloc[test_indices]
You can then use this function like this:^13
>>> train_set, test_set = split_train_test(housing, 0.2)
>>> len(train_set)
16512
>>> len(test_set)
4128
Well, this works, but it is not perfect: if you run the program again, it will generate a
different test set! Over time, you (or your Machine Learning algorithms) will get to
see the whole dataset, which is what you want to avoid.
One solution is to save the test set on the first run and then load it in subsequent
runs. Another option is to set the random number generator’s seed (e.g., np.ran
dom.seed(42))^14 before calling np.random.permutation(), so that it always generates
the same shuffled indices.
But both these solutions will break next time you fetch an updated dataset. A com‐
mon solution is to use each instance’s identifier to decide whether or not it should go
in the test set (assuming instances have a unique and immutable identifier). For
example, you could compute a hash of each instance’s identifier and put that instance
in the test set if the hash is lower or equal to 20% of the maximum hash value. This
ensures that the test set will remain consistent across multiple runs, even if you
refresh the dataset. The new test set will contain 20% of the new instances, but it will
not contain any instance that was previously in the training set. Here is a possible
implementation:
from zlib import crc32
```

```
def test_set_check(identifier, test_ratio):
return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
```

```
def split_train_test_by_id(data, test_ratio, id_column):
ids = data[id_column]
```

```
Get the Data | 55
```

15 The location information is actually quite coarse, and as a result many districts will have the exact same ID, so
they will end up in the same set (test or train). This introduces some unfortunate sampling bias.

```
in_test_set = ids.apply( lambda id_: test_set_check(id_, test_ratio))
return data.loc[~in_test_set], data.loc[in_test_set]
Unfortunately, the housing dataset does not have an identifier column. The simplest
solution is to use the row index as the ID:
housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
If you use the row index as a unique identifier, you need to make sure that new data
gets appended to the end of the dataset, and no row ever gets deleted. If this is not
possible, then you can try to use the most stable features to build a unique identifier.
For example, a district’s latitude and longitude are guaranteed to be stable for a few
million years, so you could combine them into an ID like so:^15
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
Scikit-Learn provides a few functions to split datasets into multiple subsets in various
ways. The simplest function is train_test_split, which does pretty much the same
thing as the function split_train_test defined earlier, with a couple of additional
features. First there is a random_state parameter that allows you to set the random
generator seed as explained previously, and second you can pass it multiple datasets
with an identical number of rows, and it will split them on the same indices (this is
very useful, for example, if you have a separate DataFrame for labels):
from sklearn.model_selection import train_test_split
```

```
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
So far we have considered purely random sampling methods. This is generally fine if
your dataset is large enough (especially relative to the number of attributes), but if it
is not, you run the risk of introducing a significant sampling bias. When a survey
company decides to call 1,000 people to ask them a few questions, they don’t just pick
1,000 people randomly in a phone book. They try to ensure that these 1,000 people
are representative of the whole population. For example, the US population is com‐
posed of 51.3% female and 48.7% male, so a well-conducted survey in the US would
try to maintain this ratio in the sample: 513 female and 487 male. This is called strati‐
fied sampling: the population is divided into homogeneous subgroups called strata,
and the right number of instances is sampled from each stratum to guarantee that the
test set is representative of the overall population. If they used purely random sam‐
pling, there would be about 12% chance of sampling a skewed test set with either less
than 49% female or more than 54% female. Either way, the survey results would be
significantly biased.
```

```
56 | Chapter 2: End-to-End Machine Learning Project
```

Suppose you chatted with experts who told you that the median income is a very
important attribute to predict median housing prices. You may want to ensure that
the test set is representative of the various categories of incomes in the whole dataset.
Since the median income is a continuous numerical attribute, you first need to create
an income category attribute. Let’s look at the median income histogram more closely
(back in Figure 2-8): most median income values are clustered around 1.5 to 6 (i.e.,
$15,000–$60,000), but some median incomes go far beyond 6. It is important to have
a sufficient number of instances in your dataset for each stratum, or else the estimate
of the stratum’s importance may be biased. This means that you should not have too
many strata, and each stratum should be large enough. The following code uses the
pd.cut() function to create an income category attribute with 5 categories (labeled
from 1 to 5): category 1 ranges from 0 to 1.5 (i.e., less than $15,000), category 2 from
1.5 to 3, and so on:

```
housing["income_cat"] = pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])
```

These income categories are represented in Figure 2-9:

```
housing["income_cat"].hist()
```

Figure 2-9. Histogram of income categories

Now you are ready to do stratified sampling based on the income category. For this
you can use Scikit-Learn’s StratifiedShuffleSplit class:

```
from sklearn.model_selection import StratifiedShuffleSplit
```

```
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
strat_train_set = housing.loc[train_index]
strat_test_set = housing.loc[test_index]
```

```
Get the Data | 57
```

Let’s see if this worked as expected. You can start by looking at the income category
proportions in the test set:

```
>>> strat_test_set["income_cat"].value_counts() / len(strat_test_set)
3 0.350533
2 0.318798
4 0.176357
5 0.114583
1 0.039729
Name: income_cat, dtype: float64
```

With similar code you can measure the income category proportions in the full data‐
set. Figure 2-10 compares the income category proportions in the overall dataset, in
the test set generated with stratified sampling, and in a test set generated using purely
random sampling. As you can see, the test set generated using stratified sampling has
income category proportions almost identical to those in the full dataset, whereas the
test set generated using purely random sampling is quite skewed.

Figure 2-10. Sampling bias comparison of stratified versus purely random sampling

Now you should remove the income_cat attribute so the data is back to its original
state:

```
for set_ in (strat_train_set, strat_test_set):
set_.drop("income_cat", axis=1, inplace=True)
```

We spent quite a bit of time on test set generation for a good reason: this is an often
neglected but critical part of a Machine Learning project. Moreover, many of these
ideas will be useful later when we discuss cross-validation. Now it’s time to move on
to the next stage: exploring the data.

### Discover and Visualize the Data to Gain Insights

So far you have only taken a quick glance at the data to get a general understanding of
the kind of data you are manipulating. Now the goal is to go a little bit more in depth.

First, make sure you have put the test set aside and you are only exploring the train‐
ing set. Also, if the training set is very large, you may want to sample an exploration

**58 | Chapter 2: End-to-End Machine Learning Project**

set, to make manipulations easy and fast. In our case, the set is quite small so you can
just work directly on the full set. Let’s create a copy so you can play with it without
harming the training set:

```
housing = strat_train_set.copy()
```

#### Visualizing Geographical Data

Since there is geographical information (latitude and longitude), it is a good idea to
create a scatterplot of all districts to visualize the data (Figure 2-11):

```
housing.plot(kind="scatter", x="longitude", y="latitude")
```

Figure 2-11. A geographical scatterplot of the data

This looks like California all right, but other than that it is hard to see any particular
pattern. Setting the alpha option to 0.1 makes it much easier to visualize the places
where there is a high density of data points (Figure 2-12):

```
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```

```
Discover and Visualize the Data to Gain Insights | 59
```

16 If you are reading this in grayscale, grab a red pen and scribble over most of the coastline from the Bay Area
down to San Diego (as you might expect). You can add a patch of yellow around Sacramento as well.

```
Figure 2-12. A better visualization highlighting high-density areas
```

```
Now that’s much better: you can clearly see the high-density areas, namely the Bay
Area and around Los Angeles and San Diego, plus a long line of fairly high density in
the Central Valley, in particular around Sacramento and Fresno.
More generally, our brains are very good at spotting patterns on pictures, but you
may need to play around with visualization parameters to make the patterns stand
out.
Now let’s look at the housing prices (Figure 2-13). The radius of each circle represents
the district’s population (option s), and the color represents the price (option c). We
will use a predefined color map (option cmap) called jet, which ranges from blue
(low values) to red (high prices):^16
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
```

```
60 | Chapter 2: End-to-End Machine Learning Project
```

Figure 2-13. California housing prices

```
Discover and Visualize the Data to Gain Insights | 61
```

This image tells you that the housing prices are very much related to the location
(e.g., close to the ocean) and to the population density, as you probably knew already.
It will probably be useful to use a clustering algorithm to detect the main clusters, and
add new features that measure the proximity to the cluster centers. The ocean prox‐
imity attribute may be useful as well, although in Northern California the housing
prices in coastal districts are not too high, so it is not a simple rule.

#### Looking for Correlations

Since the dataset is not too large, you can easily compute the standard correlation
coefficient (also called Pearson’s r) between every pair of attributes using the corr()
method:

```
corr_matrix = housing.corr()
```

Now let’s look at how much each attribute correlates with the median house value:

```
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value 1.000000
median_income 0.687170
total_rooms 0.135231
housing_median_age 0.114220
households 0.064702
total_bedrooms 0.047865
population -0.026699
longitude -0.047279
latitude -0.142826
Name: median_house_value, dtype: float64
```

The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that
there is a strong positive correlation; for example, the median house value tends to go
up when the median income goes up. When the coefficient is close to –1, it means
that there is a strong negative correlation; you can see a small negative correlation
between the latitude and the median house value (i.e., prices have a slight tendency to
go down when you go north). Finally, coefficients close to zero mean that there is no
linear correlation. Figure 2-14 shows various plots along with the correlation coeffi‐
cient between their horizontal and vertical axes.

**62 | Chapter 2: End-to-End Machine Learning Project**

Figure 2-14. Standard correlation coefficient of various datasets (source: Wikipedia;
public domain image)

```
The correlation coefficient only measures linear correlations (“if x
goes up, then y generally goes up/down”). It may completely miss
out on nonlinear relationships (e.g., “if x is close to zero then y gen‐
erally goes up”). Note how all the plots of the bottom row have a
correlation coefficient equal to zero despite the fact that their axes
are clearly not independent: these are examples of nonlinear rela‐
tionships. Also, the second row shows examples where the correla‐
tion coefficient is equal to 1 or –1; notice that this has nothing to
do with the slope. For example, your height in inches has a correla‐
tion coefficient of 1 with your height in feet or in nanometers.
```

Another way to check for correlation between attributes is to use Pandas’
scatter_matrix function, which plots every numerical attribute against every other
numerical attribute. Since there are now 11 numerical attributes, you would get 11^2 =
121 plots, which would not fit on a page, so let’s just focus on a few promising
attributes that seem most correlated with the median housing value (Figure 2-15):

```
from pandas.plotting import scatter_matrix
```

```
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```

```
Discover and Visualize the Data to Gain Insights | 63
```

Figure 2-15. Scatter matrix

The main diagonal (top left to bottom right) would be full of straight lines if Pandas
plotted each variable against itself, which would not be very useful. So instead Pandas
displays a histogram of each attribute (other options are available; see Pandas’ docu‐
mentation for more details).

The most promising attribute to predict the median house value is the median
income, so let’s zoom in on their correlation scatterplot (Figure 2-16):

```
housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)
```

This plot reveals a few things. First, the correlation is indeed very strong; you can
clearly see the upward trend and the points are not too dispersed. Second, the price
cap that we noticed earlier is clearly visible as a horizontal line at $500,000. But this
plot reveals other less obvious straight lines: a horizontal line around $450,000,
another around $350,000, perhaps one around $280,000, and a few more below that.
You may want to try removing the corresponding districts to prevent your algorithms
from learning to reproduce these data quirks.

**64 | Chapter 2: End-to-End Machine Learning Project**

Figure 2-16. Median income versus median house value

#### Experimenting with Attribute Combinations

Hopefully the previous sections gave you an idea of a few ways you can explore the
data and gain insights. You identified a few data quirks that you may want to clean up
before feeding the data to a Machine Learning algorithm, and you found interesting
correlations between attributes, in particular with the target attribute. You also
noticed that some attributes have a tail-heavy distribution, so you may want to trans‐
form them (e.g., by computing their logarithm). Of course, your mileage will vary
considerably with each project, but the general ideas are similar.

One last thing you may want to do before actually preparing the data for Machine
Learning algorithms is to try out various attribute combinations. For example, the
total number of rooms in a district is not very useful if you don’t know how many
households there are. What you really want is the number of rooms per household.
Similarly, the total number of bedrooms by itself is not very useful: you probably
want to compare it to the number of rooms. And the population per household also
seems like an interesting attribute combination to look at. Let’s create these new
attributes:

```
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```

And now let’s look at the correlation matrix again:

```
>>> corr_matrix = housing.corr()
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value 1.000000
```

```
Discover and Visualize the Data to Gain Insights | 65
```

```
median_income 0.687160
rooms_per_household 0.146285
total_rooms 0.135097
housing_median_age 0.114110
households 0.064506
total_bedrooms 0.047689
population_per_household -0.021985
population -0.026920
longitude -0.047432
latitude -0.142724
bedrooms_per_room -0.259984
Name: median_house_value, dtype: float64
```

Hey, not bad! The new bedrooms_per_room attribute is much more correlated with
the median house value than the total number of rooms or bedrooms. Apparently
houses with a lower bedroom/room ratio tend to be more expensive. The number of
rooms per household is also more informative than the total number of rooms in a
district—obviously the larger the houses, the more expensive they are.

This round of exploration does not have to be absolutely thorough; the point is to
start off on the right foot and quickly gain insights that will help you get a first rea‐
sonably good prototype. But this is an iterative process: once you get a prototype up
and running, you can analyze its output to gain more insights and come back to this
exploration step.

### Prepare the Data for Machine Learning Algorithms

It’s time to prepare the data for your Machine Learning algorithms. Instead of just
doing this manually, you should write functions to do that, for several good reasons:

- This will allow you to reproduce these transformations easily on any dataset (e.g.,
  the next time you get a fresh dataset).
- You will gradually build a library of transformation functions that you can reuse
  in future projects.
- You can use these functions in your live system to transform the new data before
  feeding it to your algorithms.
- This will make it possible for you to easily try various transformations and see
  which combination of transformations works best.

But first let’s revert to a clean training set (by copying strat_train_set once again),
and let’s separate the predictors and the labels since we don’t necessarily want to apply
the same transformations to the predictors and the target values (note that drop()
creates a copy of the data and does not affect strat_train_set):

```
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

**66 | Chapter 2: End-to-End Machine Learning Project**

#### Data Cleaning

Most Machine Learning algorithms cannot work with missing features, so let’s create
a few functions to take care of them. You noticed earlier that the total_bedrooms
attribute has some missing values, so let’s fix this. You have three options:

- Get rid of the corresponding districts.
- Get rid of the whole attribute.
- Set the values to some value (zero, the mean, the median, etc.).

You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna()
methods:

```
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)
```

If you choose option 3, you should compute the median value on the training set, and
use it to fill the missing values in the training set, but also don’t forget to save the
median value that you have computed. You will need it later to replace missing values
in the test set when you want to evaluate your system, and also once the system goes
live to replace missing values in new data.

Scikit-Learn provides a handy class to take care of missing values: SimpleImputer.
Here is how to use it. First, you need to create a SimpleImputer instance, specifying
that you want to replace each attribute’s missing values with the median of that
attribute:

```
from sklearn.impute import SimpleImputer
```

```
imputer = SimpleImputer(strategy="median")
```

Since the median can only be computed on numerical attributes, we need to create a
copy of the data without the text attribute ocean_proximity:

```
housing_num = housing.drop("ocean_proximity", axis=1)
```

Now you can fit the imputer instance to the training data using the fit() method:

```
imputer.fit(housing_num)
```

The imputer has simply computed the median of each attribute and stored the result
in its statistics\_ instance variable. Only the total_bedrooms attribute had missing
values, but we cannot be sure that there won’t be any missing values in new data after
the system goes live, so it is safer to apply the imputer to all the numerical attributes:

```
>>> imputer.statistics_
array([ -118.51 , 34.26 , 29. , 2119.5 , 433. , 1164. , 408. , 3.5409])
```

```
Prepare the Data for Machine Learning Algorithms | 67
```

17 For more details on the design principles, see “API design for machine learning software: experiences from
the scikit-learn project,” L. Buitinck, G. Louppe, M. Blondel, F. Pedregosa, A. Müller, et al. (2013).

```
>>> housing_num.median().values
array([ -118.51 , 34.26 , 29. , 2119.5 , 433. , 1164. , 408. , 3.5409])
Now you can use this “trained” imputer to transform the training set by replacing
missing values by the learned medians:
X = imputer.transform(housing_num)
The result is a plain NumPy array containing the transformed features. If you want to
put it back into a Pandas DataFrame, it’s simple:
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
```

```
Scikit-Learn Design
Scikit-Learn’s API is remarkably well designed. The main design principles are:^17
```

- **Consistency**. All objects share a consistent and simple interface:
  —Estimators. Any object that can estimate some parameters based on a dataset
  is called an estimator (e.g., an imputer is an estimator). The estimation itself is
  performed by the fit() method, and it takes only a dataset as a parameter (or
  two for supervised learning algorithms; the second dataset contains the
  labels). Any other parameter needed to guide the estimation process is con‐
  sidered a hyperparameter (such as an imputer’s strategy), and it must be set
  as an instance variable (generally via a constructor parameter).
  —Transformers. Some estimators (such as an imputer) can also transform a
  dataset; these are called transformers. Once again, the API is quite simple: the
  transformation is performed by the transform() method with the dataset to
  transform as a parameter. It returns the transformed dataset. This transforma‐
  tion generally relies on the learned parameters, as is the case for an imputer.
  All transformers also have a convenience method called fit_transform()
  that is equivalent to calling fit() and then transform() (but sometimes
  fit_transform() is optimized and runs much faster).
  —Predictors. Finally, some estimators are capable of making predictions given a
  dataset; they are called predictors. For example, the LinearRegression model
  in the previous chapter was a predictor: it predicted life satisfaction given a
  country’s GDP per capita. A predictor has a predict() method that takes a
  dataset of new instances and returns a dataset of corresponding predictions. It
  also has a score() method that measures the quality of the predictions given

```
68 | Chapter 2: End-to-End Machine Learning Project
```

18 Some predictors also provide methods to measure the confidence of their predictions.
19 This class is available since Scikit-Learn 0.20. If you use an earlier version, please consider upgrading, or use
Pandas’ Series.factorize() method.

```
a test set (and the corresponding labels in the case of supervised learning
algorithms).^18
```

- **Inspection**. All the estimator’s hyperparameters are accessible directly via public
  instance variables (e.g., imputer.strategy), and all the estimator’s learned
  parameters are also accessible via public instance variables with an underscore
  suffix (e.g., imputer.statistics\_).
- **Nonproliferation of classes**. Datasets are represented as NumPy arrays or SciPy
  sparse matrices, instead of homemade classes. Hyperparameters are just regular
  Python strings or numbers.
- **Composition**. Existing building blocks are reused as much as possible. For
  example, it is easy to create a Pipeline estimator from an arbitrary sequence of
  transformers followed by a final estimator, as we will see.
- **Sensible defaults**. Scikit-Learn provides reasonable default values for most
  parameters, making it easy to create a baseline working system quickly.

#### Handling Text and Categorical Attributes

```
Earlier we left out the categorical attribute ocean_proximity because it is a text
attribute so we cannot compute its median:
>>> housing_cat = housing[["ocean_proximity"]]
>>> housing_cat.head(10)
ocean_proximity
17606 <1H OCEAN
18632 <1H OCEAN
14650 NEAR OCEAN
3230 INLAND
3555 <1H OCEAN
19480 INLAND
8879 <1H OCEAN
13685 INLAND
4937 <1H OCEAN
4861 <1H OCEAN
Most Machine Learning algorithms prefer to work with numbers anyway, so let’s con‐
vert these categories from text to numbers. For this, we can use Scikit-Learn’s Ordina
lEncoder class^19 :
>>> from sklearn.preprocessing import OrdinalEncoder
>>> ordinal_encoder = OrdinalEncoder()
```

```
Prepare the Data for Machine Learning Algorithms | 69
```

20 Before Scikit-Learn 0.20, it could only encode integer categorical values, but since 0.20 it can also handle
other types of inputs, including text categorical inputs.

```
>>> housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
>>> housing_cat_encoded[:10]
array([[0.],
[0.],
[4.],
[1.],
[0.],
[1.],
[0.],
[1.],
[0.],
[0.]])
You can get the list of categories using the categories_ instance variable. It is a list
containing a 1D array of categories for each categorical attribute (in this case, a list
containing a single array since there is just one categorical attribute):
>>> ordinal_encoder.categories_
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
dtype=object)]
One issue with this representation is that ML algorithms will assume that two nearby
values are more similar than two distant values. This may be fine in some cases (e.g.,
for ordered categories such as “bad”, “average”, “good”, “excellent”), but it is obviously
not the case for the ocean_proximity column (for example, categories 0 and 4 are
clearly more similar than categories 0 and 1). To fix this issue, a common solution is
to create one binary attribute per category: one attribute equal to 1 when the category
is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is
“INLAND” (and 0 otherwise), and so on. This is called one-hot encoding, because
only one attribute will be equal to 1 (hot), while the others will be 0 (cold). The new
attributes are sometimes called dummy attributes. Scikit-Learn provides a OneHotEn
coder class to convert categorical values into one-hot vectors^20 :
>>> from sklearn.preprocessing import OneHotEncoder
>>> cat_encoder = OneHotEncoder()
>>> housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
>>> housing_cat_1hot
<16512x5 sparse matrix of type '<class 'numpy.float64'>'
with 16512 stored elements in Compressed Sparse Row format>
Notice that the output is a SciPy sparse matrix, instead of a NumPy array. This is very
useful when you have categorical attributes with thousands of categories. After one-
hot encoding we get a matrix with thousands of columns, and the matrix is full of
zeros except for a single 1 per row. Using up tons of memory mostly to store zeros
would be very wasteful, so instead a sparse matrix only stores the location of the non‐
```

```
70 | Chapter 2: End-to-End Machine Learning Project
```

21 See SciPy’s documentation for more details.

```
zero elements. You can use it mostly like a normal 2D array,^21 but if you really want to
convert it to a (dense) NumPy array, just call the toarray() method:
>>> housing_cat_1hot.toarray()
array([[1., 0., 0., 0., 0.],
[1., 0., 0., 0., 0.],
[0., 0., 0., 0., 1.],
...,
[0., 1., 0., 0., 0.],
[1., 0., 0., 0., 0.],
[0., 0., 0., 1., 0.]])
Once again, you can get the list of categories using the encoder’s categories_
instance variable:
>>> cat_encoder.categories_
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
dtype=object)]
```

```
If a categorical attribute has a large number of possible categories
(e.g., country code, profession, species, etc.), then one-hot encod‐
ing will result in a large number of input features. This may slow
down training and degrade performance. If this happens, you may
want to replace the categorical input with useful numerical features
related to the categories: for example, you could replace the
ocean_proximity feature with the distance to the ocean (similarly,
a country code could be replaced with the country’s population and
GDP per capita). Alternatively, you could replace each category
with a learnable low dimensional vector called an embedding. Each
category’s representation would be learned during training: this is
an example of representation learning (see Chapter 13 and ??? for
more details).
```

#### Custom Transformers

```
Although Scikit-Learn provides many useful transformers, you will need to write
your own for tasks such as custom cleanup operations or combining specific
attributes. You will want your transformer to work seamlessly with Scikit-Learn func‐
tionalities (such as pipelines), and since Scikit-Learn relies on duck typing (not inher‐
itance), all you need is to create a class and implement three methods: fit()
(returning self), transform(), and fit_transform(). You can get the last one for
free by simply adding TransformerMixin as a base class. Also, if you add BaseEstima
tor as a base class (and avoid *args and **kargs in your constructor) you will get
two extra methods (get_params() and set_params()) that will be useful for auto‐
```

```
Prepare the Data for Machine Learning Algorithms | 71
```

matic hyperparameter tuning. For example, here is a small transformer class that adds
the combined attributes we discussed earlier:

```
from sklearn.base import BaseEstimator, TransformerMixin
```

```
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
```

```
class CombinedAttributesAdder (BaseEstimator, TransformerMixin):
def __init__ (self, add_bedrooms_per_room = True): # no *args or **kargs
self.add_bedrooms_per_room = add_bedrooms_per_room
def fit(self, X, y=None):
return self # nothing else to do
def transform(self, X, y=None):
rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
population_per_household = X[:, population_ix] / X[:, households_ix]
if self.add_bedrooms_per_room:
bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
return np.c_[X, rooms_per_household, population_per_household,
bedrooms_per_room]
else :
return np.c_[X, rooms_per_household, population_per_household]
```

```
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

In this example the transformer has one hyperparameter, add_bedrooms_per_room,
set to True by default (it is often helpful to provide sensible defaults). This hyperpara‐
meter will allow you to easily find out whether adding this attribute helps the
Machine Learning algorithms or not. More generally, you can add a hyperparameter
to gate any data preparation step that you are not 100% sure about. The more you
automate these data preparation steps, the more combinations you can automatically
try out, making it much more likely that you will find a great combination (and sav‐
ing you a lot of time).

#### Feature Scaling

One of the most important transformations you need to apply to your data is feature
scaling. With few exceptions, Machine Learning algorithms don’t perform well when
the input numerical attributes have very different scales. This is the case for the hous‐
ing data: the total number of rooms ranges from about 6 to 39,320, while the median
incomes only range from 0 to 15. Note that scaling the target values is generally not
required.

There are two common ways to get all attributes to have the same scale: min-max
scaling and standardization.

Min-max scaling (many people call this normalization) is quite simple: values are
shifted and rescaled so that they end up ranging from 0 to 1. We do this by subtract‐
ing the min value and dividing by the max minus the min. Scikit-Learn provides a

**72 | Chapter 2: End-to-End Machine Learning Project**

transformer called MinMaxScaler for this. It has a feature_range hyperparameter
that lets you change the range if you don’t want 0–1 for some reason.

Standardization is quite different: first it subtracts the mean value (so standardized
values always have a zero mean), and then it divides by the standard deviation so that
the resulting distribution has unit variance. Unlike min-max scaling, standardization
does not bound values to a specific range, which may be a problem for some algo‐
rithms (e.g., neural networks often expect an input value ranging from 0 to 1). How‐
ever, standardization is much less affected by outliers. For example, suppose a district
had a median income equal to 100 (by mistake). Min-max scaling would then crush
all the other values from 0–15 down to 0–0.15, whereas standardization would not be
much affected. Scikit-Learn provides a transformer called StandardScaler for stand‐
ardization.

```
As with all the transformations, it is important to fit the scalers to
the training data only, not to the full dataset (including the test set).
Only then can you use them to transform the training set and the
test set (and new data).
```

#### Transformation Pipelines

As you can see, there are many data transformation steps that need to be executed in
the right order. Fortunately, Scikit-Learn provides the Pipeline class to help with
such sequences of transformations. Here is a small pipeline for the numerical
attributes:

```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

```
num_pipeline = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('attribs_adder', CombinedAttributesAdder()),
('std_scaler', StandardScaler()),
])
```

```
housing_num_tr = num_pipeline.fit_transform(housing_num)
```

The Pipeline constructor takes a list of name/estimator pairs defining a sequence of
steps. All but the last estimator must be transformers (i.e., they must have a
fit_transform() method). The names can be anything you like (as long as they are
unique and don’t contain double underscores “\_\_”): they will come in handy later for
hyperparameter tuning.

When you call the pipeline’s fit() method, it calls fit_transform() sequentially on
all transformers, passing the output of each call as the parameter to the next call, until
it reaches the final estimator, for which it just calls the fit() method.

```
Prepare the Data for Machine Learning Algorithms | 73
```

22 Just like for pipelines, the name can be anything as long as it does not contain double underscores.

```
The pipeline exposes the same methods as the final estimator. In this example, the last
estimator is a StandardScaler, which is a transformer, so the pipeline has a trans
form() method that applies all the transforms to the data in sequence (and of course
also a fit_transform() method, which is the one we used).
So far, we have handled the categorical columns and the numerical columns sepa‐
rately. It would be more convenient to have a single transformer able to handle all col‐
umns, applying the appropriate transformations to each column. In version 0.20,
Scikit-Learn introduced the ColumnTransformer for this purpose, and the good news
is that it works great with Pandas DataFrames. Let’s use it to apply all the transforma‐
tions to the housing data:
from sklearn.compose import ColumnTransformer
```

```
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
```

```
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])
```

```
housing_prepared = full_pipeline.fit_transform(housing)
Here is how this works: first we import the ColumnTransformer class, next we get the
list of numerical column names and the list of categorical column names, and we
construct a ColumnTransformer. The constructor requires a list of tuples, where each
tuple contains a name^22 , a transformer and a list of names (or indices) of columns
that the transformer should be applied to. In this example, we specify that the numer‐
ical columns should be transformed using the num_pipeline that we defined earlier,
and the categorical columns should be transformed using a OneHotEncoder. Finally,
we apply this ColumnTransformer to the housing data: it applies each transformer to
the appropriate columns and concatenates the outputs along the second axis (the
transformers must return the same number of rows).
Note that the OneHotEncoder returns a sparse matrix, while the num_pipeline returns
a dense matrix. When there is such a mix of sparse and dense matrices, the Colum
nTransformer estimates the density of the final matrix (i.e., the ratio of non-zero
cells), and it returns a sparse matrix if the density is lower than a given threshold (by
default, sparse_threshold=0.3). In this example, it returns a dense matrix. And
that’s it! We have a preprocessing pipeline that takes the full housing data and applies
the appropriate transformations to each column.
```

```
74 | Chapter 2: End-to-End Machine Learning Project
```

```
Instead of a transformer, you can specify the string "drop" if you
want the columns to be dropped. Or you can specify "pass
through" if you want the columns to be left untouched. By default,
the remaining columns (i.e., the ones that were not listed) will be
dropped, but you can set the remainder hyperparameter to any
transformer (or to "passthrough") if you want these columns to be
handled differently.
```

If you are using Scikit-Learn 0.19 or earlier, you can use a third-party library such as
sklearn-pandas, or roll out your own custom transformer to get the same function‐
ality as the ColumnTransformer. Alternatively, you can use the FeatureUnion class
which can also apply different transformers and concatenate their outputs, but you
cannot specify different columns for each transformer, they all apply to the whole
data. It is possible to work around this limitation using a custom transformer for col‐
umn selection (see the Jupyter notebook for an example).

### Select and Train a Model

At last! You framed the problem, you got the data and explored it, you sampled a
training set and a test set, and you wrote transformation pipelines to clean up and
prepare your data for Machine Learning algorithms automatically. You are now ready
to select and train a Machine Learning model.

#### Training and Evaluating on the Training Set

The good news is that thanks to all these previous steps, things are now going to be
much simpler than you might think. Let’s first train a Linear Regression model, like
we did in the previous chapter:

```
from sklearn.linear_model import LinearRegression
```

```
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

Done! You now have a working Linear Regression model. Let’s try it out on a few
instances from the training set:

```
>>> some_data = housing.iloc[:5]
>>> some_labels = housing_labels.iloc[:5]
>>> some_data_prepared = full_pipeline.transform(some_data)
>>> print ("Predictions:", lin_reg.predict(some_data_prepared))
Predictions: [ 210644.6045 317768.8069 210956.4333 59218.9888 189747.5584]
>>> print ("Labels:", list(some_labels))
Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
```

```
Select and Train a Model | 75
```

It works, although the predictions are not exactly accurate (e.g., the first prediction is
off by close to 40%!). Let’s measure this regression model’s RMSE on the whole train‐
ing set using Scikit-Learn’s mean_squared_error function:

```
>>> from sklearn.metrics import mean_squared_error
>>> housing_predictions = lin_reg.predict(housing_prepared)
>>> lin_mse = mean_squared_error(housing_labels, housing_predictions)
>>> lin_rmse = np.sqrt(lin_mse)
>>> lin_rmse
68628.19819848922
```

Okay, this is better than nothing but clearly not a great score: most districts’
median_housing_values range between $120,000 and $265,000, so a typical predic‐
tion error of $68,628 is not very satisfying. This is an example of a model underfitting
the training data. When this happens it can mean that the features do not provide
enough information to make good predictions, or that the model is not powerful
enough. As we saw in the previous chapter, the main ways to fix underfitting are to
select a more powerful model, to feed the training algorithm with better features, or
to reduce the constraints on the model. This model is not regularized, so this rules
out the last option. You could try to add more features (e.g., the log of the popula‐
tion), but first let’s try a more complex model to see how it does.

Let’s train a DecisionTreeRegressor. This is a powerful model, capable of finding
complex nonlinear relationships in the data (Decision Trees are presented in more
detail in Chapter 6). The code should look familiar by now:

```
from sklearn.tree import DecisionTreeRegressor
```

```
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
```

Now that the model is trained, let’s evaluate it on the training set:

```
>>> housing_predictions = tree_reg.predict(housing_prepared)
>>> tree_mse = mean_squared_error(housing_labels, housing_predictions)
>>> tree_rmse = np.sqrt(tree_mse)
>>> tree_rmse
0.0
```

Wait, what!? No error at all? Could this model really be absolutely perfect? Of course,
it is much more likely that the model has badly overfit the data. How can you be sure?
As we saw earlier, you don’t want to touch the test set until you are ready to launch a
model you are confident about, so you need to use part of the training set for train‐
ing, and part for model validation.

#### Better Evaluation Using Cross-Validation

One way to evaluate the Decision Tree model would be to use the train_test_split
function to split the training set into a smaller training set and a validation set, then

**76 | Chapter 2: End-to-End Machine Learning Project**

train your models against the smaller training set and evaluate them against the vali‐
dation set. It’s a bit of work, but nothing too difficult and it would work fairly well.

A great alternative is to use Scikit-Learn’s K-fold cross-validation feature. The follow‐
ing code randomly splits the training set into 10 distinct subsets called folds, then it
trains and evaluates the Decision Tree model 10 times, picking a different fold for
evaluation every time and training on the other 9 folds. The result is an array con‐
taining the 10 evaluation scores:

```
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```

```
Scikit-Learn’s cross-validation features expect a utility function
(greater is better) rather than a cost function (lower is better), so
the scoring function is actually the opposite of the MSE (i.e., a neg‐
ative value), which is why the preceding code computes -scores
before calculating the square root.
```

Let’s look at the results:

```
>>> def display_scores(scores):
... print ("Scores:", scores)
... print ("Mean:", scores.mean())
... print ("Standard deviation:", scores.std())
...
>>> display_scores(tree_rmse_scores)
Scores: [70194.33680785 66855.16363941 72432.58244769 70758.73896782
71115.88230639 75585.14172901 70262.86139133 70273.6325285
75366.87952553 71231.65726027]
Mean: 71407.68766037929
Standard deviation: 2439.4345041191004
```

Now the Decision Tree doesn’t look as good as it did earlier. In fact, it seems to per‐
form worse than the Linear Regression model! Notice that cross-validation allows
you to get not only an estimate of the performance of your model, but also a measure
of how precise this estimate is (i.e., its standard deviation). The Decision Tree has a
score of approximately 71,407, generally ±2,439. You would not have this information
if you just used one validation set. But cross-validation comes at the cost of training
the model several times, so it is not always possible.

Let’s compute the same scores for the Linear Regression model just to be sure:

```
>>> lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
... scoring="neg_mean_squared_error", cv=10)
...
>>> lin_rmse_scores = np.sqrt(-lin_scores)
>>> display_scores(lin_rmse_scores)
```

```
Select and Train a Model | 77
```

```
Scores: [66782.73843989 66960.118071 70347.95244419 74739.57052552
68031.13388938 71193.84183426 64969.63056405 68281.61137997
71552.91566558 67665.10082067]
Mean: 69052.46136345083
Standard deviation: 2731.674001798348
```

That’s right: the Decision Tree model is overfitting so badly that it performs worse
than the Linear Regression model.

Let’s try one last model now: the RandomForestRegressor. As we will see in Chap‐
ter 7, Random Forests work by training many Decision Trees on random subsets of
the features, then averaging out their predictions. Building a model on top of many
other models is called Ensemble Learning, and it is often a great way to push ML algo‐
rithms even further. We will skip most of the code since it is essentially the same as
for the other models:

```
>>> from sklearn.ensemble import RandomForestRegressor
>>> forest_reg = RandomForestRegressor()
>>> forest_reg.fit(housing_prepared, housing_labels)
>>> [...]
>>> forest_rmse
18603.515021376355
>>> display_scores(forest_rmse_scores)
Scores: [49519.80364233 47461.9115823 50029.02762854 52325.28068953
49308.39426421 53446.37892622 48634.8036574 47585.73832311
53490.10699751 50021.5852922 ]
Mean: 50182.303100336096
Standard deviation: 2097.0810550985693
```

Wow, this is much better: Random Forests look very promising. However, note that
the score on the training set is still much lower than on the validation sets, meaning
that the model is still overfitting the training set. Possible solutions for overfitting are
to simplify the model, constrain it (i.e., regularize it), or get a lot more training data.
However, before you dive much deeper in Random Forests, you should try out many
other models from various categories of Machine Learning algorithms (several Sup‐
port Vector Machines with different kernels, possibly a neural network, etc.), without
spending too much time tweaking the hyperparameters. The goal is to shortlist a few
(two to five) promising models.

**78 | Chapter 2: End-to-End Machine Learning Project**

```
You should save every model you experiment with, so you can
come back easily to any model you want. Make sure you save both
the hyperparameters and the trained parameters, as well as the
cross-validation scores and perhaps the actual predictions as well.
This will allow you to easily compare scores across model types,
and compare the types of errors they make. You can easily save
Scikit-Learn models by using Python’s pickle module, or using
sklearn.externals.joblib, which is more efficient at serializing
large NumPy arrays:
from sklearn.externals import joblib
```

```
joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")
```

### Fine-Tune Your Model

Let’s assume that you now have a shortlist of promising models. You now need to
fine-tune them. Let’s look at a few ways you can do that.

#### Grid Search

One way to do that would be to fiddle with the hyperparameters manually, until you
find a great combination of hyperparameter values. This would be very tedious work,
and you may not have time to explore many combinations.

Instead you should get Scikit-Learn’s GridSearchCV to search for you. All you need to
do is tell it which hyperparameters you want it to experiment with, and what values to
try out, and it will evaluate all the possible combinations of hyperparameter values,
using cross-validation. For example, the following code searches for the best combi‐
nation of hyperparameter values for the RandomForestRegressor:

```
from sklearn.model_selection import GridSearchCV
```

```
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
```

```
forest_reg = RandomForestRegressor()
```

```
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)
```

```
grid_search.fit(housing_prepared, housing_labels)
```

```
Fine-Tune Your Model | 79
```

```
When you have no idea what value a hyperparameter should have,
a simple approach is to try out consecutive powers of 10 (or a
smaller number if you want a more fine-grained search, as shown
in this example with the n_estimators hyperparameter).
```

This param_grid tells Scikit-Learn to first evaluate all 3 × 4 = 12 combinations of
n_estimators and max_features hyperparameter values specified in the first dict
(don’t worry about what these hyperparameters mean for now; they will be explained
in Chapter 7), then try all 2 × 3 = 6 combinations of hyperparameter values in the
second dict, but this time with the bootstrap hyperparameter set to False instead of
True (which is the default value for this hyperparameter).

All in all, the grid search will explore 12 + 6 = 18 combinations of RandomForestRe
gressor hyperparameter values, and it will train each model five times (since we are
using five-fold cross validation). In other words, all in all, there will be 18 × 5 = 90
rounds of training! It may take quite a long time, but when it is done you can get the
best combination of parameters like this:

```
>>> grid_search.best_params_
{'max_features': 8, 'n_estimators': 30}
```

```
Since 8 and 30 are the maximum values that were evaluated, you
should probably try searching again with higher values, since the
score may continue to improve.
```

You can also get the best estimator directly:

```
>>> grid_search.best_estimator_
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, min_samples_leaf=1,
min_samples_split=2, min_weight_fraction_leaf=0.0,
n_estimators=30, n_jobs=None, oob_score=False, random_state=None,
verbose=0, warm_start=False)
```

```
If GridSearchCV is initialized with refit=True (which is the
default), then once it finds the best estimator using cross-
validation, it retrains it on the whole training set. This is usually a
good idea since feeding it more data will likely improve its perfor‐
mance.
```

And of course the evaluation scores are also available:

```
>>> cvres = grid_search.cv_results_
>>> for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
```

**80 | Chapter 2: End-to-End Machine Learning Project**

```
... print (np.sqrt(-mean_score), params)
```

```
63669.05791727153 {'max_features': 2, 'n_estimators': 3}
55627.16171305252 {'max_features': 2, 'n_estimators': 10}
53384.57867637289 {'max_features': 2, 'n_estimators': 30}
60965.99185930139 {'max_features': 4, 'n_estimators': 3}
52740.98248528835 {'max_features': 4, 'n_estimators': 10}
50377.344409590376 {'max_features': 4, 'n_estimators': 30}
58663.84733372485 {'max_features': 6, 'n_estimators': 3}
52006.15355973719 {'max_features': 6, 'n_estimators': 10}
50146.465964159885 {'max_features': 6, 'n_estimators': 30}
57869.25504027614 {'max_features': 8, 'n_estimators': 3}
51711.09443660957 {'max_features': 8, 'n_estimators': 10}
49682.25345942335 {'max_features': 8, 'n_estimators': 30}
62895.088889905004 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
54658.14484390074 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
59470.399594730654 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
52725.01091081235 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
57490.612956065226 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
51009.51445842374 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
```

In this example, we obtain the best solution by setting the max_features hyperpara‐
meter to 8 , and the n_estimators hyperparameter to 30. The RMSE score for this
combination is 49,682, which is slightly better than the score you got earlier using the
default hyperparameter values (which was 50,182). Congratulations, you have suc‐
cessfully fine-tuned your best model!

```
Don’t forget that you can treat some of the data preparation steps as
hyperparameters. For example, the grid search will automatically
find out whether or not to add a feature you were not sure about
(e.g., using the add_bedrooms_per_room hyperparameter of your
CombinedAttributesAdder transformer). It may similarly be used
to automatically find the best way to handle outliers, missing fea‐
tures, feature selection, and more.
```

#### Randomized Search

The grid search approach is fine when you are exploring relatively few combinations,
like in the previous example, but when the hyperparameter search space is large, it is
often preferable to use RandomizedSearchCV instead. This class can be used in much
the same way as the GridSearchCV class, but instead of trying out all possible combi‐
nations, it evaluates a given number of random combinations by selecting a random
value for each hyperparameter at every iteration. This approach has two main bene‐
fits:

```
Fine-Tune Your Model | 81
```

- If you let the randomized search run for, say, 1,000 iterations, this approach will
  explore 1,000 different values for each hyperparameter (instead of just a few val‐
  ues per hyperparameter with the grid search approach).
- You have more control over the computing budget you want to allocate to hyper‐
  parameter search, simply by setting the number of iterations.

#### Ensemble Methods

Another way to fine-tune your system is to try to combine the models that perform
best. The group (or “ensemble”) will often perform better than the best individual
model (just like Random Forests perform better than the individual Decision Trees
they rely on), especially if the individual models make very different types of errors.
We will cover this topic in more detail in Chapter 7.

#### Analyze the Best Models and Their Errors

You will often gain good insights on the problem by inspecting the best models. For
example, the RandomForestRegressor can indicate the relative importance of each
attribute for making accurate predictions:

```
>>> feature_importances = grid_search.best_estimator_.feature_importances_
>>> feature_importances
array([7.33442355e-02, 6.29090705e-02, 4.11437985e-02, 1.46726854e-02,
1.41064835e-02, 1.48742809e-02, 1.42575993e-02, 3.66158981e-01,
5.64191792e-02, 1.08792957e-01, 5.33510773e-02, 1.03114883e-02,
1.64780994e-01, 6.02803867e-05, 1.96041560e-03, 2.85647464e-03])
```

Let’s display these importance scores next to their corresponding attribute names:

```
>>> extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
>>> cat_encoder = full_pipeline.named_transformers_["cat"]
>>> cat_one_hot_attribs = list(cat_encoder.categories_[0])
>>> attributes = num_attribs + extra_attribs + cat_one_hot_attribs
>>> sorted(zip(feature_importances, attributes), reverse=True)
[(0.3661589806181342, 'median_income'),
(0.1647809935615905, 'INLAND'),
(0.10879295677551573, 'pop_per_hhold'),
(0.07334423551601242, 'longitude'),
(0.0629090704826203, 'latitude'),
(0.05641917918195401, 'rooms_per_hhold'),
(0.05335107734767581, 'bedrooms_per_room'),
(0.041143798478729635, 'housing_median_age'),
(0.014874280890402767, 'population'),
(0.014672685420543237, 'total_rooms'),
(0.014257599323407807, 'households'),
(0.014106483453584102, 'total_bedrooms'),
(0.010311488326303787, '<1H OCEAN'),
(0.002856474637320158, 'NEAR OCEAN'),
```

**82 | Chapter 2: End-to-End Machine Learning Project**

```
(0.00196041559947807, 'NEAR BAY'),
(6.028038672736599e-05, 'ISLAND')]
```

With this information, you may want to try dropping some of the less useful features
(e.g., apparently only one ocean_proximity category is really useful, so you could try
dropping the others).

You should also look at the specific errors that your system makes, then try to under‐
stand why it makes them and what could fix the problem (adding extra features or, on
the contrary, getting rid of uninformative ones, cleaning up outliers, etc.).

#### Evaluate Your System on the Test Set

After tweaking your models for a while, you eventually have a system that performs
sufficiently well. Now is the time to evaluate the final model on the test set. There is
nothing special about this process; just get the predictors and the labels from your
test set, run your full_pipeline to transform the data (call transform(), not
fit_transform(), you do not want to fit the test set!), and evaluate the final model
on the test set:

```
final_model = grid_search.best_estimator_
```

```
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
```

```
X_test_prepared = full_pipeline.transform(X_test)
```

```
final_predictions = final_model.predict(X_test_prepared)
```

```
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2
```

In some cases, such a point estimate of the generalization error will not be quite
enough to convince you to launch: what if it is just 0.1% better than the model cur‐
rently in production? You might want to have an idea of how precise this estimate is.
For this, you can compute a 95% confidence interval for the generalization error using
scipy.stats.t.interval():

```
>>> from scipy import stats
>>> confidence = 0.95
>>> squared_errors = (final_predictions - y_test) ** 2
>>> np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
... loc=squared_errors.mean(),
... scale=stats.sem(squared_errors)))
...
array([45685.10470776, 49691.25001878])
```

The performance will usually be slightly worse than what you measured using cross-
validation if you did a lot of hyperparameter tuning (because your system ends up
fine-tuned to perform well on the validation data, and will likely not perform as well

```
Fine-Tune Your Model | 83
```

on unknown datasets). It is not the case in this example, but when this happens you
must resist the temptation to tweak the hyperparameters to make the numbers look
good on the test set; the improvements would be unlikely to generalize to new data.

Now comes the project prelaunch phase: you need to present your solution (high‐
lighting what you have learned, what worked and what did not, what assumptions
were made, and what your system’s limitations are), document everything, and create
nice presentations with clear visualizations and easy-to-remember statements (e.g.,
“the median income is the number one predictor of housing prices”). In this Califor‐
nia housing example, the final performance of the system is not better than the
experts’, but it may still be a good idea to launch it, especially if this frees up some
time for the experts so they can work on more interesting and productive tasks.

### Launch, Monitor, and Maintain Your System

Perfect, you got approval to launch! You need to get your solution ready for produc‐
tion, in particular by plugging the production input data sources into your system
and writing tests.

You also need to write monitoring code to check your system’s live performance at
regular intervals and trigger alerts when it drops. This is important to catch not only
sudden breakage, but also performance degradation. This is quite common because
models tend to “rot” as data evolves over time, unless the models are regularly trained
on fresh data.

Evaluating your system’s performance will require sampling the system’s predictions
and evaluating them. This will generally require a human analysis. These analysts
may be field experts, or workers on a crowdsourcing platform (such as Amazon
Mechanical Turk or CrowdFlower). Either way, you need to plug the human evalua‐
tion pipeline into your system.

You should also make sure you evaluate the system’s input data quality. Sometimes
performance will degrade slightly because of a poor quality signal (e.g., a malfunc‐
tioning sensor sending random values, or another team’s output becoming stale), but
it may take a while before your system’s performance degrades enough to trigger an
alert. If you monitor your system’s inputs, you may catch this earlier. Monitoring the
inputs is particularly important for online learning systems.

Finally, you will generally want to train your models on a regular basis using fresh
data. You should automate this process as much as possible. If you don’t, you are very
likely to refresh your model only every six months (at best), and your system’s perfor‐
mance may fluctuate severely over time. If your system is an online learning system,
you should make sure you save snapshots of its state at regular intervals so you can
easily roll back to a previously working state.

**84 | Chapter 2: End-to-End Machine Learning Project**

### Try It Out!

Hopefully this chapter gave you a good idea of what a Machine Learning project
looks like, and showed you some of the tools you can use to train a great system. As
you can see, much of the work is in the data preparation step, building monitoring
tools, setting up human evaluation pipelines, and automating regular model training.
The Machine Learning algorithms are also important, of course, but it is probably
preferable to be comfortable with the overall process and know three or four algo‐
rithms well rather than to spend all your time exploring advanced algorithms and not
enough time on the overall process.

So, if you have not already done so, now is a good time to pick up a laptop, select a
dataset that you are interested in, and try to go through the whole process from A to
Z. A good place to start is on a competition website such as [http://kaggle.com/:](http://kaggle.com/:) you
will have a dataset to play with, a clear goal, and people to share the experience with.

### Exercises

Using this chapter’s housing dataset:

1. Try a Support Vector Machine regressor (sklearn.svm.SVR), with various hyper‐
   parameters such as kernel="linear" (with various values for the C hyperpara‐
   meter) or kernel="rbf" (with various values for the C and gamma
   hyperparameters). Don’t worry about what these hyperparameters mean for now.
   How does the best SVR predictor perform?
2. Try replacing GridSearchCV with RandomizedSearchCV.
3. Try adding a transformer in the preparation pipeline to select only the most
   important attributes.
4. Try creating a single pipeline that does the full data preparation plus the final
   prediction.
5. Automatically explore some preparation options using GridSearchCV.

Solutions to these exercises are available in the online Jupyter notebooks at https://
github.com/ageron/handson-ml2.

```
Try It Out! | 85
```

```
1 By default Scikit-Learn caches downloaded datasets in a directory called $HOME/scikit_learn_data.
```

**CHAPTER 3**

**Classification**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 3 in the final
release of the book.
```

In Chapter 1 we mentioned that the most common supervised learning tasks are
regression (predicting values) and classification (predicting classes). In Chapter 2 we
explored a regression task, predicting housing values, using various algorithms such
as Linear Regression, Decision Trees, and Random Forests (which will be explained
in further detail in later chapters). Now we will turn our attention to classification
systems.

### MNIST

In this chapter, we will be using the MNIST dataset, which is a set of 70,000 small
images of digits handwritten by high school students and employees of the US Cen‐
sus Bureau. Each image is labeled with the digit it represents. This set has been stud‐
ied so much that it is often called the “Hello World” of Machine Learning: whenever
people come up with a new classification algorithm, they are curious to see how it
will perform on MNIST. Whenever someone learns Machine Learning, sooner or
later they tackle MNIST.

Scikit-Learn provides many helper functions to download popular datasets. MNIST is
one of them. The following code fetches the MNIST dataset:^1

```
87
```

```
>>> from sklearn.datasets import fetch_openml
>>> mnist = fetch_openml('mnist_784', version=1)
>>> mnist.keys()
dict_keys(['data', 'target', 'feature_names', 'DESCR', 'details',
'categories', 'url'])
```

Datasets loaded by Scikit-Learn generally have a similar dictionary structure includ‐
ing:

- A DESCR key describing the dataset
- A data key containing an array with one row per instance and one column per
  feature
- A target key containing an array with the labels

Let’s look at these arrays:

```
>>> X, y = mnist["data"], mnist["target"]
>>> X.shape
(70000, 784)
>>> y.shape
(70000,)
```

There are 70,000 images, and each image has 784 features. This is because each image
is 28×28 pixels, and each feature simply represents one pixel’s intensity, from 0
(white) to 255 (black). Let’s take a peek at one digit from the dataset. All you need to
do is grab an instance’s feature vector, reshape it to a 28×28 array, and display it using
Matplotlib’s imshow() function:

```
import matplotlib as mpl
import matplotlib.pyplot as plt
```

```
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
```

```
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
```

This looks like a 5, and indeed that’s what the label tells us:

**88 | Chapter 3: Classification**

```
>>> y[0]
'5'
```

Note that the label is a string. We prefer numbers, so let’s cast y to integers:

```
>>> y = y.astype(np.uint8)
```

Figure 3-1 shows a few more images from the MNIST dataset to give you a feel for
the complexity of the classification task.

Figure 3-1. A few digits from the MNIST dataset

But wait! You should always create a test set and set it aside before inspecting the data
closely. The MNIST dataset is actually already split into a training set (the first 60,000
images) and a test set (the last 10,000 images):

```
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

The training set is already shuffled for us, which is good as this guarantees that all
cross-validation folds will be similar (you don’t want one fold to be missing some dig‐
its). Moreover, some learning algorithms are sensitive to the order of the training

```
MNIST | 89
```

```
2 Shuffling may be a bad idea in some contexts—for example, if you are working on time series data (such as
stock market prices or weather conditions). We will explore this in the next chapters.
```

instances, and they perform poorly if they get many similar instances in a row. Shuf‐
fling the dataset ensures that this won’t happen.^2

### Training a Binary Classifier

Let’s simplify the problem for now and only try to identify one digit—for example,
the number 5. This “5-detector” will be an example of a binary classifier, capable of
distinguishing between just two classes, 5 and not-5. Let’s create the target vectors for
this classification task:

```
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)
```

Okay, now let’s pick a classifier and train it. A good place to start is with a Stochastic
Gradient Descent (SGD) classifier, using Scikit-Learn’s SGDClassifier class. This clas‐
sifier has the advantage of being capable of handling very large datasets efficiently.
This is in part because SGD deals with training instances independently, one at a time
(which also makes SGD well suited for online learning), as we will see later. Let’s create
an SGDClassifier and train it on the whole training set:

```
from sklearn.linear_model import SGDClassifier
```

```
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

```
The SGDClassifier relies on randomness during training (hence
the name “stochastic”). If you want reproducible results, you
should set the random_state parameter.
```

Now you can use it to detect images of the number 5:

```
>>> sgd_clf.predict([some_digit])
array([ True])
```

The classifier guesses that this image represents a 5 (True). Looks like it guessed right
in this particular case! Now, let’s evaluate this model’s performance.

### Performance Measures

Evaluating a classifier is often significantly trickier than evaluating a regressor, so we
will spend a large part of this chapter on this topic. There are many performance

**90 | Chapter 3: Classification**

measures available, so grab another coffee and get ready to learn many new concepts
and acronyms!

#### Measuring Accuracy Using Cross-Validation

A good way to evaluate a model is to use cross-validation, just as you did in Chap‐
ter 2.

```
Implementing Cross-Validation
Occasionally you will need more control over the cross-validation process than what
Scikit-Learn provides off-the-shelf. In these cases, you can implement cross-
validation yourself; it is actually fairly straightforward. The following code does
roughly the same thing as Scikit-Learn’s cross_val_score() function, and prints the
same result:
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
```

```
skfolds = StratifiedKFold(n_splits=3, random_state=42)
```

```
for train_index, test_index in skfolds.split(X_train, y_train_5):
clone_clf = clone(sgd_clf)
X_train_folds = X_train[train_index]
y_train_folds = y_train_5[train_index]
X_test_fold = X_train[test_index]
y_test_fold = y_train_5[test_index]
```

```
clone_clf.fit(X_train_folds, y_train_folds)
y_pred = clone_clf.predict(X_test_fold)
n_correct = sum(y_pred == y_test_fold)
print (n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495
The StratifiedKFold class performs stratified sampling (as explained in Chapter 2)
to produce folds that contain a representative ratio of each class. At each iteration the
code creates a clone of the classifier, trains that clone on the training folds, and makes
predictions on the test fold. Then it counts the number of correct predictions and
outputs the ratio of correct predictions.
```

Let’s use the cross_val_score() function to evaluate your SGDClassifier model
using K-fold cross-validation, with three folds. Remember that K-fold cross-
validation means splitting the training set into K-folds (in this case, three), then mak‐
ing predictions and evaluating them on each fold using a model trained on the
remaining folds (see Chapter 2):

```
Performance Measures | 91
```

```
>>> from sklearn.model_selection import cross_val_score
>>> cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
array([0.96355, 0.93795, 0.95615])
```

Wow! Above 93% accuracy (ratio of correct predictions) on all cross-validation folds?
This looks amazing, doesn’t it? Well, before you get too excited, let’s look at a very
dumb classifier that just classifies every single image in the “not-5” class:

```
from sklearn.base import BaseEstimator
```

```
class Never5Classifier (BaseEstimator):
def fit(self, X, y=None):
pass
def predict(self, X):
return np.zeros((len(X), 1), dtype=bool)
```

Can you guess this model’s accuracy? Let’s find out:

```
>>> never_5_clf = Never5Classifier()
>>> cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
array([0.91125, 0.90855, 0.90915])
```

That’s right, it has over 90% accuracy! This is simply because only about 10% of the
images are 5s, so if you always guess that an image is not a 5, you will be right about
90% of the time. Beats Nostradamus.

This demonstrates why accuracy is generally not the preferred performance measure
for classifiers, especially when you are dealing with skewed datasets (i.e., when some
classes are much more frequent than others).

#### Confusion Matrix

A much better way to evaluate the performance of a classifier is to look at the confu‐
sion matrix. The general idea is to count the number of times instances of class A are
classified as class B. For example, to know the number of times the classifier confused
images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion
matrix.

To compute the confusion matrix, you first need to have a set of predictions, so they
can be compared to the actual targets. You could make predictions on the test set, but
let’s keep it untouched for now (remember that you want to use the test set only at the
very end of your project, once you have a classifier that you are ready to launch).
Instead, you can use the cross_val_predict() function:

```
from sklearn.model_selection import cross_val_predict
```

```
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

Just like the cross_val_score() function, cross_val_predict() performs K-fold
cross-validation, but instead of returning the evaluation scores, it returns the predic‐

**92 | Chapter 3: Classification**

tions made on each test fold. This means that you get a clean prediction for each
instance in the training set (“clean” meaning that the prediction is made by a model
that never saw the data during training).

Now you are ready to get the confusion matrix using the confusion_matrix() func‐
tion. Just pass it the target classes (y_train_5) and the predicted classes
(y_train_pred):

```
>>> from sklearn.metrics import confusion_matrix
>>> confusion_matrix(y_train_5, y_train_pred)
array([[53057, 1522],
[ 1325, 4096]])
```

Each row in a confusion matrix represents an actual class, while each column repre‐
sents a predicted class. The first row of this matrix considers non-5 images (the nega‐
tive class): 53,057 of them were correctly classified as non-5s (they are called true
negatives), while the remaining 1,522 were wrongly classified as 5s (false positives).
The second row considers the images of 5s (the positive class): 1,325 were wrongly
classified as non-5s (false negatives), while the remaining 4,096 were correctly classi‐
fied as 5s (true positives). A perfect classifier would have only true positives and true
negatives, so its confusion matrix would have nonzero values only on its main diago‐
nal (top left to bottom right):

```
>>> y_train_perfect_predictions = y_train_5 # pretend we reached perfection
>>> confusion_matrix(y_train_5, y_train_perfect_predictions)
array([[54579, 0],
[ 0, 5421]])
```

The confusion matrix gives you a lot of information, but sometimes you may prefer a
more concise metric. An interesting one to look at is the accuracy of the positive pre‐
dictions; this is called the precision of the classifier (Equation 3-1).

```
Equation 3-1. Precision
```

```
precision =TPTP+FP
```

TP is the number of true positives, and FP is the number of false positives.

A trivial way to have perfect precision is to make one single positive prediction and
ensure it is correct (precision = 1/1 = 100%). This would not be very useful since the
classifier would ignore all but one positive instance. So precision is typically used
along with another metric named recall, also called sensitivity or true positive rate

```
Performance Measures | 93
```

(TPR): this is the ratio of positive instances that are correctly detected by the classifier
(Equation 3-2).

```
Equation 3-2. Recall
```

```
recall =TPTP+FN
```

FN is of course the number of false negatives.

If you are confused about the confusion matrix, Figure 3-2 may help.

Figure 3-2. An illustrated confusion matrix

#### Precision and Recall

Scikit-Learn provides several functions to compute classifier metrics, including preci‐
sion and recall:

```
>>> from sklearn.metrics import precision_score, recall_score
>>> precision_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1522)
0.7290850836596654
>>> recall_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1325)
0.7555801512636044
```

Now your 5-detector does not look as shiny as it did when you looked at its accuracy.
When it claims an image represents a 5, it is correct only 72.9% of the time. More‐
over, it only detects 75.6% of the 5s.

It is often convenient to combine precision and recall into a single metric called the F 1
score, in particular if you need a simple way to compare two classifiers. The F 1 score is
the harmonic mean of precision and recall (Equation 3-3). Whereas the regular mean

**94 | Chapter 3: Classification**

treats all values equally, the harmonic mean gives much more weight to low values.
As a result, the classifier will only get a high F 1 score if both recall and precision are
high.

```
Equation 3-3. F 1
```

```
F 1 = 1 2
precision+
```

```
1
recall
```

```
= 2 × precision × recallprecision + recall = TP
TP+FN 2 +FP
```

To compute the F 1 score, simply call the f1_score() function:

```
>>> from sklearn.metrics import f1_score
>>> f1_score(y_train_5, y_train_pred)
0.7420962043663375
```

The F 1 score favors classifiers that have similar precision and recall. This is not always
what you want: in some contexts you mostly care about precision, and in other con‐
texts you really care about recall. For example, if you trained a classifier to detect vid‐
eos that are safe for kids, you would probably prefer a classifier that rejects many
good videos (low recall) but keeps only safe ones (high precision), rather than a clas‐
sifier that has a much higher recall but lets a few really bad videos show up in your
product (in such cases, you may even want to add a human pipeline to check the clas‐
sifier’s video selection). On the other hand, suppose you train a classifier to detect
shoplifters on surveillance images: it is probably fine if your classifier has only 30%
precision as long as it has 99% recall (sure, the security guards will get a few false
alerts, but almost all shoplifters will get caught).

Unfortunately, you can’t have it both ways: increasing precision reduces recall, and
vice versa. This is called the precision/recall tradeoff.

#### Precision/Recall Tradeoff

To understand this tradeoff, let’s look at how the SGDClassifier makes its classifica‐
tion decisions. For each instance, it computes a score based on a decision function,
and if that score is greater than a threshold, it assigns the instance to the positive
class, or else it assigns it to the negative class. Figure 3-3 shows a few digits positioned
from the lowest score on the left to the highest score on the right. Suppose the deci‐
sion threshold is positioned at the central arrow (between the two 5s): you will find 4
true positives (actual 5s) on the right of that threshold, and one false positive (actually
a 6). Therefore, with that threshold, the precision is 80% (4 out of 5). But out of 6
actual 5s, the classifier only detects 4, so the recall is 67% (4 out of 6). Now if you
raise the threshold (move it to the arrow on the right), the false positive (the 6)
becomes a true negative, thereby increasing precision (up to 100% in this case), but
one true positive becomes a false negative, decreasing recall down to 50%. Conversely,
lowering the threshold increases recall and reduces precision.

```
Performance Measures | 95
```

Figure 3-3. Decision threshold and precision/recall tradeoff

Scikit-Learn does not let you set the threshold directly, but it does give you access to
the decision scores that it uses to make predictions. Instead of calling the classifier’s
predict() method, you can call its decision_function() method, which returns a
score for each instance, and then make predictions based on those scores using any
threshold you want:

```
>>> y_scores = sgd_clf.decision_function([some_digit])
>>> y_scores
array([2412.53175101])
>>> threshold = 0
>>> y_some_digit_pred = (y_scores > threshold)
array([ True])
```

The SGDClassifier uses a threshold equal to 0, so the previous code returns the same
result as the predict() method (i.e., True). Let’s raise the threshold:

```
>>> threshold = 8000
>>> y_some_digit_pred = (y_scores > threshold)
>>> y_some_digit_pred
array([False])
```

This confirms that raising the threshold decreases recall. The image actually repre‐
sents a 5, and the classifier detects it when the threshold is 0, but it misses it when the
threshold is increased to 8,000.

Now how do you decide which threshold to use? For this you will first need to get the
scores of all instances in the training set using the cross_val_predict() function
again, but this time specifying that you want it to return decision scores instead of
predictions:

```
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
method="decision_function")
```

Now with these scores you can compute precision and recall for all possible thresh‐
olds using the precision_recall_curve() function:

**96 | Chapter 3: Classification**

```
from sklearn.metrics import precision_recall_curve
```

```
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

Finally, you can plot precision and recall as functions of the threshold value using
Matplotlib (Figure 3-4):

```
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
[...] # highlight the threshold, add the legend, axis label and grid
```

```
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
```

Figure 3-4. Precision and recall versus the decision threshold

```
You may wonder why the precision curve is bumpier than the recall
curve in Figure 3-4. The reason is that precision may sometimes go
down when you raise the threshold (although in general it will go
up). To understand why, look back at Figure 3-3 and notice what
happens when you start from the central threshold and move it just
one digit to the right: precision goes from 4/5 (80%) down to 3/4
(75%). On the other hand, recall can only go down when the thres‐
hold is increased, which explains why its curve looks smooth.
```

Another way to select a good precision/recall tradeoff is to plot precision directly
against recall, as shown in Figure 3-5 (the same threshold as earlier is highlighed).

```
Performance Measures | 97
```

Figure 3-5. Precision versus recall

You can see that precision really starts to fall sharply around 80% recall. You will
probably want to select a precision/recall tradeoff just before that drop—for example,
at around 60% recall. But of course the choice depends on your project.

So let’s suppose you decide to aim for 90% precision. You look up the first plot and
find that you need to use a threshold of about 8,000. To be more precise you can
search for the lowest threshold that gives you at least 90% precision (np.argmax()
will give us the first index of the maximum value, which in this case means the first
True value):

```
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] # ~7816
```

To make predictions (on the training set for now), instead of calling the classifier’s
predict() method, you can just run this code:

```
y_train_pred_90 = (y_scores >= threshold_90_precision)
```

Let’s check these predictions’ precision and recall:

```
>>> precision_score(y_train_5, y_train_pred_90)
0.9000380083618396
>>> recall_score(y_train_5, y_train_pred_90)
0.4368197749492714
```

Great, you have a 90% precision classifier! As you can see, it is fairly easy to create a
classifier with virtually any precision you want: just set a high enough threshold, and
you’re done. Hmm, not so fast. A high-precision classifier is not very useful if its
recall is too low!

**98 | Chapter 3: Classification**

```
If someone says “let’s reach 99% precision,” you should ask, “at
what recall?”
```

#### The ROC Curve

The receiver operating characteristic (ROC) curve is another common tool used with
binary classifiers. It is very similar to the precision/recall curve, but instead of plot‐
ting precision versus recall, the ROC curve plots the true positive rate (another name
for recall) against the false positive rate. The FPR is the ratio of negative instances that
are incorrectly classified as positive. It is equal to one minus the true negative rate,
which is the ratio of negative instances that are correctly classified as negative. The
TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus
1 – specificity.

To plot the ROC curve, you first need to compute the TPR and FPR for various thres‐
hold values, using the roc_curve() function:

```
from sklearn.metrics import roc_curve
```

```
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

Then you can plot the FPR against the TPR using Matplotlib. This code produces the
plot in Figure 3-6:

```
def plot_roc_curve(fpr, tpr, label=None):
plt.plot(fpr, tpr, linewidth=2, label=label)
plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
[...] # Add axis labels and grid
```

```
plot_roc_curve(fpr, tpr)
plt.show()
```

```
Performance Measures | 99
```

Figure 3-6. ROC curve

Once again there is a tradeoff: the higher the recall (TPR), the more false positives
(FPR) the classifier produces. The dotted line represents the ROC curve of a purely
random classifier; a good classifier stays as far away from that line as possible (toward
the top-left corner).

One way to compare classifiers is to measure the area under the curve (AUC). A per‐
fect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will
have a ROC AUC equal to 0.5. Scikit-Learn provides a function to compute the ROC
AUC:

```
>>> from sklearn.metrics import roc_auc_score
>>> roc_auc_score(y_train_5, y_scores)
0.9611778893101814
```

```
Since the ROC curve is so similar to the precision/recall (or PR)
curve, you may wonder how to decide which one to use. As a rule
of thumb, you should prefer the PR curve whenever the positive
class is rare or when you care more about the false positives than
the false negatives, and the ROC curve otherwise. For example,
looking at the previous ROC curve (and the ROC AUC score), you
may think that the classifier is really good. But this is mostly
because there are few positives (5s) compared to the negatives
(non-5s). In contrast, the PR curve makes it clear that the classifier
has room for improvement (the curve could be closer to the top-
right corner).
```

**100 | Chapter 3: Classification**

Let’s train a RandomForestClassifier and compare its ROC curve and ROC AUC
score to the SGDClassifier. First, you need to get scores for each instance in the
training set. But due to the way it works (see Chapter 7), the RandomForestClassi
fier class does not have a decision_function() method. Instead it has a pre
dict_proba() method. Scikit-Learn classifiers generally have one or the other. The
predict_proba() method returns an array containing a row per instance and a col‐
umn per class, each containing the probability that the given instance belongs to the
given class (e.g., 70% chance that the image represents a 5):

```
from sklearn.ensemble import RandomForestClassifier
```

```
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
method="predict_proba")
```

But to plot a ROC curve, you need scores, not probabilities. A simple solution is to
use the positive class’s probability as the score:

```
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
```

Now you are ready to plot the ROC curve. It is useful to plot the first ROC curve as
well to see how they compare (Figure 3-7):

```
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()
```

Figure 3-7. Comparing ROC curves

```
Performance Measures | 101
```

As you can see in Figure 3-7, the RandomForestClassifier’s ROC curve looks much
better than the SGDClassifier’s: it comes much closer to the top-left corner. As a
result, its ROC AUC score is also significantly better:

```
>>> roc_auc_score(y_train_5, y_scores_forest)
0.9983436731328145
```

Try measuring the precision and recall scores: you should find 99.0% precision and
86.6% recall. Not too bad!

Hopefully you now know how to train binary classifiers, choose the appropriate met‐
ric for your task, evaluate your classifiers using cross-validation, select the precision/
recall tradeoff that fits your needs, and compare various models using ROC curves
and ROC AUC scores. Now let’s try to detect more than just the 5s.

### Multiclass Classification

Whereas binary classifiers distinguish between two classes, multiclass classifiers (also
called multinomial classifiers) can distinguish between more than two classes.

Some algorithms (such as Random Forest classifiers or naive Bayes classifiers) are
capable of handling multiple classes directly. Others (such as Support Vector Machine
classifiers or Linear classifiers) are strictly binary classifiers. However, there are vari‐
ous strategies that you can use to perform multiclass classification using multiple
binary classifiers.

For example, one way to create a system that can classify the digit images into 10
classes (from 0 to 9) is to train 10 binary classifiers, one for each digit (a 0-detector, a
1-detector, a 2-detector, and so on). Then when you want to classify an image, you get
the decision score from each classifier for that image and you select the class whose
classifier outputs the highest score. This is called the one-versus-all (OvA) strategy
(also called one-versus-the-rest).

Another strategy is to train a binary classifier for every pair of digits: one to distin‐
guish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and so on.
This is called the one-versus-one (OvO) strategy. If there are N classes, you need to
train N × (N – 1) / 2 classifiers. For the MNIST problem, this means training 45
binary classifiers! When you want to classify an image, you have to run the image
through all 45 classifiers and see which class wins the most duels. The main advan‐
tage of OvO is that each classifier only needs to be trained on the part of the training
set for the two classes that it must distinguish.

Some algorithms (such as Support Vector Machine classifiers) scale poorly with the
size of the training set, so for these algorithms OvO is preferred since it is faster to
train many classifiers on small training sets than training few classifiers on large
training sets. For most binary classification algorithms, however, OvA is preferred.

**102 | Chapter 3: Classification**

Scikit-Learn detects when you try to use a binary classification algorithm for a multi‐
class classification task, and it automatically runs OvA (except for SVM classifiers for
which it uses OvO). Let’s try this with the SGDClassifier:

```
>>> sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
>>> sgd_clf.predict([some_digit])
array([5], dtype=uint8)
```

That was easy! This code trains the SGDClassifier on the training set using the origi‐
nal target classes from 0 to 9 (y_train), instead of the 5-versus-all target classes
(y_train_5). Then it makes a prediction (a correct one in this case). Under the hood,
Scikit-Learn actually trained 10 binary classifiers, got their decision scores for the
image, and selected the class with the highest score.

To see that this is indeed the case, you can call the decision_function() method.
Instead of returning just one score per instance, it now returns 10 scores, one per
class:

```
>>> some_digit_scores = sgd_clf.decision_function([some_digit])
>>> some_digit_scores
array([[-15955.22627845, -38080.96296175, -13326.66694897,
573.52692379, -17680.6846644 , 2412.53175101,
-25526.86498156, -12290.15704709, -7946.05205023,
-10631.35888549]])
```

The highest score is indeed the one corresponding to class 5:

```
>>> np.argmax(some_digit_scores)
5
>>> sgd_clf.classes_
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
>>> sgd_clf.classes_[5]
5
```

```
When a classifier is trained, it stores the list of target classes in its
classes_ attribute, ordered by value. In this case, the index of each
class in the classes_ array conveniently matches the class itself
(e.g., the class at index 5 happens to be class 5), but in general you
won’t be so lucky.
```

If you want to force ScikitLearn to use one-versus-one or one-versus-all, you can use
the OneVsOneClassifier or OneVsRestClassifier classes. Simply create an instance
and pass a binary classifier to its constructor. For example, this code creates a multi‐
class classifier using the OvO strategy, based on a SGDClassifier:

```
>>> from sklearn.multiclass import OneVsOneClassifier
>>> ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
>>> ovo_clf.fit(X_train, y_train)
>>> ovo_clf.predict([some_digit])
```

```
Multiclass Classification | 103
```

```
array([5], dtype=uint8)
>>> len(ovo_clf.estimators_)
45
```

Training a RandomForestClassifier is just as easy:

```
>>> forest_clf.fit(X_train, y_train)
>>> forest_clf.predict([some_digit])
array([5], dtype=uint8)
```

This time Scikit-Learn did not have to run OvA or OvO because Random Forest
classifiers can directly classify instances into multiple classes. You can call
predict_proba() to get the list of probabilities that the classifier assigned to each
instance for each class:

```
>>> forest_clf.predict_proba([some_digit])
array([[0. , 0. , 0.01, 0.08, 0. , 0.9 , 0. , 0. , 0. , 0.01]])
```

You can see that the classifier is fairly confident about its prediction: the 0.9 at the 5th
index in the array means that the model estimates a 90% probability that the image
represents a 5. It also thinks that the image could instead be a 2, a 3 or a 9, respec‐
tively with 1%, 8% and 1% probability.

Now of course you want to evaluate these classifiers. As usual, you want to use cross-
validation. Let’s evaluate the SGDClassifier’s accuracy using the cross_val_score()
function:

```
>>> cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
array([0.8489802 , 0.87129356, 0.86988048])
```

It gets over 84% on all test folds. If you used a random classifier, you would get 10%
accuracy, so this is not such a bad score, but you can still do much better. For exam‐
ple, simply scaling the inputs (as discussed in Chapter 2) increases accuracy above
89%:

```
>>> from sklearn.preprocessing import StandardScaler
>>> scaler = StandardScaler()
>>> X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
>>> cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
array([0.89707059, 0.8960948 , 0.90693604])
```

### Error Analysis

Of course, if this were a real project, you would follow the steps in your Machine
Learning project checklist (see ???): exploring data preparation options, trying out
multiple models, shortlisting the best ones and fine-tuning their hyperparameters
using GridSearchCV, and automating as much as possible, as you did in the previous
chapter. Here, we will assume that you have found a promising model and you want
to find ways to improve it. One way to do this is to analyze the types of errors it
makes.

**104 | Chapter 3: Classification**

First, you can look at the confusion matrix. You need to make predictions using the
cross_val_predict() function, then call the confusion_matrix() function, just like
you did earlier:

```
>>> y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
>>> conf_mx = confusion_matrix(y_train, y_train_pred)
>>> conf_mx
array([[5578, 0, 22, 7, 8, 45, 35, 5, 222, 1],
[ 0, 6410, 35, 26, 4, 44, 4, 8, 198, 13],
[ 28, 27, 5232, 100, 74, 27, 68, 37, 354, 11],
[ 23, 18, 115, 5254, 2, 209, 26, 38, 373, 73],
[ 11, 14, 45, 12, 5219, 11, 33, 26, 299, 172],
[ 26, 16, 31, 173, 54, 4484, 76, 14, 482, 65],
[ 31, 17, 45, 2, 42, 98, 5556, 3, 123, 1],
[ 20, 10, 53, 27, 50, 13, 3, 5696, 173, 220],
[ 17, 64, 47, 91, 3, 125, 24, 11, 5421, 48],
[ 24, 18, 29, 67, 116, 39, 1, 174, 329, 5152]])
```

That’s a lot of numbers. It’s often more convenient to look at an image representation
of the confusion matrix, using Matplotlib’s matshow() function:

```
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
```

This confusion matrix looks fairly good, since most images are on the main diagonal,
which means that they were classified correctly. The 5s look slightly darker than the
other digits, which could mean that there are fewer images of 5s in the dataset or that
the classifier does not perform as well on 5s as on other digits. In fact, you can verify
that both are the case.

Let’s focus the plot on the errors. First, you need to divide each value in the confusion
matrix by the number of images in the corresponding class, so you can compare error

```
Error Analysis | 105
```

rates instead of absolute number of errors (which would make abundant classes look
unfairly bad):

```
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
```

Now let’s fill the diagonal with zeros to keep only the errors, and let’s plot the result:

```
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
```

Now you can clearly see the kinds of errors the classifier makes. Remember that rows
represent actual classes, while columns represent predicted classes. The column for
class 8 is quite bright, which tells you that many images get misclassified as 8s. How‐
ever, the row for class 8 is not that bad, telling you that actual 8s in general get prop‐
erly classified as 8s. As you can see, the confusion matrix is not necessarily
symmetrical. You can also see that 3s and 5s often get confused (in both directions).

Analyzing the confusion matrix can often give you insights on ways to improve your
classifier. Looking at this plot, it seems that your efforts should be spent on reducing
the false 8s. For example, you could try to gather more training data for digits that
look like 8s (but are not) so the classifier can learn to distinguish them from real 8s.
Or you could engineer new features that would help the classifier—for example, writ‐
ing an algorithm to count the number of closed loops (e.g., 8 has two, 6 has one, 5 has
none). Or you could preprocess the images (e.g., using Scikit-Image, Pillow, or
OpenCV) to make some patterns stand out more, such as closed loops.

Analyzing individual errors can also be a good way to gain insights on what your
classifier is doing and why it is failing, but it is more difficult and time-consuming.

**106 | Chapter 3: Classification**

```
3 But remember that our brain is a fantastic pattern recognition system, and our visual system does a lot of
complex preprocessing before any information reaches our consciousness, so the fact that it feels simple does
not mean that it is.
```

For example, let’s plot examples of 3s and 5s (the plot_digits() function just uses
Matplotlib’s imshow() function; see this chapter’s Jupyter notebook for details):

```
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
```

```
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()
```

The two 5×5 blocks on the left show digits classified as 3s, and the two 5×5 blocks on
the right show images classified as 5s. Some of the digits that the classifier gets wrong
(i.e., in the bottom-left and top-right blocks) are so badly written that even a human
would have trouble classifying them (e.g., the 5 on the 1st row and 2nd column truly
looks like a badly written 3). However, most misclassified images seem like obvious
errors to us, and it’s hard to understand why the classifier made the mistakes it did.^3
The reason is that we used a simple SGDClassifier, which is a linear model. All it
does is assign a weight per class to each pixel, and when it sees a new image it just
sums up the weighted pixel intensities to get a score for each class. So since 3s and 5s
differ only by a few pixels, this model will easily confuse them.

```
Error Analysis | 107
```

The main difference between 3s and 5s is the position of the small line that joins the
top line to the bottom arc. If you draw a 3 with the junction slightly shifted to the left,
the classifier might classify it as a 5, and vice versa. In other words, this classifier is
quite sensitive to image shifting and rotation. So one way to reduce the 3/5 confusion
would be to preprocess the images to ensure that they are well centered and not too
rotated. This will probably help reduce other errors as well.

### Multilabel Classification

Until now each instance has always been assigned to just one class. In some cases you
may want your classifier to output multiple classes for each instance. For example,
consider a face-recognition classifier: what should it do if it recognizes several people
on the same picture? Of course it should attach one tag per person it recognizes. Say
the classifier has been trained to recognize three faces, Alice, Bob, and Charlie; then
when it is shown a picture of Alice and Charlie, it should output [1, 0, 1] (meaning
“Alice yes, Bob no, Charlie yes”). Such a classification system that outputs multiple
binary tags is called a multilabel classification system.

We won’t go into face recognition just yet, but let’s look at a simpler example, just for
illustration purposes:

```
from sklearn.neighbors import KNeighborsClassifier
```

```
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
```

```
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

This code creates a y_multilabel array containing two target labels for each digit
image: the first indicates whether or not the digit is large (7, 8, or 9) and the second
indicates whether or not it is odd. The next lines create a KNeighborsClassifier
instance (which supports multilabel classification, but not all classifiers do) and we
train it using the multiple targets array. Now you can make a prediction, and notice
that it outputs two labels:

```
>>> knn_clf.predict([some_digit])
array([[False, True]])
```

And it gets it right! The digit 5 is indeed not large (False) and odd (True).

There are many ways to evaluate a multilabel classifier, and selecting the right metric
really depends on your project. For example, one approach is to measure the F 1 score
for each individual label (or any other binary classifier metric discussed earlier), then
simply compute the average score. This code computes the average F 1 score across all
labels:

**108 | Chapter 3: Classification**

```
4 Scikit-Learn offers a few other averaging options and multilabel classifier metrics; see the documentation for
more details.
```

```
>>> y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
>>> f1_score(y_multilabel, y_train_knn_pred, average="macro")
0.976410265560605
```

This assumes that all labels are equally important, which may not be the case. In par‐
ticular, if you have many more pictures of Alice than of Bob or Charlie, you may want
to give more weight to the classifier’s score on pictures of Alice. One simple option is
to give each label a weight equal to its support (i.e., the number of instances with that
target label). To do this, simply set average="weighted" in the preceding code.^4

### Multioutput Classification

The last type of classification task we are going to discuss here is called multioutput-
multiclass classification (or simply multioutput classification). It is simply a generaliza‐
tion of multilabel classification where each label can be multiclass (i.e., it can have
more than two possible values).

To illustrate this, let’s build a system that removes noise from images. It will take as
input a noisy digit image, and it will (hopefully) output a clean digit image, repre‐
sented as an array of pixel intensities, just like the MNIST images. Notice that the
classifier’s output is multilabel (one label per pixel) and each label can have multiple
values (pixel intensity ranges from 0 to 255). It is thus an example of a multioutput
classification system.

```
The line between classification and regression is sometimes blurry,
such as in this example. Arguably, predicting pixel intensity is more
akin to regression than to classification. Moreover, multioutput
systems are not limited to classification tasks; you could even have
a system that outputs multiple labels per instance, including both
class labels and value labels.
```

Let’s start by creating the training and test sets by taking the MNIST images and
adding noise to their pixel intensities using NumPy’s randint() function. The target
images will be the original images:

```
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
```

```
Multioutput Classification | 109
```

```
5 You can use the shift() function from the scipy.ndimage.interpolation module. For example,
shift(image, [2, 1], cval=0) shifts the image 2 pixels down and 1 pixel to the right.
```

Let’s take a peek at an image from the test set (yes, we’re snooping on the test data, so
you should be frowning right now):

On the left is the noisy input image, and on the right is the clean target image. Now
let’s train the classifier and make it clean this image:

```
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
```

Looks close enough to the target! This concludes our tour of classification. Hopefully
you should now know how to select good metrics for classification tasks, pick the
appropriate precision/recall tradeoff, compare classifiers, and more generally build
good classification systems for a variety of tasks.

### Exercises

1. Try to build a classifier for the MNIST dataset that achieves over 97% accuracy
   on the test set. Hint: the KNeighborsClassifier works quite well for this task;
   you just need to find good hyperparameter values (try a grid search on the
   weights and n_neighbors hyperparameters).
2. Write a function that can shift an MNIST image in any direction (left, right, up,
   or down) by one pixel.^5 Then, for each image in the training set, create four shif‐

**110 | Chapter 3: Classification**

```
ted copies (one per direction) and add them to the training set. Finally, train your
best model on this expanded training set and measure its accuracy on the test set.
You should observe that your model performs even better now! This technique of
artificially growing the training set is called data augmentation or training set
expansion.
```

3. Tackle the Titanic dataset. A great place to start is on Kaggle.
4. Build a spam classifier (a more challenging exercise):
   - Download examples of spam and ham from Apache SpamAssassin’s public
     datasets.
   - Unzip the datasets and familiarize yourself with the data format.
   - Split the datasets into a training set and a test set.
   - Write a data preparation pipeline to convert each email into a feature vector.
     Your preparation pipeline should transform an email into a (sparse) vector
     indicating the presence or absence of each possible word. For example, if all
     emails only ever contain four words, “Hello,” “how,” “are,” “you,” then the email
     “Hello you Hello Hello you” would be converted into a vector [1, 0, 0, 1]
     (meaning [“Hello” is present, “how” is absent, “are” is absent, “you” is
     present]), or [3, 0, 0, 2] if you prefer to count the number of occurrences of
     each word.
   - You may want to add hyperparameters to your preparation pipeline to control
     whether or not to strip off email headers, convert each email to lowercase,
     remove punctuation, replace all URLs with “URL,” replace all numbers with
     “NUMBER,” or even perform stemming (i.e., trim off word endings; there are
     Python libraries available to do this).
   - Then try out several classifiers and see if you can build a great spam classifier,
     with both high recall and high precision.

Solutions to these exercises are available in the online Jupyter notebooks at https://
github.com/ageron/handson-ml2.

```
Exercises | 111
```

**CHAPTER 4**

**Training Models**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 4 in the final
release of the book.
```

So far we have treated Machine Learning models and their training algorithms mostly
like black boxes. If you went through some of the exercises in the previous chapters,
you may have been surprised by how much you can get done without knowing any‐
thing about what’s under the hood: you optimized a regression system, you improved
a digit image classifier, and you even built a spam classifier from scratch—all this
without knowing how they actually work. Indeed, in many situations you don’t really
need to know the implementation details.

However, having a good understanding of how things work can help you quickly
home in on the appropriate model, the right training algorithm to use, and a good set
of hyperparameters for your task. Understanding what’s under the hood will also help
you debug issues and perform error analysis more efficiently. Lastly, most of the top‐
ics discussed in this chapter will be essential in understanding, building, and training
neural networks (discussed in Part II of this book).

In this chapter, we will start by looking at the Linear Regression model, one of the
simplest models there is. We will discuss two very different ways to train it:

- Using a direct “closed-form” equation that directly computes the model parame‐
  ters that best fit the model to the training set (i.e., the model parameters that
  minimize the cost function over the training set).

```
113
```

- Using an iterative optimization approach, called Gradient Descent (GD), that
  gradually tweaks the model parameters to minimize the cost function over the
  training set, eventually converging to the same set of parameters as the first
  method. We will look at a few variants of Gradient Descent that we will use again
  and again when we study neural networks in Part II: Batch GD, Mini-batch GD,
  and Stochastic GD.

Next we will look at Polynomial Regression, a more complex model that can fit non‐
linear datasets. Since this model has more parameters than Linear Regression, it is
more prone to overfitting the training data, so we will look at how to detect whether
or not this is the case, using learning curves, and then we will look at several regulari‐
zation techniques that can reduce the risk of overfitting the training set.

Finally, we will look at two more models that are commonly used for classification
tasks: Logistic Regression and Softmax Regression.

```
There will be quite a few math equations in this chapter, using basic
notions of linear algebra and calculus. To understand these equa‐
tions, you will need to know what vectors and matrices are, how to
transpose them, multiply them, and inverse them, and what partial
derivatives are. If you are unfamiliar with these concepts, please go
through the linear algebra and calculus introductory tutorials avail‐
able as Jupyter notebooks in the online supplemental material. For
those who are truly allergic to mathematics, you should still go
through this chapter and simply skip the equations; hopefully, the
text will be sufficient to help you understand most of the concepts.
```

### Linear Regression

In Chapter 1, we looked at a simple regression model of life satisfaction: life_satisfac‐
tion = θ 0 + θ 1 × GDP_per_capita.

This model is just a linear function of the input feature GDP_per_capita. θ 0 and θ 1 are
the model’s parameters.

More generally, a linear model makes a prediction by simply computing a weighted
sum of the input features, plus a constant called the bias term (also called the intercept
term), as shown in Equation 4-1.

```
Equation 4-1. Linear Regression model prediction
y=θ 0 +θ 1 x 1 +θ 2 x 2 +⋯+θnxn
```

- ŷ is the predicted value.

**114 | Chapter 4: Training Models**

- n is the number of features.
- xi is the ith feature value.
- θj is the jth model parameter (including the bias term θ 0 and the feature weights
  θ 1 , θ 2 , ⋯, θn).

This can be written much more concisely using a vectorized form, as shown in Equa‐
tion 4-2.

```
Equation 4-2. Linear Regression model prediction (vectorized form)
y=h θx = θ · x
```

- **θ** is the model’s parameter vector, containing the bias term θ 0 and the feature
  weights θ 1 to θn.
- **x** is the instance’s feature vector, containing x 0 to xn, with x 0 always equal to 1.
- **θ** · **x** is the dot product of the vectors **θ** and **x** , which is of course equal to
  θ 0 x 0 +θ 1 x 1 +θ 2 x 2 +⋯+θnxn.
- h **θ** is the hypothesis function, using the model parameters **θ**.

```
In Machine Learning, vectors are often represented as column vec‐
tors, which are 2D arrays with a single column. If θ and x are col‐
umn vectors, then the prediction is: y= θTx , where θT is the
transpose of θ (a row vector instead of a column vector) and θTx is
the matrix multiplication of θT and x. It is of course the same pre‐
diction, except it is now represented as a single cell matrix rather
than a scalar value. In this book we will use this notation to avoid
switching between dot products and matrix multiplications.
```

Okay, that’s the Linear Regression model, so now how do we train it? Well, recall that
training a model means setting its parameters so that the model best fits the training
set. For this purpose, we first need a measure of how well (or poorly) the model fits
the training data. In Chapter 2 we saw that the most common performance measure
of a regression model is the Root Mean Square Error (RMSE) (Equation 2-1). There‐
fore, to train a Linear Regression model, you need to find the value of **θ** that minimi‐
zes the RMSE. In practice, it is simpler to minimize the Mean Square Error (MSE)

```
Linear Regression | 115
```

```
1 It is often the case that a learning algorithm will try to optimize a different function than the performance
measure used to evaluate the final model. This is generally because that function is easier to compute, because
it has useful differentiation properties that the performance measure lacks, or because we want to constrain
the model during training, as we will see when we discuss regularization.
2 The demonstration that this returns the value of θ that minimizes the cost function is outside the scope of this
book.
```

than the RMSE, and it leads to the same result (because the value that minimizes a
function also minimizes its square root).^1

The MSE of a Linear Regression hypothesis h **θ** on a training set **X** is calculated using
Equation 4-3.

```
Equation 4-3. MSE cost function for a Linear Regression model
```

```
MSE X ,h θ =m^1 ∑
i= 1
```

```
m
θ T x i−yi^2
```

Most of these notations were presented in Chapter 2 (see “Notations” on page 43 ).
The only difference is that we write h **θ** instead of just h in order to make it clear that
the model is parametrized by the vector **θ**. To simplify notations, we will just write
MSE( **θ** ) instead of MSE( **X** , h **θ** ).

#### The Normal Equation

To find the value of **θ** that minimizes the cost function, there is a closed-form solution
—in other words, a mathematical equation that gives the result directly. This is called
the Normal Equation (Equation 4-4).^2

```
Equation 4-4. Normal Equation
```

```
θ = X T X
```

```
−1
X T y
```

- **θ** is the value of **θ** that minimizes the cost function.
- **y** is the vector of target values containing y(1) to y(m).

Let’s generate some linear-looking data to test this equation on (Figure 4-1):

```
import numpy as np
```

```
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```

**116 | Chapter 4: Training Models**

Figure 4-1. Randomly generated linear dataset

Now let’s compute **θ** using the Normal Equation. We will use the inv() function from
NumPy’s Linear Algebra module (np.linalg) to compute the inverse of a matrix, and
the dot() method for matrix multiplication:

```
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```

The actual function that we used to generate the data is y = 4 + 3x 1 + Gaussian noise.
Let’s see what the equation found:

```
>>> theta_best
array([[4.21509616],
[2.77011339]])
```

We would have hoped for θ 0 = 4 and θ 1 = 3 instead of θ 0 = 4.215 and θ 1 = 2.770. Close
enough, but the noise made it impossible to recover the exact parameters of the origi‐
nal function.

Now you can make predictions using **θ** :

```
>>> X_new = np.array([[0], [2]])
>>> X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
>>> y_predict = X_new_b.dot(theta_best)
>>> y_predict
array([[4.21509616],
[9.75532293]])
```

Let’s plot this model’s predictions (Figure 4-2):

```
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
```

```
Linear Regression | 117
```

```
3 Note that Scikit-Learn separates the bias term (intercept_) from the feature weights (coef_).
```

```
plt.axis([0, 2, 0, 15])
plt.show()
```

Figure 4-2. Linear Regression model predictions

Performing linear regression using Scikit-Learn is quite simple:^3

```
>>> from sklearn.linear_model import LinearRegression
>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([4.21509616]), array([[2.77011339]]))
>>> lin_reg.predict(X_new)
array([[4.21509616],
[9.75532293]])
```

The LinearRegression class is based on the scipy.linalg.lstsq() function (the
name stands for “least squares”), which you could call directly:

```
>>> theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
>>> theta_best_svd
array([[4.21509616],
[2.77011339]])
```

This function computes **θ** = **X** + **y** , where 𝐀+ is the pseudoinverse of **X** (specifically the
Moore-Penrose inverse). You can use np.linalg.pinv() to compute the pseudoin‐
verse directly:

```
>>> np.linalg.pinv(X_b).dot(y)
array([[4.21509616],
[2.77011339]])
```

**118 | Chapter 4: Training Models**

The pseudoinverse itself is computed using a standard matrix factorization technique
called Singular Value Decomposition (SVD) that can decompose the training set
matrix **X** into the matrix multiplication of three matrices **U Σ V** T (see

numpy.linalg.svd()). The pseudoinverse is computed as **X** += **VΣ** + **U** T. To compute
the matrix **Σ** +, the algorithm takes **Σ** and sets to zero all values smaller than a tiny
threshold value, then it replaces all the non-zero values with their inverse, and finally
it transposes the resulting matrix. This approach is more efficient than computing the
Normal Equation, plus it handles edge cases nicely: indeed, the Normal Equation may
not work if the matrix **X** T **X** is not invertible (i.e., singular), such as if m < n or if some
features are redundant, but the pseudoinverse is always defined.

#### Computational Complexity

The Normal Equation computes the inverse of **X** T **X** , which is an (n + 1) × (n + 1)
matrix (where n is the number of features). The computational complexity of inverting
such a matrix is typically about O(n2.4) to O(n^3 ) (depending on the implementation).
In other words, if you double the number of features, you multiply the computation
time by roughly 22.4 = 5.3 to 2^3 = 8.

The SVD approach used by Scikit-Learn’s LinearRegression class is about O(n^2 ). If
you double the number of features, you multiply the computation time by roughly 4.

```
Both the Normal Equation and the SVD approach get very slow
when the number of features grows large (e.g., 100,000). On the
positive side, both are linear with regards to the number of instan‐
ces in the training set (they are O(m)), so they handle large training
sets efficiently, provided they can fit in memory.
```

Also, once you have trained your Linear Regression model (using the Normal Equa‐
tion or any other algorithm), predictions are very fast: the computational complexity
is linear with regards to both the number of instances you want to make predictions
on and the number of features. In other words, making predictions on twice as many
instances (or twice as many features) will just take roughly twice as much time.

Now we will look at very different ways to train a Linear Regression model, better
suited for cases where there are a large number of features, or too many training
instances to fit in memory.

### Gradient Descent

Gradient Descent is a very generic optimization algorithm capable of finding optimal
solutions to a wide range of problems. The general idea of Gradient Descent is to
tweak parameters iteratively in order to minimize a cost function.

```
Gradient Descent | 119
```

Suppose you are lost in the mountains in a dense fog; you can only feel the slope of
the ground below your feet. A good strategy to get to the bottom of the valley quickly
is to go downhill in the direction of the steepest slope. This is exactly what Gradient
Descent does: it measures the local gradient of the error function with regards to the
parameter vector **θ** , and it goes in the direction of descending gradient. Once the gra‐
dient is zero, you have reached a minimum!

Concretely, you start by filling **θ** with random values (this is called random initializa‐
tion), and then you improve it gradually, taking one baby step at a time, each step
attempting to decrease the cost function (e.g., the MSE), until the algorithm converges
to a minimum (see Figure 4-3).

Figure 4-3. Gradient Descent

An important parameter in Gradient Descent is the size of the steps, determined by
the learning rate hyperparameter. If the learning rate is too small, then the algorithm
will have to go through many iterations to converge, which will take a long time (see
Figure 4-4).

**120 | Chapter 4: Training Models**

Figure 4-4. Learning rate too small

On the other hand, if the learning rate is too high, you might jump across the valley
and end up on the other side, possibly even higher up than you were before. This
might make the algorithm diverge, with larger and larger values, failing to find a good
solution (see Figure 4-5).

Figure 4-5. Learning rate too large

Finally, not all cost functions look like nice regular bowls. There may be holes, ridges,
plateaus, and all sorts of irregular terrains, making convergence to the minimum very
difficult. Figure 4-6 shows the two main challenges with Gradient Descent: if the ran‐
dom initialization starts the algorithm on the left, then it will converge to a local mini‐
mum, which is not as good as the global minimum. If it starts on the right, then it will
take a very long time to cross the plateau, and if you stop too early you will never
reach the global minimum.

```
Gradient Descent | 121
```

```
4 Technically speaking, its derivative is Lipschitz continuous.
5 Since feature 1 is smaller, it takes a larger change in θ 1 to affect the cost function, which is why the bowl is
elongated along the θ 1 axis.
```

Figure 4-6. Gradient Descent pitfalls

Fortunately, the MSE cost function for a Linear Regression model happens to be a
convex function, which means that if you pick any two points on the curve, the line
segment joining them never crosses the curve. This implies that there are no local
minima, just one global minimum. It is also a continuous function with a slope that
never changes abruptly.^4 These two facts have a great consequence: Gradient Descent
is guaranteed to approach arbitrarily close the global minimum (if you wait long
enough and if the learning rate is not too high).

In fact, the cost function has the shape of a bowl, but it can be an elongated bowl if
the features have very different scales. Figure 4-7 shows Gradient Descent on a train‐
ing set where features 1 and 2 have the same scale (on the left), and on a training set
where feature 1 has much smaller values than feature 2 (on the right).^5

Figure 4-7. Gradient Descent with and without feature scaling

**122 | Chapter 4: Training Models**

As you can see, on the left the Gradient Descent algorithm goes straight toward the
minimum, thereby reaching it quickly, whereas on the right it first goes in a direction
almost orthogonal to the direction of the global minimum, and it ends with a long
march down an almost flat valley. It will eventually reach the minimum, but it will
take a long time.

```
When using Gradient Descent, you should ensure that all features
have a similar scale (e.g., using Scikit-Learn’s StandardScaler
class), or else it will take much longer to converge.
```

This diagram also illustrates the fact that training a model means searching for a
combination of model parameters that minimizes a cost function (over the training
set). It is a search in the model’s parameter space: the more parameters a model has,
the more dimensions this space has, and the harder the search is: searching for a nee‐
dle in a 300-dimensional haystack is much trickier than in three dimensions. Fortu‐
nately, since the cost function is convex in the case of Linear Regression, the needle is
simply at the bottom of the bowl.

#### Batch Gradient Descent

To implement Gradient Descent, you need to compute the gradient of the cost func‐
tion with regards to each model parameter θj. In other words, you need to calculate
how much the cost function will change if you change θj just a little bit. This is called
a partial derivative. It is like asking “what is the slope of the mountain under my feet
if I face east?” and then asking the same question facing north (and so on for all other
dimensions, if you can imagine a universe with more than three dimensions). Equa‐
tion 4-5 computes the partial derivative of the cost function with regards to parame‐

ter θj, noted ∂∂θ
j

```
MSE( θ ).
```

```
Equation 4-5. Partial derivatives of the cost function
```

```
∂
∂θjMSE θ =
```

```
2
mi∑= 1
```

```
m
θ T x i−yi xji
```

Instead of computing these partial derivatives individually, you can use Equation 4-6
to compute them all in one go. The gradient vector, noted ∇ **θ** MSE( **θ** ), contains all the
partial derivatives of the cost function (one for each model parameter).

```
Gradient Descent | 123
```

```
6 Eta (η) is the 7th letter of the Greek alphabet.
```

```
Equation 4-6. Gradient vector of the cost function
```

```
∇ θ MSE θ =
```

```
∂
∂θ 0 MSE θ
∂
∂θ 1 MSE θ
⋮
∂
∂θnMSE θ
```

```
=m^2 X T Xθ − y
```

```
Notice that this formula involves calculations over the full training
set X , at each Gradient Descent step! This is why the algorithm is
called Batch Gradient Descent: it uses the whole batch of training
data at every step (actually, Full Gradient Descent would probably
be a better name). As a result it is terribly slow on very large train‐
ing sets (but we will see much faster Gradient Descent algorithms
shortly). However, Gradient Descent scales well with the number of
features; training a Linear Regression model when there are hun‐
dreds of thousands of features is much faster using Gradient
Descent than using the Normal Equation or SVD decomposition.
```

Once you have the gradient vector, which points uphill, just go in the opposite direc‐
tion to go downhill. This means subtracting ∇ **θ** MSE( **θ** ) from **θ**. This is where the
learning rate η comes into play:^6 multiply the gradient vector by η to determine the
size of the downhill step (Equation 4-7).

```
Equation 4-7. Gradient Descent step
```

```
θ next step= θ −η∇ θ MSE θ
```

Let’s look at a quick implementation of this algorithm:

```
eta = 0.1 # learning rate
n_iterations = 1000
m = 100
```

```
theta = np.random.randn(2,1) # random initialization
```

```
for iteration in range(n_iterations):
gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
theta = theta - eta * gradients
```

**124 | Chapter 4: Training Models**

That wasn’t too hard! Let’s look at the resulting theta:

```
>>> theta
array([[4.21509616],
[2.77011339]])
```

Hey, that’s exactly what the Normal Equation found! Gradient Descent worked per‐
fectly. But what if you had used a different learning rate eta? Figure 4-8 shows the
first 10 steps of Gradient Descent using three different learning rates (the dashed line
represents the starting point).

Figure 4-8. Gradient Descent with various learning rates

On the left, the learning rate is too low: the algorithm will eventually reach the solu‐
tion, but it will take a long time. In the middle, the learning rate looks pretty good: in
just a few iterations, it has already converged to the solution. On the right, the learn‐
ing rate is too high: the algorithm diverges, jumping all over the place and actually
getting further and further away from the solution at every step.

To find a good learning rate, you can use grid search (see Chapter 2). However, you
may want to limit the number of iterations so that grid search can eliminate models
that take too long to converge.

You may wonder how to set the number of iterations. If it is too low, you will still be
far away from the optimal solution when the algorithm stops, but if it is too high, you
will waste time while the model parameters do not change anymore. A simple solu‐
tion is to set a very large number of iterations but to interrupt the algorithm when the
gradient vector becomes tiny—that is, when its norm becomes smaller than a tiny
number ε (called the tolerance)—because this happens when Gradient Descent has
(almost) reached the minimum.

```
Gradient Descent | 125
```

```
7 Out-of-core algorithms are discussed in Chapter 1.
```

```
Convergence Rate
When the cost function is convex and its slope does not change abruptly (as is the
case for the MSE cost function), Batch Gradient Descent with a fixed learning rate
will eventually converge to the optimal solution, but you may have to wait a while: it
can take O(1/ε) iterations to reach the optimum within a range of ε depending on the
shape of the cost function. If you divide the tolerance by 10 to have a more precise
solution, then the algorithm may have to run about 10 times longer.
```

#### Stochastic Gradient Descent

The main problem with Batch Gradient Descent is the fact that it uses the whole
training set to compute the gradients at every step, which makes it very slow when
the training set is large. At the opposite extreme, Stochastic Gradient Descent just
picks a random instance in the training set at every step and computes the gradients
based only on that single instance. Obviously this makes the algorithm much faster
since it has very little data to manipulate at every iteration. It also makes it possible to
train on huge training sets, since only one instance needs to be in memory at each
iteration (SGD can be implemented as an out-of-core algorithm.^7 )

On the other hand, due to its stochastic (i.e., random) nature, this algorithm is much
less regular than Batch Gradient Descent: instead of gently decreasing until it reaches
the minimum, the cost function will bounce up and down, decreasing only on aver‐
age. Over time it will end up very close to the minimum, but once it gets there it will
continue to bounce around, never settling down (see Figure 4-9). So once the algo‐
rithm stops, the final parameter values are good, but not optimal.

Figure 4-9. Stochastic Gradient Descent

**126 | Chapter 4: Training Models**

When the cost function is very irregular (as in Figure 4-6), this can actually help the
algorithm jump out of local minima, so Stochastic Gradient Descent has a better
chance of finding the global minimum than Batch Gradient Descent does.

Therefore randomness is good to escape from local optima, but bad because it means
that the algorithm can never settle at the minimum. One solution to this dilemma is
to gradually reduce the learning rate. The steps start out large (which helps make
quick progress and escape local minima), then get smaller and smaller, allowing the
algorithm to settle at the global minimum. This process is akin to simulated anneal‐
ing, an algorithm inspired from the process of annealing in metallurgy where molten
metal is slowly cooled down. The function that determines the learning rate at each
iteration is called the learning schedule. If the learning rate is reduced too quickly, you
may get stuck in a local minimum, or even end up frozen halfway to the minimum. If
the learning rate is reduced too slowly, you may jump around the minimum for a
long time and end up with a suboptimal solution if you halt training too early.

This code implements Stochastic Gradient Descent using a simple learning schedule:

```
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
```

```
def learning_schedule(t):
return t0 / (t + t1)
```

```
theta = np.random.randn(2,1) # random initialization
```

```
for epoch in range(n_epochs):
for i in range(m):
random_index = np.random.randint(m)
xi = X_b[random_index:random_index+1]
yi = y[random_index:random_index+1]
gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
eta = learning_schedule(epoch * m + i)
theta = theta - eta * gradients
```

By convention we iterate by rounds of m iterations; each round is called an epoch.
While the Batch Gradient Descent code iterated 1,000 times through the whole train‐
ing set, this code goes through the training set only 50 times and reaches a fairly good
solution:

```
>>> theta
array([[4.21076011],
[2.74856079]])
```

Figure 4-10 shows the first 20 steps of training (notice how irregular the steps are).

```
Gradient Descent | 127
```

Figure 4-10. Stochastic Gradient Descent first 20 steps

Note that since instances are picked randomly, some instances may be picked several
times per epoch while others may not be picked at all. If you want to be sure that the
algorithm goes through every instance at each epoch, another approach is to shuffle
the training set (making sure to shuffle the input features and the labels jointly), then
go through it instance by instance, then shuffle it again, and so on. However, this gen‐
erally converges more slowly.

```
When using Stochastic Gradient Descent, the training instances
must be independent and identically distributed (IID), to ensure
that the parameters get pulled towards the global optimum, on
average. A simple way to ensure this is to shuffle the instances dur‐
ing training (e.g., pick each instance randomly, or shuffle the train‐
ing set at the beginning of each epoch). If you do not do this, for
example if the instances are sorted by label, then SGD will start by
optimizing for one label, then the next, and so on, and it will not
settle close to the global minimum.
```

To perform Linear Regression using SGD with Scikit-Learn, you can use the SGDRe
gressor class, which defaults to optimizing the squared error cost function. The fol‐
lowing code runs for maximum 1000 epochs (max_iter=1000) or until the loss drops
by less than 1e-3 during one epoch (tol=1e-3), starting with a learning rate of 0.1
(eta0=0.1), using the default learning schedule (different from the preceding one),
and it does not use any regularization (penalty=None; more details on this shortly):

```
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
```

**128 | Chapter 4: Training Models**

Once again, you find a solution quite close to the one returned by the Normal Equa‐
tion:

```
>>> sgd_reg.intercept_, sgd_reg.coef_
(array([4.24365286]), array([2.8250878]))
```

#### Mini-batch Gradient Descent

The last Gradient Descent algorithm we will look at is called Mini-batch Gradient
Descent. It is quite simple to understand once you know Batch and Stochastic Gradi‐
ent Descent: at each step, instead of computing the gradients based on the full train‐
ing set (as in Batch GD) or based on just one instance (as in Stochastic GD), Mini-
batch GD computes the gradients on small random sets of instances called mini-
batches. The main advantage of Mini-batch GD over Stochastic GD is that you can
get a performance boost from hardware optimization of matrix operations, especially
when using GPUs.

The algorithm’s progress in parameter space is less erratic than with SGD, especially
with fairly large mini-batches. As a result, Mini-batch GD will end up walking
around a bit closer to the minimum than SGD. But, on the other hand, it may be
harder for it to escape from local minima (in the case of problems that suffer from
local minima, unlike Linear Regression as we saw earlier). Figure 4-11 shows the
paths taken by the three Gradient Descent algorithms in parameter space during
training. They all end up near the minimum, but Batch GD’s path actually stops at the
minimum, while both Stochastic GD and Mini-batch GD continue to walk around.
However, don’t forget that Batch GD takes a lot of time to take each step, and Stochas‐
tic GD and Mini-batch GD would also reach the minimum if you used a good learn‐
ing schedule.

Figure 4-11. Gradient Descent paths in parameter space

```
Gradient Descent | 129
```

```
8 While the Normal Equation can only perform Linear Regression, the Gradient Descent algorithms can be
used to train many other models, as we will see.
9 A quadratic equation is of the form y = ax^2 + bx + c.
```

Let’s compare the algorithms we’ve discussed so far for Linear Regression^8 (recall that
m is the number of training instances and n is the number of features); see Table 4-1.

Table 4-1. Comparison of algorithms for Linear Regression

```
Algorithm Large m Out-of-core support Large n Hyperparams Scaling required Scikit-Learn
Normal Equation Fast No Slow 0 No n/a
SVD Fast No Slow 0 No LinearRegression
Batch GD Slow No Fast 2 Yes SGDRegressor
Stochastic GD Fast Yes Fast ≥2 Yes SGDRegressor
Mini-batch GD Fast Yes Fast ≥2 Yes SGDRegressor
```

```
There is almost no difference after training: all these algorithms
end up with very similar models and make predictions in exactly
the same way.
```

### Polynomial Regression

What if your data is actually more complex than a simple straight line? Surprisingly,
you can actually use a linear model to fit nonlinear data. A simple way to do this is to
add powers of each feature as new features, then train a linear model on this extended
set of features. This technique is called Polynomial Regression.

Let’s look at an example. First, let’s generate some nonlinear data, based on a simple
quadratic equation^9 (plus some noise; see Figure 4-12):

```
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
```

**130 | Chapter 4: Training Models**

Figure 4-12. Generated nonlinear and noisy dataset

Clearly, a straight line will never fit this data properly. So let’s use Scikit-Learn’s Poly
nomialFeatures class to transform our training data, adding the square (2nd-degree
polynomial) of each feature in the training set as new features (in this case there is
just one feature):

```
>>> from sklearn.preprocessing import PolynomialFeatures
>>> poly_features = PolynomialFeatures(degree=2, include_bias=False)
>>> X_poly = poly_features.fit_transform(X)
>>> X[0]
array([-0.75275929])
>>> X_poly[0]
array([-0.75275929, 0.56664654])
```

X_poly now contains the original feature of X plus the square of this feature. Now you
can fit a LinearRegression model to this extended training data (Figure 4-13):

```
>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X_poly, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([1.78134581]), array([[0.93366893, 0.56456263]]))
```

```
Polynomial Regression | 131
```

Figure 4-13. Polynomial Regression model predictions

Not bad: the model estimates y= 0. 56x 12 + 0. 93x 1 + 1. 78 when in fact the original

function was y= 0. 5x 12 + 1. 0x 1 + 2. 0 + Gaussian noise.

Note that when there are multiple features, Polynomial Regression is capable of find‐
ing relationships between features (which is something a plain Linear Regression
model cannot do). This is made possible by the fact that PolynomialFeatures also
adds all combinations of features up to the given degree. For example, if there were
two features a and b, PolynomialFeatures with degree=3 would not only add the
features a^2 , a^3 , b^2 , and b^3 , but also the combinations ab, a^2 b, and ab^2.

```
PolynomialFeatures(degree=d) transforms an array containing n
features into an array containing nd+!nd!! features, where n! is the
factorial of n, equal to 1 × 2 × 3 × ⋯ × n. Beware of the combinato‐
rial explosion of the number of features!
```

### Learning Curves

If you perform high-degree Polynomial Regression, you will likely fit the training
data much better than with plain Linear Regression. For example, Figure 4-14 applies
a 300-degree polynomial model to the preceding training data, and compares the
result with a pure linear model and a quadratic model (2nd-degree polynomial).
Notice how the 300-degree polynomial model wiggles around to get as close as possi‐
ble to the training instances.

**132 | Chapter 4: Training Models**

Figure 4-14. High-degree Polynomial Regression

Of course, this high-degree Polynomial Regression model is severely overfitting the
training data, while the linear model is underfitting it. The model that will generalize
best in this case is the quadratic model. It makes sense since the data was generated
using a quadratic model, but in general you won’t know what function generated the
data, so how can you decide how complex your model should be? How can you tell
that your model is overfitting or underfitting the data?

In Chapter 2 you used cross-validation to get an estimate of a model’s generalization
performance. If a model performs well on the training data but generalizes poorly
according to the cross-validation metrics, then your model is overfitting. If it per‐
forms poorly on both, then it is underfitting. This is one way to tell when a model is
too simple or too complex.

Another way is to look at the learning curves: these are plots of the model’s perfor‐
mance on the training set and the validation set as a function of the training set size
(or the training iteration). To generate the plots, simply train the model several times
on different sized subsets of the training set. The following code defines a function
that plots the learning curves of a model given some training data:

```
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

```
def plot_learning_curves(model, X, y):
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
train_errors, val_errors = [], []
for m in range(1, len(X_train)):
model.fit(X_train[:m], y_train[:m])
y_train_predict = model.predict(X_train[:m])
```

```
Learning Curves | 133
```

```
y_val_predict = model.predict(X_val)
train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
val_errors.append(mean_squared_error(y_val, y_val_predict))
plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
```

Let’s look at the learning curves of the plain Linear Regression model (a straight line;
Figure 4-15):

```
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
```

Figure 4-15. Learning curves

This deserves a bit of explanation. First, let’s look at the performance on the training
data: when there are just one or two instances in the training set, the model can fit
them perfectly, which is why the curve starts at zero. But as new instances are added
to the training set, it becomes impossible for the model to fit the training data per‐
fectly, both because the data is noisy and because it is not linear at all. So the error on
the training data goes up until it reaches a plateau, at which point adding new instan‐
ces to the training set doesn’t make the average error much better or worse. Now let’s
look at the performance of the model on the validation data. When the model is
trained on very few training instances, it is incapable of generalizing properly, which
is why the validation error is initially quite big. Then as the model is shown more
training examples, it learns and thus the validation error slowly goes down. However,
once again a straight line cannot do a good job modeling the data, so the error ends
up at a plateau, very close to the other curve.

These learning curves are typical of an underfitting model. Both curves have reached
a plateau; they are close and fairly high.

**134 | Chapter 4: Training Models**

```
If your model is underfitting the training data, adding more train‐
ing examples will not help. You need to use a more complex model
or come up with better features.
```

Now let’s look at the learning curves of a 10th-degree polynomial model on the same
data (Figure 4-16):

```
from sklearn.pipeline import Pipeline
```

```
polynomial_regression = Pipeline([
("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
("lin_reg", LinearRegression()),
])
```

```
plot_learning_curves(polynomial_regression, X, y)
```

These learning curves look a bit like the previous ones, but there are two very impor‐
tant differences:

- The error on the training data is much lower than with the Linear Regression
  model.
- There is a gap between the curves. This means that the model performs signifi‐
  cantly better on the training data than on the validation data, which is the hall‐
  mark of an overfitting model. However, if you used a much larger training set,
  the two curves would continue to get closer.

Figure 4-16. Learning curves for the polynomial model

```
Learning Curves | 135
```

10 This notion of bias is not to be confused with the bias term of linear models.

```
One way to improve an overfitting model is to feed it more training
data until the validation error reaches the training error.
```

```
The Bias/Variance Tradeoff
An important theoretical result of statistics and Machine Learning is the fact that a
model’s generalization error can be expressed as the sum of three very different
errors:
Bias
This part of the generalization error is due to wrong assumptions, such as assum‐
ing that the data is linear when it is actually quadratic. A high-bias model is most
likely to underfit the training data.^10
Variance
This part is due to the model’s excessive sensitivity to small variations in the
training data. A model with many degrees of freedom (such as a high-degree pol‐
ynomial model) is likely to have high variance, and thus to overfit the training
data.
Irreducible error
This part is due to the noisiness of the data itself. The only way to reduce this
part of the error is to clean up the data (e.g., fix the data sources, such as broken
sensors, or detect and remove outliers).
Increasing a model’s complexity will typically increase its variance and reduce its bias.
Conversely, reducing a model’s complexity increases its bias and reduces its variance.
This is why it is called a tradeoff.
```

### Regularized Linear Models

```
As we saw in Chapters 1 and 2, a good way to reduce overfitting is to regularize the
model (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be
for it to overfit the data. For example, a simple way to regularize a polynomial model
is to reduce the number of polynomial degrees.
For a linear model, regularization is typically achieved by constraining the weights of
the model. We will now look at Ridge Regression, Lasso Regression, and Elastic Net,
which implement three different ways to constrain the weights.
```

```
136 | Chapter 4: Training Models
```

11 It is common to use the notation J( **θ** ) for cost functions that don’t have a short name; we will often use this
notation throughout the rest of this book. The context will make it clear which cost function is being dis‐
cussed.
12 Norms are discussed in Chapter 2.

#### Ridge Regression

```
Ridge Regression (also called Tikhonov regularization) is a regularized version of Lin‐
ear Regression: a regularization term equal to α∑in= 1θi^2 is added to the cost function.
This forces the learning algorithm to not only fit the data but also keep the model
weights as small as possible. Note that the regularization term should only be added
to the cost function during training. Once the model is trained, you want to evaluate
the model’s performance using the unregularized performance measure.
```

```
It is quite common for the cost function used during training to be
different from the performance measure used for testing. Apart
from regularization, another reason why they might be different is
that a good training cost function should have optimization-
friendly derivatives, while the performance measure used for test‐
ing should be as close as possible to the final objective. A good
example of this is a classifier trained using a cost function such as
the log loss (discussed in a moment) but evaluated using precision/
recall.
```

```
The hyperparameter α controls how much you want to regularize the model. If α = 0
then Ridge Regression is just Linear Regression. If α is very large, then all weights end
up very close to zero and the result is a flat line going through the data’s mean. Equa‐
tion 4-8 presents the Ridge Regression cost function.^11
```

```
Equation 4-8. Ridge Regression cost function
```

```
J θ = MSE θ +α^12 ∑in= 1θi^2
```

```
Note that the bias term θ 0 is not regularized (the sum starts at i = 1, not 0). If we
define w as the vector of feature weights (θ 1 to θn), then the regularization term is
simply equal to ½(∥ w ∥ 2 )^2 , where ∥ w ∥ 2 represents the ℓ 2 norm of the weight vector.^12
For Gradient Descent, just add α w to the MSE gradient vector (Equation 4-6).
```

```
It is important to scale the data (e.g., using a StandardScaler)
before performing Ridge Regression, as it is sensitive to the scale of
the input features. This is true of most regularized models.
```

```
Regularized Linear Models | 137
```

13 A square matrix full of 0s except for 1s on the main diagonal (top-left to bottom-right).

```
Figure 4-17 shows several Ridge models trained on some linear data using different α
value. On the left, plain Ridge models are used, leading to linear predictions. On the
right, the data is first expanded using PolynomialFeatures(degree=10), then it is
scaled using a StandardScaler, and finally the Ridge models are applied to the result‐
ing features: this is Polynomial Regression with Ridge regularization. Note how
increasing α leads to flatter (i.e., less extreme, more reasonable) predictions; this
reduces the model’s variance but increases its bias.
As with Linear Regression, we can perform Ridge Regression either by computing a
closed-form equation or by performing Gradient Descent. The pros and cons are the
same. Equation 4-9 shows the closed-form solution (where A is the (n + 1) × (n + 1)
identity matrix^13 except with a 0 in the top-left cell, corresponding to the bias term).
```

```
Figure 4-17. Ridge Regression
```

```
Equation 4-9. Ridge Regression closed-form solution
```

```
θ = X T X +α A
```

```
−1
X T y
```

```
Here is how to perform Ridge Regression with Scikit-Learn using a closed-form solu‐
tion (a variant of Equation 4-9 using a matrix factorization technique by André-Louis
Cholesky):
>>> from sklearn.linear_model import Ridge
>>> ridge_reg = Ridge(alpha=1, solver="cholesky")
>>> ridge_reg.fit(X, y)
```

```
138 | Chapter 4: Training Models
```

14 Alternatively you can use the Ridge class with the "sag" solver. Stochastic Average GD is a variant of SGD.
For more details, see the presentation “Minimizing Finite Sums with the Stochastic Average Gradient Algo‐
rithm” by Mark Schmidt et al. from the University of British Columbia.

```
>>> ridge_reg.predict([[1.5]])
array([[1.55071465]])
And using Stochastic Gradient Descent:^14
>>> sgd_reg = SGDRegressor(penalty="l2")
>>> sgd_reg.fit(X, y.ravel())
>>> sgd_reg.predict([[1.5]])
array([1.47012588])
The penalty hyperparameter sets the type of regularization term to use. Specifying
"l2" indicates that you want SGD to add a regularization term to the cost function
equal to half the square of the ℓ 2 norm of the weight vector: this is simply Ridge
Regression.
```

#### Lasso Regression

```
Least Absolute Shrinkage and Selection Operator Regression (simply called Lasso
Regression) is another regularized version of Linear Regression: just like Ridge
Regression, it adds a regularization term to the cost function, but it uses the ℓ 1 norm
of the weight vector instead of half the square of the ℓ 2 norm (see Equation 4-10).
```

```
Equation 4-10. Lasso Regression cost function
```

```
J θ = MSE θ +α∑in= 1θi
```

```
Figure 4-18 shows the same thing as Figure 4-17 but replaces Ridge models with
Lasso models and uses smaller α values.
```

```
Regularized Linear Models | 139
```

Figure 4-18. Lasso Regression

An important characteristic of Lasso Regression is that it tends to completely elimi‐
nate the weights of the least important features (i.e., set them to zero). For example,
the dashed line in the right plot on Figure 4-18 (with α = 10-7) looks quadratic, almost
linear: all the weights for the high-degree polynomial features are equal to zero. In
other words, Lasso Regression automatically performs feature selection and outputs a
sparse model (i.e., with few nonzero feature weights).

You can get a sense of why this is the case by looking at Figure 4-19: on the top-left
plot, the background contours (ellipses) represent an unregularized MSE cost func‐
tion (α = 0), and the white circles show the Batch Gradient Descent path with that
cost function. The foreground contours (diamonds) represent the ℓ 1 penalty, and the
triangles show the BGD path for this penalty only (α → ∞). Notice how the path first
reaches θ 1 = 0, then rolls down a gutter until it reaches θ 2 = 0. On the top-right plot,
the contours represent the same cost function plus an ℓ 1 penalty with α = 0.5. The
global minimum is on the θ 2 = 0 axis. BGD first reaches θ 2 = 0, then rolls down the
gutter until it reaches the global minimum. The two bottom plots show the same
thing but uses an ℓ 2 penalty instead. The regularized minimum is closer to **θ** = **0** than
the unregularized minimum, but the weights do not get fully eliminated.

**140 | Chapter 4: Training Models**

15 You can think of a subgradient vector at a nondifferentiable point as an intermediate vector between the gra‐
dient vectors around that point.

```
Figure 4-19. Lasso versus Ridge regularization
```

```
On the Lasso cost function, the BGD path tends to bounce across
the gutter toward the end. This is because the slope changes
abruptly at θ 2 = 0. You need to gradually reduce the learning rate in
order to actually converge to the global minimum.
```

```
The Lasso cost function is not differentiable at θi = 0 (for i = 1, 2, ⋯, n), but Gradient
Descent still works fine if you use a subgradient vector g^15 instead when any θi = 0.
Equation 4-11 shows a subgradient vector equation you can use for Gradient Descent
with the Lasso cost function.
```

```
Equation 4-11. Lasso Regression subgradient vector
```

```
g θ ,J =∇ θ MSE θ +α
```

```
sign θ 1
sign θ 2
⋮
sign θn
```

```
where  sign θi =
```

```
−1 if θi< 0
0 if θi= 0
+1 if θi> 0
```

```
Regularized Linear Models | 141
```

Here is a small Scikit-Learn example using the Lasso class. Note that you could
instead use an SGDRegressor(penalty="l1").

```
>>> from sklearn.linear_model import Lasso
>>> lasso_reg = Lasso(alpha=0.1)
>>> lasso_reg.fit(X, y)
>>> lasso_reg.predict([[1.5]])
array([1.53788174])
```

#### Elastic Net

Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The
regularization term is a simple mix of both Ridge and Lasso’s regularization terms,
and you can control the mix ratio r. When r = 0, Elastic Net is equivalent to Ridge
Regression, and when r = 1, it is equivalent to Lasso Regression (see Equation 4-12).

```
Equation 4-12. Elastic Net cost function
```

```
J θ = MSE θ +rα∑in= 1θi+1 − 2 rα∑in= 1θi^2
```

So when should you use plain Linear Regression (i.e., without any regularization),
Ridge, Lasso, or Elastic Net? It is almost always preferable to have at least a little bit of
regularization, so generally you should avoid plain Linear Regression. Ridge is a good
default, but if you suspect that only a few features are actually useful, you should pre‐
fer Lasso or Elastic Net since they tend to reduce the useless features’ weights down to
zero as we have discussed. In general, Elastic Net is preferred over Lasso since Lasso
may behave erratically when the number of features is greater than the number of
training instances or when several features are strongly correlated.

Here is a short example using Scikit-Learn’s ElasticNet (l1_ratio corresponds to
the mix ratio r):

```
>>> from sklearn.linear_model import ElasticNet
>>> elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
>>> elastic_net.fit(X, y)
>>> elastic_net.predict([[1.5]])
array([1.54333232])
```

#### Early Stopping

A very different way to regularize iterative learning algorithms such as Gradient
Descent is to stop training as soon as the validation error reaches a minimum. This is
called early stopping. Figure 4-20 shows a complex model (in this case a high-degree
Polynomial Regression model) being trained using Batch Gradient Descent. As the
epochs go by, the algorithm learns and its prediction error (RMSE) on the training set
naturally goes down, and so does its prediction error on the validation set. However,

**142 | Chapter 4: Training Models**

after a while the validation error stops decreasing and actually starts to go back up.
This indicates that the model has started to overfit the training data. With early stop‐
ping you just stop training as soon as the validation error reaches the minimum. It is
such a simple and efficient regularization technique that Geoffrey Hinton called it a
“beautiful free lunch.”

Figure 4-20. Early stopping regularization

```
With Stochastic and Mini-batch Gradient Descent, the curves are
not so smooth, and it may be hard to know whether you have
reached the minimum or not. One solution is to stop only after the
validation error has been above the minimum for some time (when
you are confident that the model will not do any better), then roll
back the model parameters to the point where the validation error
was at a minimum.
```

Here is a basic implementation of early stopping:

```
from sklearn.base import clone
```

```
# prepare the data
poly_scaler = Pipeline([
("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
("std_scaler", StandardScaler())
])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)
```

```
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
penalty=None, learning_rate="constant", eta0=0.0005)
```

```
Regularized Linear Models | 143
```

```
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
y_val_predict = sgd_reg.predict(X_val_poly_scaled)
val_error = mean_squared_error(y_val, y_val_predict)
if val_error < minimum_val_error:
minimum_val_error = val_error
best_epoch = epoch
best_model = clone(sgd_reg)
```

Note that with warm_start=True, when the fit() method is called, it just continues
training where it left off instead of restarting from scratch.

### Logistic Regression

As we discussed in Chapter 1, some regression algorithms can be used for classifica‐
tion as well (and vice versa). Logistic Regression (also called Logit Regression) is com‐
monly used to estimate the probability that an instance belongs to a particular class
(e.g., what is the probability that this email is spam?). If the estimated probability is
greater than 50%, then the model predicts that the instance belongs to that class
(called the positive class, labeled “1”), or else it predicts that it does not (i.e., it
belongs to the negative class, labeled “0”). This makes it a binary classifier.

#### Estimating Probabilities

So how does it work? Just like a Linear Regression model, a Logistic Regression
model computes a weighted sum of the input features (plus a bias term), but instead
of outputting the result directly like the Linear Regression model does, it outputs the
logistic of this result (see Equation 4-13).

```
Equation 4-13. Logistic Regression model estimated probability (vectorized form)
```

```
p=h θx =σ x T θ
```

The logistic—noted σ(·)—is a sigmoid function (i.e., S-shaped) that outputs a number
between 0 and 1. It is defined as shown in Equation 4-14 and Figure 4-21.

```
Equation 4-14. Logistic function
```

```
σt =1 + exp^1 −t
```

**144 | Chapter 4: Training Models**

Figure 4-21. Logistic function

Once the Logistic Regression model has estimated the probability p = h **θ** ( **x** ) that an
instance **x** belongs to the positive class, it can make its prediction ŷ easily (see Equa‐
tion 4-15).

```
Equation 4-15. Logistic Regression model prediction
```

```
y=
```

```
0 if p< 0. 5
1 if p≥ 0. 5
```

Notice that σ(t) < 0.5 when t < 0, and σ(t) ≥ 0.5 when t ≥ 0, so a Logistic Regression
model predicts 1 if **x** T **θ** is positive, and 0 if it is negative.

```
The score t is often called the logit: this name comes from the fact
that the logit function, defined as logit(p) = log(p / (1 - p)), is the
inverse of the logistic function. Indeed, if you compute the logit of
the estimated probability p, you will find that the result is t. The
logit is also called the log-odds, since it is the log of the ratio
between the estimated probability for the positive class and the
estimated probability for the negative class.
```

#### Training and Cost Function

Good, now you know how a Logistic Regression model estimates probabilities and
makes predictions. But how is it trained? The objective of training is to set the param‐
eter vector **θ** so that the model estimates high probabilities for positive instances (y =

1. and low probabilities for negative instances (y = 0). This idea is captured by the
   cost function shown in Equation 4-16 for a single training instance **x**.

```
Equation 4-16. Cost function of a single training instance
```

```
c θ =
```

```
−logp if y= 1
−log1 −p if y= 0
```

```
Logistic Regression | 145
```

This cost function makes sense because – log(t) grows very large when t approaches
0, so the cost will be large if the model estimates a probability close to 0 for a positive
instance, and it will also be very large if the model estimates a probability close to 1
for a negative instance. On the other hand, – log(t) is close to 0 when t is close to 1, so
the cost will be close to 0 if the estimated probability is close to 0 for a negative
instance or close to 1 for a positive instance, which is precisely what we want.

The cost function over the whole training set is simply the average cost over all train‐
ing instances. It can be written in a single expression (as you can verify easily), called
the log loss, shown in Equation 4-17.

```
Equation 4-17. Logistic Regression cost function (log loss)
```

```
J θ = −m^1 ∑im= 1yilogpi + 1 −yi log1 −pi
```

The bad news is that there is no known closed-form equation to compute the value of
**θ** that minimizes this cost function (there is no equivalent of the Normal Equation).
But the good news is that this cost function is convex, so Gradient Descent (or any
other optimization algorithm) is guaranteed to find the global minimum (if the learn‐
ing rate is not too large and you wait long enough). The partial derivatives of the cost
function with regards to the jth model parameter θj is given by Equation 4-18.

```
Equation 4-18. Logistic cost function partial derivatives
```

```
∂
∂θjJ θ =
```

```
1
mi∑= 1
```

```
m
σ θ T x i −yi xji
```

This equation looks very much like Equation 4-5: for each instance it computes the
prediction error and multiplies it by the jth feature value, and then it computes the
average over all training instances. Once you have the gradient vector containing all
the partial derivatives you can use it in the Batch Gradient Descent algorithm. That’s
it: you now know how to train a Logistic Regression model. For Stochastic GD you
would of course just take one instance at a time, and for Mini-batch GD you would
use a mini-batch at a time.

#### Decision Boundaries

Let’s use the iris dataset to illustrate Logistic Regression. This is a famous dataset that
contains the sepal and petal length and width of 150 iris flowers of three different
species: Iris-Setosa, Iris-Versicolor, and Iris-Virginica (see Figure 4-22).

**146 | Chapter 4: Training Models**

16 Photos reproduced from the corresponding Wikipedia pages. Iris-Virginica photo by Frank Mayfield (Crea‐
tive Commons BY-SA 2.0), Iris-Versicolor photo by D. Gordon E. Robertson (Creative Commons BY-SA 3.0),
and Iris-Setosa photo is public domain.
17 NumPy’s reshape() function allows one dimension to be –1, which means “unspecified”: the value is inferred
from the length of the array and the remaining dimensions.

```
Figure 4-22. Flowers of three iris plant species^16
```

```
Let’s try to build a classifier to detect the Iris-Virginica type based only on the petal
width feature. First let’s load the data:
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> list(iris.keys())
['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
>>> X = iris["data"][:, 3:] # petal width
>>> y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0
Now let’s train a Logistic Regression model:
from sklearn.linear_model import LogisticRegression
```

```
log_reg = LogisticRegression()
log_reg.fit(X, y)
Let’s look at the model’s estimated probabilities for flowers with petal widths varying
from 0 to 3 cm (Figure 4-23)^17 :
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
```

```
Logistic Regression | 147
```

18 It is the the set of points **x** such that θ 0 + θ 1 x 1 + θ 2 x 2 = 0, which defines a straight line.

```
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
# + more Matplotlib code to make the image look pretty
```

```
Figure 4-23. Estimated probabilities and decision boundary
```

```
The petal width of Iris-Virginica flowers (represented by triangles) ranges from 1.4
cm to 2.5 cm, while the other iris flowers (represented by squares) generally have a
smaller petal width, ranging from 0.1 cm to 1.8 cm. Notice that there is a bit of over‐
lap. Above about 2 cm the classifier is highly confident that the flower is an Iris-
Virginica (it outputs a high probability to that class), while below 1 cm it is highly
confident that it is not an Iris-Virginica (high probability for the “Not Iris-Virginica”
class). In between these extremes, the classifier is unsure. However, if you ask it to
predict the class (using the predict() method rather than the predict_proba()
method), it will return whichever class is the most likely. Therefore, there is a decision
boundary at around 1.6 cm where both probabilities are equal to 50%: if the petal
width is higher than 1.6 cm, the classifier will predict that the flower is an Iris-
Virginica, or else it will predict that it is not (even if it is not very confident):
>>> log_reg.predict([[1.7], [1.5]])
array([1, 0])
Figure 4-24 shows the same dataset but this time displaying two features: petal width
and length. Once trained, the Logistic Regression classifier can estimate the probabil‐
ity that a new flower is an Iris-Virginica based on these two features. The dashed line
represents the points where the model estimates a 50% probability: this is the model’s
decision boundary. Note that it is a linear boundary.^18 Each parallel line represents the
points where the model outputs a specific probability, from 15% (bottom left) to 90%
(top right). All the flowers beyond the top-right line have an over 90% chance of
being Iris-Virginica according to the model.
```

```
148 | Chapter 4: Training Models
```

Figure 4-24. Linear decision boundary

Just like the other linear models, Logistic Regression models can be regularized using
ℓ 1 or ℓ 2 penalties. Scitkit-Learn actually adds an ℓ 2 penalty by default.

```
The hyperparameter controlling the regularization strength of a
Scikit-Learn LogisticRegression model is not alpha (as in other
linear models), but its inverse: C. The higher the value of C, the less
the model is regularized.
```

#### Softmax Regression

The Logistic Regression model can be generalized to support multiple classes directly,
without having to train and combine multiple binary classifiers (as discussed in
Chapter 3). This is called Softmax Regression, or Multinomial Logistic Regression.

The idea is quite simple: when given an instance **x** , the Softmax Regression model
first computes a score sk( **x** ) for each class k, then estimates the probability of each
class by applying the softmax function (also called the normalized exponential) to the
scores. The equation to compute sk( **x** ) should look familiar, as it is just like the equa‐
tion for Linear Regression prediction (see Equation 4-19).

```
Equation 4-19. Softmax score for class k
```

```
sk x = x T θ k
```

Note that each class has its own dedicated parameter vector **θ** (k). All these vectors are
typically stored as rows in a parameter matrix **Θ**.

Once you have computed the score of every class for the instance **x** , you can estimate
the probability pk that the instance belongs to class k by running the scores through
the softmax function (Equation 4-20): it computes the exponential of every score,

```
Logistic Regression | 149
```

then normalizes them (dividing by the sum of all the exponentials). The scores are
generally called logits or log-odds (although they are actually unnormalized log-
odds).

```
Equation 4-20. Softmax function
```

```
pk=σ sx k=
```

```
exp sk x
∑Kj= 1exp sj x
```

- K is the number of classes.
- **s** ( **x** ) is a vector containing the scores of each class for the instance **x**.
- σ( **s** ( **x** ))k is the estimated probability that the instance **x** belongs to class k given
  the scores of each class for that instance.

Just like the Logistic Regression classifier, the Softmax Regression classifier predicts
the class with the highest estimated probability (which is simply the class with the
highest score), as shown in Equation 4-21.

```
Equation 4-21. Softmax Regression classifier prediction
```

```
y= argmax
k
```

```
σ sx k= argmax
k
```

```
sk x = argmax
k
```

```
θ k T x
```

- The argmax operator returns the value of a variable that maximizes a function. In
  this equation, it returns the value of k that maximizes the estimated probability
  σ( **s** ( **x** ))k.

```
The Softmax Regression classifier predicts only one class at a time
(i.e., it is multiclass, not multioutput) so it should be used only with
mutually exclusive classes such as different types of plants. You
cannot use it to recognize multiple people in one picture.
```

Now that you know how the model estimates probabilities and makes predictions,
let’s take a look at training. The objective is to have a model that estimates a high
probability for the target class (and consequently a low probability for the other
classes). Minimizing the cost function shown in Equation 4-22, called the cross
entropy, should lead to this objective because it penalizes the model when it estimates
a low probability for a target class. Cross entropy is frequently used to measure how

**150 | Chapter 4: Training Models**

well a set of estimated class probabilities match the target classes (we will use it again
several times in the following chapters).

```
Equation 4-22. Cross entropy cost function
```

```
J Θ = −m^1 ∑im= 1∑kK= 1ykilogpki
```

- yki is the target probability that the ith instance belongs to class k. In general, it is
  either equal to 1 or 0, depending on whether the instance belongs to the class or
  not.

Notice that when there are just two classes (K = 2), this cost function is equivalent to
the Logistic Regression’s cost function (log loss; see Equation 4-17).

```
Cross Entropy
Cross entropy originated from information theory. Suppose you want to efficiently
transmit information about the weather every day. If there are eight options (sunny,
rainy, etc.), you could encode each option using 3 bits since 2^3 = 8. However, if you
think it will be sunny almost every day, it would be much more efficient to code
“sunny” on just one bit (0) and the other seven options on 4 bits (starting with a 1).
Cross entropy measures the average number of bits you actually send per option. If
your assumption about the weather is perfect, cross entropy will just be equal to the
entropy of the weather itself (i.e., its intrinsic unpredictability). But if your assump‐
tions are wrong (e.g., if it rains often), cross entropy will be greater by an amount
called the Kullback–Leibler divergence.
The cross entropy between two probability distributions p and q is defined as
Hp,q = − ∑xpx logqx (at least when the distributions are discrete). For more
details, check out this video.
```

The gradient vector of this cost function with regards to **θ** (k) is given by Equation
4-23:

```
Equation 4-23. Cross entropy gradient vector for class k
```

```
∇
θ k
```

```
J Θ =m^1 ∑
i= 1
```

```
m
pki−yki x i
```

Now you can compute the gradient vector for every class, then use Gradient Descent
(or any other optimization algorithm) to find the parameter matrix **Θ** that minimizes
the cost function.

```
Logistic Regression | 151
```

Let’s use Softmax Regression to classify the iris flowers into all three classes. Scikit-
Learn’s LogisticRegression uses one-versus-all by default when you train it on more
than two classes, but you can set the multi_class hyperparameter to "multinomial"
to switch it to Softmax Regression instead. You must also specify a solver that sup‐
ports Softmax Regression, such as the "lbfgs" solver (see Scikit-Learn’s documenta‐
tion for more details). It also applies ℓ 2 regularization by default, which you can
control using the hyperparameter C.

```
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]
```

```
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)
```

So the next time you find an iris with 5 cm long and 2 cm wide petals, you can ask
your model to tell you what type of iris it is, and it will answer Iris-Virginica (class 2)
with 94.2% probability (or Iris-Versicolor with 5.8% probability):

```
>>> softmax_reg.predict([[5, 2]])
array([2])
>>> softmax_reg.predict_proba([[5, 2]])
array([[6.38014896e-07, 5.74929995e-02, 9.42506362e-01]])
```

Figure 4-25 shows the resulting decision boundaries, represented by the background
colors. Notice that the decision boundaries between any two classes are linear. The
figure also shows the probabilities for the Iris-Versicolor class, represented by the
curved lines (e.g., the line labeled with 0.450 represents the 45% probability bound‐
ary). Notice that the model can predict a class that has an estimated probability below
50%. For example, at the point where all decision boundaries meet, all classes have an
equal estimated probability of 33%.

Figure 4-25. Softmax Regression decision boundaries

**152 | Chapter 4: Training Models**

### Exercises

1. What Linear Regression training algorithm can you use if you have a training set
   with millions of features?
2. Suppose the features in your training set have very different scales. What algo‐
   rithms might suffer from this, and how? What can you do about it?
3. Can Gradient Descent get stuck in a local minimum when training a Logistic
   Regression model?
4. Do all Gradient Descent algorithms lead to the same model provided you let
   them run long enough?
5. Suppose you use Batch Gradient Descent and you plot the validation error at
   every epoch. If you notice that the validation error consistently goes up, what is
   likely going on? How can you fix this?
6. Is it a good idea to stop Mini-batch Gradient Descent immediately when the vali‐
   dation error goes up?
7. Which Gradient Descent algorithm (among those we discussed) will reach the
   vicinity of the optimal solution the fastest? Which will actually converge? How
   can you make the others converge as well?
8. Suppose you are using Polynomial Regression. You plot the learning curves and
   you notice that there is a large gap between the training error and the validation
   error. What is happening? What are three ways to solve this?
9. Suppose you are using Ridge Regression and you notice that the training error
   and the validation error are almost equal and fairly high. Would you say that the
   model suffers from high bias or high variance? Should you increase the regulari‐
   zation hyperparameter α or reduce it?
10. Why would you want to use:

- Ridge Regression instead of plain Linear Regression (i.e., without any regulari‐
  zation)?
- Lasso instead of Ridge Regression?
- Elastic Net instead of Lasso?

11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime.
    Should you implement two Logistic Regression classifiers or one Softmax Regres‐
    sion classifier?
12. Implement Batch Gradient Descent with early stopping for Softmax Regression
    (without using Scikit-Learn).

Solutions to these exercises are available in ???.

```
Exercises | 153
```

**CHAPTER 5**

**Support Vector Machines**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 5 in the final
release of the book.
```

A Support Vector Machine (SVM) is a very powerful and versatile Machine Learning
model, capable of performing linear or nonlinear classification, regression, and even
outlier detection. It is one of the most popular models in Machine Learning, and any‐
one interested in Machine Learning should have it in their toolbox. SVMs are partic‐
ularly well suited for classification of complex but small- or medium-sized datasets.

This chapter will explain the core concepts of SVMs, how to use them, and how they
work.

### Linear SVM Classification

The fundamental idea behind SVMs is best explained with some pictures. Figure 5-1
shows part of the iris dataset that was introduced at the end of Chapter 4. The two
classes can clearly be separated easily with a straight line (they are linearly separable).
The left plot shows the decision boundaries of three possible linear classifiers. The
model whose decision boundary is represented by the dashed line is so bad that it
does not even separate the classes properly. The other two models work perfectly on
this training set, but their decision boundaries come so close to the instances that
these models will probably not perform as well on new instances. In contrast, the
solid line in the plot on the right represents the decision boundary of an SVM classi‐
fier; this line not only separates the two classes but also stays as far away from the
closest training instances as possible. You can think of an SVM classifier as fitting the

```
155
```

widest possible street (represented by the parallel dashed lines) between the classes.
This is called large margin classification.

Figure 5-1. Large margin classification

Notice that adding more training instances “off the street” will not affect the decision
boundary at all: it is fully determined (or “supported”) by the instances located on the
edge of the street. These instances are called the support vectors (they are circled in
Figure 5-1).

```
SVMs are sensitive to the feature scales, as you can see in
Figure 5-2: on the left plot, the vertical scale is much larger than the
horizontal scale, so the widest possible street is close to horizontal.
After feature scaling (e.g., using Scikit-Learn’s StandardScaler),
the decision boundary looks much better (on the right plot).
```

Figure 5-2. Sensitivity to feature scales

#### Soft Margin Classification

If we strictly impose that all instances be off the street and on the right side, this is
called hard margin classification. There are two main issues with hard margin classifi‐
cation. First, it only works if the data is linearly separable, and second it is quite sensi‐
tive to outliers. Figure 5-3 shows the iris dataset with just one additional outlier: on
the left, it is impossible to find a hard margin, and on the right the decision boundary
ends up very different from the one we saw in Figure 5-1 without the outlier, and it
will probably not generalize as well.

**156 | Chapter 5: Support Vector Machines**

Figure 5-3. Hard margin sensitivity to outliers

To avoid these issues it is preferable to use a more flexible model. The objective is to
find a good balance between keeping the street as large as possible and limiting the
margin violations (i.e., instances that end up in the middle of the street or even on the
wrong side). This is called soft margin classification.

In Scikit-Learn’s SVM classes, you can control this balance using the C hyperparame‐
ter: a smaller C value leads to a wider street but more margin violations. Figure 5-4
shows the decision boundaries and margins of two soft margin SVM classifiers on a
nonlinearly separable dataset. On the left, using a low C value the margin is quite
large, but many instances end up on the street. On the right, using a high C value the
classifier makes fewer margin violations but ends up with a smaller margin. However,
it seems likely that the first classifier will generalize better: in fact even on this train‐
ing set it makes fewer prediction errors, since most of the margin violations are
actually on the correct side of the decision boundary.

Figure 5-4. Large margin (left) versus fewer margin violations (right)

```
If your SVM model is overfitting, you can try regularizing it by
reducing C.
```

The following Scikit-Learn code loads the iris dataset, scales the features, and then
trains a linear SVM model (using the LinearSVC class with C = 1 and the hinge loss
function, described shortly) to detect Iris-Virginica flowers. The resulting model is
represented on the left of Figure 5-4.

```
Linear SVM Classification | 157
```

```
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
```

```
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
```

```
svm_clf = Pipeline([
("scaler", StandardScaler()),
("linear_svc", LinearSVC(C=1, loss="hinge")),
])
```

```
svm_clf.fit(X, y)
```

Then, as usual, you can use the model to make predictions:

```
>>> svm_clf.predict([[5.5, 1.7]])
array([1.])
```

```
Unlike Logistic Regression classifiers, SVM classifiers do not out‐
put probabilities for each class.
```

Alternatively, you could use the SVC class, using SVC(kernel="linear", C=1), but it
is much slower, especially with large training sets, so it is not recommended. Another
option is to use the SGDClassifier class, with SGDClassifier(loss="hinge",
alpha=1/(m\*C)). This applies regular Stochastic Gradient Descent (see Chapter 4) to
train a linear SVM classifier. It does not converge as fast as the LinearSVC class, but it
can be useful to handle huge datasets that do not fit in memory (out-of-core train‐
ing), or to handle online classification tasks.

```
The LinearSVC class regularizes the bias term, so you should center
the training set first by subtracting its mean. This is automatic if
you scale the data using the StandardScaler. Moreover, make sure
you set the loss hyperparameter to "hinge", as it is not the default
value. Finally, for better performance you should set the dual
hyperparameter to False, unless there are more features than
training instances (we will discuss duality later in the chapter).
```

**158 | Chapter 5: Support Vector Machines**

### Nonlinear SVM Classification

Although linear SVM classifiers are efficient and work surprisingly well in many
cases, many datasets are not even close to being linearly separable. One approach to
handling nonlinear datasets is to add more features, such as polynomial features (as
you did in Chapter 4); in some cases this can result in a linearly separable dataset.
Consider the left plot in Figure 5-5: it represents a simple dataset with just one feature
x 1. This dataset is not linearly separable, as you can see. But if you add a second fea‐
ture x 2 = (x 1 )^2 , the resulting 2D dataset is perfectly linearly separable.

Figure 5-5. Adding features to make a dataset linearly separable

To implement this idea using Scikit-Learn, you can create a Pipeline containing a
PolynomialFeatures transformer (discussed in “Polynomial Regression” on page
130 ), followed by a StandardScaler and a LinearSVC. Let’s test this on the moons
dataset: this is a toy dataset for binary classification in which the data points are sha‐
ped as two interleaving half circles (see Figure 5-6). You can generate this dataset
using the make_moons() function:

```
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
```

```
polynomial_svm_clf = Pipeline([
("poly_features", PolynomialFeatures(degree=3)),
("scaler", StandardScaler()),
("svm_clf", LinearSVC(C=10, loss="hinge"))
])
```

```
polynomial_svm_clf.fit(X, y)
```

```
Nonlinear SVM Classification | 159
```

Figure 5-6. Linear SVM classifier using polynomial features

#### Polynomial Kernel

Adding polynomial features is simple to implement and can work great with all sorts
of Machine Learning algorithms (not just SVMs), but at a low polynomial degree it
cannot deal with very complex datasets, and with a high polynomial degree it creates
a huge number of features, making the model too slow.

Fortunately, when using SVMs you can apply an almost miraculous mathematical
technique called the kernel trick (it is explained in a moment). It makes it possible to
get the same result as if you added many polynomial features, even with very high-
degree polynomials, without actually having to add them. So there is no combinato‐
rial explosion of the number of features since you don’t actually add any features. This
trick is implemented by the SVC class. Let’s test it on the moons dataset:

```
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X, y)
```

This code trains an SVM classifier using a 3rd-degree polynomial kernel. It is repre‐
sented on the left of Figure 5-7. On the right is another SVM classifier using a 10th-
degree polynomial kernel. Obviously, if your model is overfitting, you might want to

**160 | Chapter 5: Support Vector Machines**

reduce the polynomial degree. Conversely, if it is underfitting, you can try increasing
it. The hyperparameter coef0 controls how much the model is influenced by high-
degree polynomials versus low-degree polynomials.

Figure 5-7. SVM classifiers with a polynomial kernel

```
A common approach to find the right hyperparameter values is to
use grid search (see Chapter 2). It is often faster to first do a very
coarse grid search, then a finer grid search around the best values
found. Having a good sense of what each hyperparameter actually
does can also help you search in the right part of the hyperparame‐
ter space.
```

#### Adding Similarity Features

Another technique to tackle nonlinear problems is to add features computed using a
similarity function that measures how much each instance resembles a particular
landmark. For example, let’s take the one-dimensional dataset discussed earlier and
add two landmarks to it at x 1 = –2 and x 1 = 1 (see the left plot in Figure 5-8). Next,
let’s define the similarity function to be the Gaussian Radial Basis Function (RBF)
with γ = 0.3 (see Equation 5-1).

```
Equation 5-1. Gaussian RBF
```

```
φγ x , ℓ = exp −γ∥ x − ℓ∥^2
```

It is a bell-shaped function varying from 0 (very far away from the landmark) to 1 (at
the landmark). Now we are ready to compute the new features. For example, let’s look
at the instance x 1 = –1: it is located at a distance of 1 from the first landmark, and 2
from the second landmark. Therefore its new features are x 2 = exp (–0.3 × 1^2 ) ≈ 0.74
and x 3 = exp (–0.3 × 2^2 ) ≈ 0.30. The plot on the right of Figure 5-8 shows the trans‐
formed dataset (dropping the original features). As you can see, it is now linearly
separable.

```
Nonlinear SVM Classification | 161
```

Figure 5-8. Similarity features using the Gaussian RBF

You may wonder how to select the landmarks. The simplest approach is to create a
landmark at the location of each and every instance in the dataset. This creates many
dimensions and thus increases the chances that the transformed training set will be
linearly separable. The downside is that a training set with m instances and n features
gets transformed into a training set with m instances and m features (assuming you
drop the original features). If your training set is very large, you end up with an
equally large number of features.

#### Gaussian RBF Kernel

Just like the polynomial features method, the similarity features method can be useful
with any Machine Learning algorithm, but it may be computationally expensive to
compute all the additional features, especially on large training sets. However, once
again the kernel trick does its SVM magic: it makes it possible to obtain a similar
result as if you had added many similarity features, without actually having to add
them. Let’s try the Gaussian RBF kernel using the SVC class:

```
rbf_kernel_svm_clf = Pipeline([
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)
```

This model is represented on the bottom left of Figure 5-9. The other plots show
models trained with different values of hyperparameters gamma (γ) and C. Increasing
gamma makes the bell-shape curve narrower (see the left plot of Figure 5-8), and as a
result each instance’s range of influence is smaller: the decision boundary ends up
being more irregular, wiggling around individual instances. Conversely, a small gamma
value makes the bell-shaped curve wider, so instances have a larger range of influ‐
ence, and the decision boundary ends up smoother. So γ acts like a regularization
hyperparameter: if your model is overfitting, you should reduce it, and if it is under‐
fitting, you should increase it (similar to the C hyperparameter).

**162 | Chapter 5: Support Vector Machines**

```
1 “A Dual Coordinate Descent Method for Large-scale Linear SVM,” Lin et al. (2008).
```

Figure 5-9. SVM classifiers using an RBF kernel

Other kernels exist but are used much more rarely. For example, some kernels are
specialized for specific data structures. String kernels are sometimes used when classi‐
fying text documents or DNA sequences (e.g., using the string subsequence kernel or
kernels based on the Levenshtein distance).

```
With so many kernels to choose from, how can you decide which
one to use? As a rule of thumb, you should always try the linear
kernel first (remember that LinearSVC is much faster than SVC(ker
nel="linear")), especially if the training set is very large or if it
has plenty of features. If the training set is not too large, you should
try the Gaussian RBF kernel as well; it works well in most cases.
Then if you have spare time and computing power, you can also
experiment with a few other kernels using cross-validation and grid
search, especially if there are kernels specialized for your training
set’s data structure.
```

#### Computational Complexity

The LinearSVC class is based on the liblinear library, which implements an optimized
algorithm for linear SVMs.^1 It does not support the kernel trick, but it scales almost

```
Nonlinear SVM Classification | 163
```

```
2 “Sequential Minimal Optimization (SMO),” J. Platt (1998).
```

linearly with the number of training instances and the number of features: its training
time complexity is roughly O(m × n).

The algorithm takes longer if you require a very high precision. This is controlled by
the tolerance hyperparameter ε (called tol in Scikit-Learn). In most classification
tasks, the default tolerance is fine.

The SVC class is based on the libsvm library, which implements an algorithm that sup‐
ports the kernel trick.^2 The training time complexity is usually between O(m^2 × n)
and O(m^3 × n). Unfortunately, this means that it gets dreadfully slow when the num‐
ber of training instances gets large (e.g., hundreds of thousands of instances). This
algorithm is perfect for complex but small or medium training sets. However, it scales
well with the number of features, especially with sparse features (i.e., when each
instance has few nonzero features). In this case, the algorithm scales roughly with the
average number of nonzero features per instance. Table 5-1 compares Scikit-Learn’s
SVM classification classes.

Table 5-1. Comparison of Scikit-Learn classes for SVM classification

```
Class Time complexity Out-of-core support Scaling required Kernel trick
LinearSVC O(m × n) No Yes No
SGDClassifier O(m × n) Yes Yes No
SVC O(m² × n) to O(m³ × n) No Yes Yes
```

### SVM Regression

As we mentioned earlier, the SVM algorithm is quite versatile: not only does it sup‐
port linear and nonlinear classification, but it also supports linear and nonlinear
regression. The trick is to reverse the objective: instead of trying to fit the largest pos‐
sible street between two classes while limiting margin violations, SVM Regression
tries to fit as many instances as possible on the street while limiting margin violations
(i.e., instances off the street). The width of the street is controlled by a hyperparame‐
ter ε. Figure 5-10 shows two linear SVM Regression models trained on some random
linear data, one with a large margin (ε = 1.5) and the other with a small margin (ε =
0.5).

**164 | Chapter 5: Support Vector Machines**

Figure 5-10. SVM Regression

Adding more training instances within the margin does not affect the model’s predic‐
tions; thus, the model is said to be ε-insensitive.

You can use Scikit-Learn’s LinearSVR class to perform linear SVM Regression. The
following code produces the model represented on the left of Figure 5-10 (the train‐
ing data should be scaled and centered first):

```
from sklearn.svm import LinearSVR
```

```
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
```

To tackle nonlinear regression tasks, you can use a kernelized SVM model. For exam‐
ple, Figure 5-11 shows SVM Regression on a random quadratic training set, using a
2 nd-degree polynomial kernel. There is little regularization on the left plot (i.e., a large
C value), and much more regularization on the right plot (i.e., a small C value).

Figure 5-11. SVM regression using a 2nd-degree polynomial kernel

```
SVM Regression | 165
```

The following code produces the model represented on the left of Figure 5-11 using
Scikit-Learn’s SVR class (which supports the kernel trick). The SVR class is the regres‐
sion equivalent of the SVC class, and the LinearSVR class is the regression equivalent
of the LinearSVC class. The LinearSVR class scales linearly with the size of the train‐
ing set (just like the LinearSVC class), while the SVR class gets much too slow when
the training set grows large (just like the SVC class).

```
from sklearn.svm import SVR
```

```
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
```

```
SVMs can also be used for outlier detection; see Scikit-Learn’s doc‐
umentation for more details.
```

### Under the Hood

This section explains how SVMs make predictions and how their training algorithms
work, starting with linear SVM classifiers. You can safely skip it and go straight to the
exercises at the end of this chapter if you are just getting started with Machine Learn‐
ing, and come back later when you want to get a deeper understanding of SVMs.

First, a word about notations: in Chapter 4 we used the convention of putting all the
model parameters in one vector **θ** , including the bias term θ 0 and the input feature
weights θ 1 to θn, and adding a bias input x 0 = 1 to all instances. In this chapter, we will
use a different convention, which is more convenient (and more common) when you
are dealing with SVMs: the bias term will be called b and the feature weights vector
will be called **w**. No bias feature will be added to the input feature vectors.

#### Decision Function and Predictions

The linear SVM classifier model predicts the class of a new instance **x** by simply com‐
puting the decision function **w** T **x** + b = w 1 x 1 + ⋯ + wn xn + b: if the result is positive,
the predicted class ŷ is the positive class (1), or else it is the negative class (0); see
Equation 5-2.

```
Equation 5-2. Linear SVM classifier prediction
```

```
y=
```

```
0 if w T x +b< 0,
1 if w T x +b≥ 0
```

**166 | Chapter 5: Support Vector Machines**

```
3 More generally, when there are n features, the decision function is an n-dimensional hyperplane, and the deci‐
sion boundary is an (n – 1)-dimensional hyperplane.
```

Figure 5-12 shows the decision function that corresponds to the model on the left of
Figure 5-4: it is a two-dimensional plane since this dataset has two features (petal
width and petal length). The decision boundary is the set of points where the decision
function is equal to 0: it is the intersection of two planes, which is a straight line (rep‐
resented by the thick solid line).^3

Figure 5-12. Decision function for the iris dataset

The dashed lines represent the points where the decision function is equal to 1 or –1:
they are parallel and at equal distance to the decision boundary, forming a margin
around it. Training a linear SVM classifier means finding the value of **w** and b that
make this margin as wide as possible while avoiding margin violations (hard margin)
or limiting them (soft margin).

#### Training Objective

Consider the slope of the decision function: it is equal to the norm of the weight vec‐
tor, ∥ **w** ∥. If we divide this slope by 2, the points where the decision function is equal
to ±1 are going to be twice as far away from the decision boundary. In other words,
dividing the slope by 2 will multiply the margin by 2. Perhaps this is easier to visual‐
ize in 2D in Figure 5-13. The smaller the weight vector **w** , the larger the margin.

```
Under the Hood | 167
```

```
4 Zeta (ζ) is the 6th letter of the Greek alphabet.
```

Figure 5-13. A smaller weight vector results in a larger margin

So we want to minimize ∥ **w** ∥ to get a large margin. However, if we also want to avoid
any margin violation (hard margin), then we need the decision function to be greater
than 1 for all positive training instances, and lower than –1 for negative training
instances. If we define t(i) = –1 for negative instances (if y(i) = 0) and t(i) = 1 for positive
instances (if y(i) = 1), then we can express this constraint as t(i)( **w** T **x** (i) + b) ≥ 1 for all
instances.

We can therefore express the hard margin linear SVM classifier objective as the con‐
strained optimization problem in Equation 5-3.

```
Equation 5-3. Hard margin linear SVM classifier objective
```

```
minimize w ,b^12 w T w
```

```
subject to ti w T x i+b ≥ 1 fori= 1, 2,⋯,m
```

```
We are minimizing^12 w T w , which is equal to^12 ∥ w ∥^2 , rather than
minimizing ∥ w ∥. Indeed,^12 ∥ w ∥^2 has a nice and simple derivative
(it is just w ) while ∥ w ∥ is not differentiable at w = 0. Optimization
algorithms work much better on differentiable functions.
```

To get the soft margin objective, we need to introduce a slack variable ζ(i) ≥ 0 for each
instance:^4 ζ(i) measures how much the ith instance is allowed to violate the margin. We
now have two conflicting objectives: making the slack variables as small as possible to

reduce the margin violations, and making^12 **w** T **w** as small as possible to increase the

margin. This is where the C hyperparameter comes in: it allows us to define the trade‐

**168 | Chapter 5: Support Vector Machines**

```
5 To learn more about Quadratic Programming, you can start by reading Stephen Boyd and Lieven Vanden‐
berghe, Convex Optimization (Cambridge, UK: Cambridge University Press, 2004) or watch Richard Brown’s
series of video lectures.
```

off between these two objectives. This gives us the constrained optimization problem
in Equation 5-4.

```
Equation 5-4. Soft margin linear SVM classifier objective
```

```
minimize w ,b,ζ^12 w T w +C∑
i= 1
```

```
m
ζi
```

```
subject to ti w T x i+b ≥ 1 −ζi and ζi ≥ 0 fori= 1, 2,⋯,m
```

#### Quadratic Programming

The hard margin and soft margin problems are both convex quadratic optimization
problems with linear constraints. Such problems are known as Quadratic Program‐
ming (QP) problems. Many off-the-shelf solvers are available to solve QP problems
using a variety of techniques that are outside the scope of this book.^5 The general
problem formulation is given by Equation 5-5.

```
Equation 5-5. Quadratic Programming problem
```

```
Minimize
p
```

```
1
2 p
```

```
T Hp + f T p
```

```
subject to Ap ≤ b
```

```
where
```

```
p is annp‐dimensional vector (np= number of parameters),
H is annp×npmatrix,
f is annp‐dimensional vector,
A is annc×npmatrix (nc= number of constraints),
b is annc‐dimensional vector.
```

Note that the expression **A p** ≤ **b** actually defines nc constraints: **p** T **a** (i) ≤ b(i) for i = 1,
2, ⋯, nc, where **a** (i) is the vector containing the elements of the ith row of **A** and b(i) is
the ith element of **b**.

You can easily verify that if you set the QP parameters in the following way, you get
the hard margin linear SVM classifier objective:

- np = n + 1, where n is the number of features (the +1 is for the bias term).

```
Under the Hood | 169
```

```
6 The objective function is convex, and the inequality constraints are continuously differentiable and convex
functions.
```

- nc = m, where m is the number of training instances.
- **H** is the np × np identity matrix, except with a zero in the top-left cell (to ignore
  the bias term).
- **f** = **0** , an np-dimensional vector full of 0s.
- **b** = **–1** , an nc-dimensional vector full of –1s.
- **a** (i) = –t(i) **x** ̇ (i), where **x** ̇ (i) is equal to **x** (i) with an extra bias feature **x** ̇ 0 = 1.

So one way to train a hard margin linear SVM classifier is just to use an off-the-shelf
QP solver by passing it the preceding parameters. The resulting vector **p** will contain
the bias term b = p 0 and the feature weights wi = pi for i = 1, 2, ⋯, n. Similarly, you
can use a QP solver to solve the soft margin problem (see the exercises at the end of
the chapter).

However, to use the kernel trick we are going to look at a different constrained opti‐
mization problem.

#### The Dual Problem

Given a constrained optimization problem, known as the primal problem, it is possi‐
ble to express a different but closely related problem, called its dual problem. The sol‐
ution to the dual problem typically gives a lower bound to the solution of the primal
problem, but under some conditions it can even have the same solutions as the pri‐
mal problem. Luckily, the SVM problem happens to meet these conditions,^6 so you
can choose to solve the primal problem or the dual problem; both will have the same
solution. Equation 5-6 shows the dual form of the linear SVM objective (if you are
interested in knowing how to derive the dual problem from the primal problem,
see ???).

```
Equation 5-6. Dual form of the linear SVM objective
```

```
minimizeα^12 ∑
i= 1
```

```
m
∑
j= 1
```

```
m
αiαjtitj x i
```

```
T
x j − ∑
i= 1
```

```
m
αi
```

```
subject to αi ≥ 0 fori= 1, 2,⋯,m
```

**170 | Chapter 5: Support Vector Machines**

```
7 As explained in Chapter 4, the dot product of two vectors a and b is normally noted a · b. However, in
Machine Learning, vectors are frequently represented as column vectors (i.e., single-column matrices), so the
dot product is achieved by computing a T b. To remain consistent with the rest of the book, we will use this
notation here, ignoring the fact that this technically results in a single-cell matrix rather than a scalar value.
```

Once you find the vector α that minimizes this equation (using a QP solver), you can

compute **w** and b that minimize the primal problem by using Equation 5-7.

```
Equation 5-7. From the dual solution to the primal solution
```

```
w = ∑
i= 1
```

```
m
αiti x i
```

```
b=n^1
s
```

```
∑
i= 1
αi > 0
```

```
m
ti− w T x i
```

The dual problem is faster to solve than the primal when the number of training
instances is smaller than the number of features. More importantly, it makes the ker‐
nel trick possible, while the primal does not. So what is this kernel trick anyway?

#### Kernelized SVM

Suppose you want to apply a 2nd-degree polynomial transformation to a two-
dimensional training set (such as the moons training set), then train a linear SVM
classifier on the transformed training set. Equation 5-8 shows the 2nd-degree polyno‐
mial mapping function φ that you want to apply.

```
Equation 5-8. Second-degree polynomial mapping
```

```
φ x =φ
```

```
x 1
x 2
```

```
=
```

```
x 12
2 x 1 x 2
```

```
x 22
```

Notice that the transformed vector is three-dimensional instead of two-dimensional.
Now let’s look at what happens to a couple of two-dimensional vectors, **a** and **b** , if we
apply this 2nd-degree polynomial mapping and then compute the dot product^7 of the
transformed vectors (See Equation 5-9).

```
Under the Hood | 171
```

```
Equation 5-9. Kernel trick for a 2nd-degree polynomial mapping
```

```
φ a Tφ b =
```

```
a 12
2 a 1 a 2
```

```
a 22
```

```
T
b 12
2 b 1 b 2
```

```
b 22
```

```
=a 12 b 12 + 2a 1 b 1 a 2 b 2 +a 22 b 22
```

```
= a 1 b 1 +a 2 b 22 =
```

```
a 1
a 2
```

```
Tb
1
b 2
```

```
2
= a T b^2
```

How about that? The dot product of the transformed vectors is equal to the square of
the dot product of the original vectors: φ( **a** )T φ( **b** ) = ( **a** T **b** )^2.

Now here is the key insight: if you apply the transformation φ to all training instan‐
ces, then the dual problem (see Equation 5-6) will contain the dot product φ( **x** (i))T
φ( **x** (j)). But if φ is the 2nd-degree polynomial transformation defined in Equation 5-8,

then you can replace this dot product of transformed vectors simply by **x** i

```
T
x j
```

```
2
```

. So

you don’t actually need to transform the training instances at all: just replace the dot
product by its square in Equation 5-6. The result will be strictly the same as if you
went through the trouble of actually transforming the training set then fitting a linear
SVM algorithm, but this trick makes the whole process much more computationally
efficient. This is the essence of the kernel trick.

The function K( **a** , **b** ) = ( **a** T **b** )^2 is called a 2nd-degree polynomial kernel. In Machine
Learning, a kernel is a function capable of computing the dot product φ( **a** )T φ( **b** )
based only on the original vectors **a** and **b** , without having to compute (or even to
know about) the transformation φ. Equation 5-10 lists some of the most commonly
used kernels.

```
Equation 5-10. Common kernels
```

```
Linear: K a , b = a T b
Polynomial: K a , b = γ a T b +rd
```

```
Gaussian RBF: K a , b = exp −γ∥ a − b ∥^2
Sigmoid: K a , b = tanh γ a T b +r
```

**172 | Chapter 5: Support Vector Machines**

```
Mercer’s Theorem
According to Mercer’s theorem, if a function K( a , b ) respects a few mathematical con‐
ditions called Mercer’s conditions (K must be continuous, symmetric in its arguments
so K( a , b ) = K( b , a ), etc.), then there exists a function φ that maps a and b into
another space (possibly with much higher dimensions) such that K( a , b ) = φ( a )T φ( b ).
So you can use K as a kernel since you know φ exists, even if you don’t know what φ
is. In the case of the Gaussian RBF kernel, it can be shown that φ actually maps each
training instance to an infinite-dimensional space, so it’s a good thing you don’t need
to actually perform the mapping!
Note that some frequently used kernels (such as the Sigmoid kernel) don’t respect all
of Mercer’s conditions, yet they generally work well in practice.
```

There is still one loose end we must tie. Equation 5-7 shows how to go from the dual
solution to the primal solution in the case of a linear SVM classifier, but if you apply
the kernel trick you end up with equations that include φ(x(i)). In fact, **w** must have
the same number of dimensions as φ(x(i)), which may be huge or even infinite, so you
can’t compute it. But how can you make predictions without knowing **w**? Well, the
good news is that you can plug in the formula for **w** from Equation 5-7 into the deci‐
sion function for a new instance **x** (n), and you get an equation with only dot products
between input vectors. This makes it possible to use the kernel trick, once again
(Equation 5-11).

```
Equation 5-11. Making predictions with a kernelized SVM
```

```
h w ,bφ x n = w Tφ x n +b= ∑
i= 1
```

```
m
αitiφ x i
```

```
T
φ x n +b
```

```
= ∑
i= 1
```

```
m
αiti φ x i
```

```
T
φ x n +b
```

```
= ∑
i= 1
αi> 0
```

```
m
αitiK x i, x n +b
```

Note that since α(i) ≠ 0 only for support vectors, making predictions involves comput‐
ing the dot product of the new input vector **x** (n) with only the support vectors, not all

the training instances. Of course, you also need to compute the bias term b, using the
same trick (Equation 5-12).

```
Under the Hood | 173
```

```
Equation 5-12. Computing the bias term using the kernel trick
```

```
b=n^1
s
```

```
∑
i= 1
αi> 0
```

```
m
ti− w Tφ x i =n^1
s
```

```
∑
i= 1
αi> 0
```

```
m
ti− ∑
j= 1
```

```
m
αjtjφ x j
```

```
T
φ x i
```

```
=n^1
s
```

```
∑
i= 1
αi> 0
```

```
m
ti− ∑
j= 1
αj> 0
```

```
m
αjtjK x i, x j
```

If you are starting to get a headache, it’s perfectly normal: it’s an unfortunate side
effect of the kernel trick.

#### Online SVMs

Before concluding this chapter, let’s take a quick look at online SVM classifiers (recall
that online learning means learning incrementally, typically as new instances arrive).

For linear SVM classifiers, one method is to use Gradient Descent (e.g., using
SGDClassifier) to minimize the cost function in Equation 5-13, which is derived
from the primal problem. Unfortunately it converges much more slowly than the
methods based on QP.

```
Equation 5-13. Linear SVM classifier cost function
```

```
J w ,b =^12 w T w + C∑
i= 1
```

```
m
max0, 1 −ti w T x i+b
```

The first sum in the cost function will push the model to have a small weight vector
**w** , leading to a larger margin. The second sum computes the total of all margin viola‐
tions. An instance’s margin violation is equal to 0 if it is located off the street and on
the correct side, or else it is proportional to the distance to the correct side of the
street. Minimizing this term ensures that the model makes the margin violations as
small and as few as possible

```
Hinge Loss
The function max(0, 1 – t) is called the hinge loss function (represented below). It is
equal to 0 when t ≥ 1. Its derivative (slope) is equal to –1 if t < 1 and 0 if t > 1. It is not
differentiable at t = 1, but just like for Lasso Regression (see “Lasso Regression” on
page 139 ) you can still use Gradient Descent using any subderivative at t = 1 (i.e., any
value between –1 and 0).
```

**174 | Chapter 5: Support Vector Machines**

```
8 “Incremental and Decremental Support Vector Machine Learning,” G. Cauwenberghs, T. Poggio (2001).
9 “Fast Kernel Classifiers with Online and Active Learning,“ A. Bordes, S. Ertekin, J. Weston, L. Bottou (2005).
```

It is also possible to implement online kernelized SVMs—for example, using “Incre‐
mental and Decremental SVM Learning”^8 or “Fast Kernel Classifiers with Online and
Active Learning.”^9 However, these are implemented in Matlab and C++. For large-
scale nonlinear problems, you may want to consider using neural networks instead
(see Part II).

### Exercises

1. What is the fundamental idea behind Support Vector Machines?
2. What is a support vector?
3. Why is it important to scale the inputs when using SVMs?
4. Can an SVM classifier output a confidence score when it classifies an instance?
   What about a probability?
5. Should you use the primal or the dual form of the SVM problem to train a model
   on a training set with millions of instances and hundreds of features?
6. Say you trained an SVM classifier with an RBF kernel. It seems to underfit the
   training set: should you increase or decrease γ (gamma)? What about C?
7. How should you set the QP parameters ( **H** , **f** , **A** , and **b** ) to solve the soft margin
   linear SVM classifier problem using an off-the-shelf QP solver?
8. Train a LinearSVC on a linearly separable dataset. Then train an SVC and a
   SGDClassifier on the same dataset. See if you can get them to produce roughly
   the same model.
9. Train an SVM classifier on the MNIST dataset. Since SVM classifiers are binary
   classifiers, you will need to use one-versus-all to classify all 10 digits. You may

```
Exercises | 175
```

```
want to tune the hyperparameters using small validation sets to speed up the pro‐
cess. What accuracy can you reach?
```

10. Train an SVM regressor on the California housing dataset.

Solutions to these exercises are available in ???.

**176 | Chapter 5: Support Vector Machines**

**CHAPTER 6**

**Decision Trees**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 6 in the final
release of the book.
```

Like SVMs, Decision Trees are versatile Machine Learning algorithms that can per‐
form both classification and regression tasks, and even multioutput tasks. They are
very powerful algorithms, capable of fitting complex datasets. For example, in Chap‐
ter 2 you trained a DecisionTreeRegressor model on the California housing dataset,
fitting it perfectly (actually overfitting it).

Decision Trees are also the fundamental components of Random Forests (see Chap‐
ter 7), which are among the most powerful Machine Learning algorithms available
today.

In this chapter we will start by discussing how to train, visualize, and make predic‐
tions with Decision Trees. Then we will go through the CART training algorithm
used by Scikit-Learn, and we will discuss how to regularize trees and use them for
regression tasks. Finally, we will discuss some of the limitations of Decision Trees.

### Training and Visualizing a Decision Tree

To understand Decision Trees, let’s just build one and take a look at how it makes pre‐
dictions. The following code trains a DecisionTreeClassifier on the iris dataset
(see Chapter 4):

```
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
```

```
177
```

```
1 Graphviz is an open source graph visualization software package, available at http://www.graphviz.org/.
```

```
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
```

```
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```

You can visualize the trained Decision Tree by first using the export_graphviz()
method to output a graph definition file called iris_tree.dot:

```
from sklearn.tree import export_graphviz
```

```
export_graphviz(
tree_clf,
out_file=image_path("iris_tree.dot"),
feature_names=iris.feature_names[2:],
class_names=iris.target_names,
rounded=True,
filled=True
)
```

Then you can convert this .dot file to a variety of formats such as PDF or PNG using
the dot command-line tool from the graphviz package.^1 This command line converts
the .dot file to a .png image file:

```
$ dot -Tpng iris_tree.dot -o iris_tree.png
```

Your first decision tree looks like Figure 6-1.

**178 | Chapter 6: Decision Trees**

Figure 6-1. Iris Decision Tree

### Making Predictions

Let’s see how the tree represented in Figure 6-1 makes predictions. Suppose you find
an iris flower and you want to classify it. You start at the root node (depth 0, at the
top): this node asks whether the flower’s petal length is smaller than 2.45 cm. If it is,
then you move down to the root’s left child node (depth 1, left). In this case, it is a leaf
node (i.e., it does not have any children nodes), so it does not ask any questions: you
can simply look at the predicted class for that node and the Decision Tree predicts
that your flower is an Iris-Setosa (class=setosa).

Now suppose you find another flower, but this time the petal length is greater than
2.45 cm. You must move down to the root’s right child node (depth 1, right), which is
not a leaf node, so it asks another question: is the petal width smaller than 1.75 cm? If
it is, then your flower is most likely an Iris-Versicolor (depth 2, left). If not, it is likely
an Iris-Virginica (depth 2, right). It’s really that simple.

```
One of the many qualities of Decision Trees is that they require
very little data preparation. In particular, they don’t require feature
scaling or centering at all.
```

```
Making Predictions | 179
```

A node’s samples attribute counts how many training instances it applies to. For
example, 100 training instances have a petal length greater than 2.45 cm (depth 1,
right), among which 54 have a petal width smaller than 1.75 cm (depth 2, left). A
node’s value attribute tells you how many training instances of each class this node
applies to: for example, the bottom-right node applies to 0 Iris-Setosa, 1 Iris-
Versicolor, and 45 Iris-Virginica. Finally, a node’s gini attribute measures its impur‐
ity: a node is “pure” (gini=0) if all training instances it applies to belong to the same
class. For example, since the depth-1 left node applies only to Iris-Setosa training
instances, it is pure and its gini score is 0. Equation 6-1 shows how the training algo‐
rithm computes the gini score Gi of the ith node. For example, the depth-2 left node
has a gini score equal to 1 – (0/54)^2 – (49/54)^2 – (5/54)^2 ≈ 0.168. Another impurity
measure is discussed shortly.

```
Equation 6-1. Gini impurity
```

```
Gi= 1 − ∑
k= 1
```

```
n
pi,k^2
```

- pi,k is the ratio of class k instances among the training instances in the ith node.

```
Scikit-Learn uses the CART algorithm, which produces only binary
trees: nonleaf nodes always have two children (i.e., questions only
have yes/no answers). However, other algorithms such as ID3 can
produce Decision Trees with nodes that have more than two chil‐
dren.
```

Figure 6-2 shows this Decision Tree’s decision boundaries. The thick vertical line rep‐
resents the decision boundary of the root node (depth 0): petal length = 2.45 cm.
Since the left area is pure (only Iris-Setosa), it cannot be split any further. However,
the right area is impure, so the depth-1 right node splits it at petal width = 1.75 cm
(represented by the dashed line). Since max_depth was set to 2, the Decision Tree
stops right there. However, if you set max_depth to 3, then the two depth-2 nodes
would each add another decision boundary (represented by the dotted lines).

**180 | Chapter 6: Decision Trees**

Figure 6-2. Decision Tree decision boundaries

```
Model Interpretation: White Box Versus Black Box
As you can see Decision Trees are fairly intuitive and their decisions are easy to inter‐
pret. Such models are often called white box models. In contrast, as we will see, Ran‐
dom Forests or neural networks are generally considered black box models. They
make great predictions, and you can easily check the calculations that they performed
to make these predictions; nevertheless, it is usually hard to explain in simple terms
why the predictions were made. For example, if a neural network says that a particu‐
lar person appears on a picture, it is hard to know what actually contributed to this
prediction: did the model recognize that person’s eyes? Her mouth? Her nose? Her
shoes? Or even the couch that she was sitting on? Conversely, Decision Trees provide
nice and simple classification rules that can even be applied manually if need be (e.g.,
for flower classification).
```

### Estimating Class Probabilities

A Decision Tree can also estimate the probability that an instance belongs to a partic‐
ular class k: first it traverses the tree to find the leaf node for this instance, and then it
returns the ratio of training instances of class k in this node. For example, suppose
you have found a flower whose petals are 5 cm long and 1.5 cm wide. The corre‐
sponding leaf node is the depth-2 left node, so the Decision Tree should output the
following probabilities: 0% for Iris-Setosa (0/54), 90.7% for Iris-Versicolor (49/54),
and 9.3% for Iris-Virginica (5/54). And of course if you ask it to predict the class, it
should output Iris-Versicolor (class 1) since it has the highest probability. Let’s check
this:

```
>>> tree_clf.predict_proba([[5, 1.5]])
array([[0. , 0.90740741, 0.09259259]])
```

```
Estimating Class Probabilities | 181
```

```
>>> tree_clf.predict([[5, 1.5]])
array([1])
```

Perfect! Notice that the estimated probabilities would be identical anywhere else in
the bottom-right rectangle of Figure 6-2—for example, if the petals were 6 cm long
and 1.5 cm wide (even though it seems obvious that it would most likely be an Iris-
Virginica in this case).

### The CART Training Algorithm

Scikit-Learn uses the Classification And Regression Tree (CART) algorithm to train
Decision Trees (also called “growing” trees). The idea is really quite simple: the algo‐
rithm first splits the training set in two subsets using a single feature k and a thres‐
hold tk (e.g., “petal length ≤ 2.45 cm”). How does it choose k and tk? It searches for the
pair (k, tk) that produces the purest subsets (weighted by their size). The cost function
that the algorithm tries to minimize is given by Equation 6-2.

```
Equation 6-2. CART cost function for classification
```

```
Jk,tk =
```

```
mleft
m Gleft+
```

```
mright
m Gright
```

```
where
```

```
Gleft/rightmeasures the impurity of the left/right subset,
mleft/rightis the number of instances in the left/right subset.
```

Once it has successfully split the training set in two, it splits the subsets using the
same logic, then the sub-subsets and so on, recursively. It stops recursing once it rea‐
ches the maximum depth (defined by the max_depth hyperparameter), or if it cannot
find a split that will reduce impurity. A few other hyperparameters (described in a
moment) control additional stopping conditions (min_samples_split, min_sam
ples_leaf, min_weight_fraction_leaf, and max_leaf_nodes).

**182 | Chapter 6: Decision Trees**

```
2 P is the set of problems that can be solved in polynomial time. NP is the set of problems whose solutions can
be verified in polynomial time. An NP-Hard problem is a problem to which any NP problem can be reduced
in polynomial time. An NP-Complete problem is both NP and NP-Hard. A major open mathematical ques‐
tion is whether or not P = NP. If P ≠ NP (which seems likely), then no polynomial algorithm will ever be
found for any NP-Complete problem (except perhaps on a quantum computer).
3 log 2 is the binary logarithm. It is equal to log 2 (m) = log(m) / log(2).
4 A reduction of entropy is often called an information gain.
```

```
As you can see, the CART algorithm is a greedy algorithm: it greed‐
ily searches for an optimum split at the top level, then repeats the
process at each level. It does not check whether or not the split will
lead to the lowest possible impurity several levels down. A greedy
algorithm often produces a reasonably good solution, but it is not
guaranteed to be the optimal solution.
Unfortunately, finding the optimal tree is known to be an NP-
Complete problem:^2 it requires O(exp(m)) time, making the prob‐
lem intractable even for fairly small training sets. This is why we
must settle for a “reasonably good” solution.
```

### Computational Complexity

Making predictions requires traversing the Decision Tree from the root to a leaf.
Decision Trees are generally approximately balanced, so traversing the Decision Tree
requires going through roughly O(log 2 (m)) nodes.^3 Since each node only requires
checking the value of one feature, the overall prediction complexity is just O(log 2 (m)),
independent of the number of features. So predictions are very fast, even when deal‐
ing with large training sets.

However, the training algorithm compares all features (or less if max_features is set)
on all samples at each node. This results in a training complexity of O(n × m log(m)).
For small training sets (less than a few thousand instances), Scikit-Learn can speed up
training by presorting the data (set presort=True), but this slows down training con‐
siderably for larger training sets.

### Gini Impurity or Entropy?

By default, the Gini impurity measure is used, but you can select the entropy impurity
measure instead by setting the criterion hyperparameter to "entropy". The concept
of entropy originated in thermodynamics as a measure of molecular disorder:
entropy approaches zero when molecules are still and well ordered. It later spread to a
wide variety of domains, including Shannon’s information theory, where it measures
the average information content of a message:^4 entropy is zero when all messages are
identical. In Machine Learning, it is frequently used as an impurity measure: a set’s

```
Computational Complexity | 183
```

```
5 See Sebastian Raschka’s interesting analysis for more details.
```

entropy is zero when it contains instances of only one class. Equation 6-3 shows the
definition of the entropy of the ith node. For example, the depth-2 left node in

Figure 6-1 has an entropy equal to −^4954 log 2 4954 − 545 log 2 545 ≈ 0.445.

```
Equation 6-3. Entropy
```

```
Hi= − ∑
k= 1
pi,k≠ 0
```

```
n
pi,klog 2 pi,k
```

So should you use Gini impurity or entropy? The truth is, most of the time it does not
make a big difference: they lead to similar trees. Gini impurity is slightly faster to
compute, so it is a good default. However, when they differ, Gini impurity tends to
isolate the most frequent class in its own branch of the tree, while entropy tends to
produce slightly more balanced trees.^5

### Regularization Hyperparameters

Decision Trees make very few assumptions about the training data (as opposed to lin‐
ear models, which obviously assume that the data is linear, for example). If left
unconstrained, the tree structure will adapt itself to the training data, fitting it very
closely, and most likely overfitting it. Such a model is often called a nonparametric
model, not because it does not have any parameters (it often has a lot) but because the
number of parameters is not determined prior to training, so the model structure is
free to stick closely to the data. In contrast, a parametric model such as a linear model
has a predetermined number of parameters, so its degree of freedom is limited,
reducing the risk of overfitting (but increasing the risk of underfitting).

To avoid overfitting the training data, you need to restrict the Decision Tree’s freedom
during training. As you know by now, this is called regularization. The regularization
hyperparameters depend on the algorithm used, but generally you can at least restrict
the maximum depth of the Decision Tree. In Scikit-Learn, this is controlled by the
max_depth hyperparameter (the default value is None, which means unlimited).
Reducing max_depth will regularize the model and thus reduce the risk of overfitting.

The DecisionTreeClassifier class has a few other parameters that similarly restrict
the shape of the Decision Tree: min_samples_split (the minimum number of sam‐
ples a node must have before it can be split), min_samples_leaf (the minimum num‐
ber of samples a leaf node must have), min_weight_fraction_leaf (same as
min_samples_leaf but expressed as a fraction of the total number of weighted

**184 | Chapter 6: Decision Trees**

instances), max*leaf_nodes (maximum number of leaf nodes), and max_features
(maximum number of features that are evaluated for splitting at each node). Increas‐
ing min*\_ hyperparameters or reducing max\_\_ hyperparameters will regularize the
model.

```
Other algorithms work by first training the Decision Tree without
restrictions, then pruning (deleting) unnecessary nodes. A node
whose children are all leaf nodes is considered unnecessary if the
purity improvement it provides is not statistically significant. Stan‐
dard statistical tests, such as the χ^2 test, are used to estimate the
probability that the improvement is purely the result of chance
(which is called the null hypothesis). If this probability, called the p-
value, is higher than a given threshold (typically 5%, controlled by
a hyperparameter), then the node is considered unnecessary and its
children are deleted. The pruning continues until all unnecessary
nodes have been pruned.
```

Figure 6-3 shows two Decision Trees trained on the moons dataset (introduced in
Chapter 5). On the left, the Decision Tree is trained with the default hyperparameters
(i.e., no restrictions), and on the right the Decision Tree is trained with min_sam
ples_leaf=4. It is quite obvious that the model on the left is overfitting, and the
model on the right will probably generalize better.

Figure 6-3. Regularization using min_samples_leaf

### Regression

Decision Trees are also capable of performing regression tasks. Let’s build a regres‐
sion tree using Scikit-Learn’s DecisionTreeRegressor class, training it on a noisy
quadratic dataset with max_depth=2:

```
from sklearn.tree import DecisionTreeRegressor
```

```
Regression | 185
```

```
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
```

The resulting tree is represented on Figure 6-4.

Figure 6-4. A Decision Tree for regression

This tree looks very similar to the classification tree you built earlier. The main differ‐
ence is that instead of predicting a class in each node, it predicts a value. For example,
suppose you want to make a prediction for a new instance with x 1 = 0.6. You traverse
the tree starting at the root, and you eventually reach the leaf node that predicts
value=0.1106. This prediction is simply the average target value of the 110 training
instances associated to this leaf node. This prediction results in a Mean Squared Error
(MSE) equal to 0.0151 over these 110 instances.

This model’s predictions are represented on the left of Figure 6-5. If you set
max_depth=3, you get the predictions represented on the right. Notice how the pre‐
dicted value for each region is always the average target value of the instances in that
region. The algorithm splits each region in a way that makes most training instances
as close as possible to that predicted value.

**186 | Chapter 6: Decision Trees**

Figure 6-5. Predictions of two Decision Tree regression models

The CART algorithm works mostly the same way as earlier, except that instead of try‐
ing to split the training set in a way that minimizes impurity, it now tries to split the
training set in a way that minimizes the MSE. Equation 6-4 shows the cost function
that the algorithm tries to minimize.

```
Equation 6-4. CART cost function for regression
```

```
Jk,tk =
```

```
mleft
m MSEleft+
```

```
mright
m MSEright where
```

```
MSEnode= ∑
i∈node
```

```
ynode−yi
```

```
2
```

```
ynode=m^1
node
```

```
∑
i∈node
```

```
yi
```

Just like for classification tasks, Decision Trees are prone to overfitting when dealing
with regression tasks. Without any regularization (i.e., using the default hyperpara‐
meters), you get the predictions on the left of Figure 6-6. It is obviously overfitting
the training set very badly. Just setting min_samples_leaf=10 results in a much more
reasonable model, represented on the right of Figure 6-6.

Figure 6-6. Regularizing a Decision Tree regressor

```
Regression | 187
```

```
6 It randomly selects the set of features to evaluate at each node.
```

### Instability

Hopefully by now you are convinced that Decision Trees have a lot going for them:
they are simple to understand and interpret, easy to use, versatile, and powerful.
However they do have a few limitations. First, as you may have noticed, Decision
Trees love orthogonal decision boundaries (all splits are perpendicular to an axis),
which makes them sensitive to training set rotation. For example, Figure 6-7 shows a
simple linearly separable dataset: on the left, a Decision Tree can split it easily, while
on the right, after the dataset is rotated by 45°, the decision boundary looks unneces‐
sarily convoluted. Although both Decision Trees fit the training set perfectly, it is very
likely that the model on the right will not generalize well. One way to limit this prob‐
lem is to use PCA (see Chapter 8), which often results in a better orientation of the
training data.

Figure 6-7. Sensitivity to training set rotation

More generally, the main issue with Decision Trees is that they are very sensitive to
small variations in the training data. For example, if you just remove the widest Iris-
Versicolor from the iris training set (the one with petals 4.8 cm long and 1.8 cm wide)
and train a new Decision Tree, you may get the model represented in Figure 6-8. As
you can see, it looks very different from the previous Decision Tree (Figure 6-2).
Actually, since the training algorithm used by Scikit-Learn is stochastic^6 you may
get very different models even on the same training data (unless you set the
random_state hyperparameter).

**188 | Chapter 6: Decision Trees**

Figure 6-8. Sensitivity to training set details

Random Forests can limit this instability by averaging predictions over many trees, as
we will see in the next chapter.

### Exercises

1. What is the approximate depth of a Decision Tree trained (without restrictions)
   on a training set with 1 million instances?
2. Is a node’s Gini impurity generally lower or greater than its parent’s? Is it gener‐
   ally lower/greater, or always lower/greater?
3. If a Decision Tree is overfitting the training set, is it a good idea to try decreasing
   max_depth?
4. If a Decision Tree is underfitting the training set, is it a good idea to try scaling
   the input features?
5. If it takes one hour to train a Decision Tree on a training set containing 1 million
   instances, roughly how much time will it take to train another Decision Tree on a
   training set containing 10 million instances?
6. If your training set contains 100,000 instances, will setting presort=True speed
   up training?
7. Train and fine-tune a Decision Tree for the moons dataset.
   a.Generate a moons dataset using make_moons(n_samples=10000, noise=0.4).
   b. Split it into a training set and a test set using train_test_split().

```
Exercises | 189
```

```
c.Use grid search with cross-validation (with the help of the GridSearchCV
class) to find good hyperparameter values for a DecisionTreeClassifier.
Hint: try various values for max_leaf_nodes.
d.Train it on the full training set using these hyperparameters, and measure
your model’s performance on the test set. You should get roughly 85% to 87%
accuracy.
```

8. Grow a forest.
   a.Continuing the previous exercise, generate 1,000 subsets of the training set,
   each containing 100 instances selected randomly. Hint: you can use Scikit-
   Learn’s ShuffleSplit class for this.
   b. Train one Decision Tree on each subset, using the best hyperparameter values
   found above. Evaluate these 1,000 Decision Trees on the test set. Since they
   were trained on smaller sets, these Decision Trees will likely perform worse
   than the first Decision Tree, achieving only about 80% accuracy.
   c.Now comes the magic. For each test set instance, generate the predictions of
   the 1,000 Decision Trees, and keep only the most frequent prediction (you can
   use SciPy’s mode() function for this). This gives you majority-vote predictions
   over the test set.
   d.Evaluate these predictions on the test set: you should obtain a slightly higher
   accuracy than your first model (about 0.5 to 1.5% higher). Congratulations,
   you have trained a Random Forest classifier!

Solutions to these exercises are available in ???.

**190 | Chapter 6: Decision Trees**

**CHAPTER 7**

**Ensemble Learning and Random Forests**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 7 in the final
release of the book.
```

Suppose you ask a complex question to thousands of random people, then aggregate
their answers. In many cases you will find that this aggregated answer is better than
an expert’s answer. This is called the wisdom of the crowd. Similarly, if you aggregate
the predictions of a group of predictors (such as classifiers or regressors), you will
often get better predictions than with the best individual predictor. A group of pre‐
dictors is called an ensemble; thus, this technique is called Ensemble Learning, and an
Ensemble Learning algorithm is called an Ensemble method.

For example, you can train a group of Decision Tree classifiers, each on a different
random subset of the training set. To make predictions, you just obtain the predic‐
tions of all individual trees, then predict the class that gets the most votes (see the last
exercise in Chapter 6). Such an ensemble of Decision Trees is called a Random Forest,
and despite its simplicity, this is one of the most powerful Machine Learning algo‐
rithms available today.

Moreover, as we discussed in Chapter 2, you will often use Ensemble methods near
the end of a project, once you have already built a few good predictors, to combine
them into an even better predictor. In fact, the winning solutions in Machine Learn‐
ing competitions often involve several Ensemble methods (most famously in the Net‐
flix Prize competition).

In this chapter we will discuss the most popular Ensemble methods, including bag‐
ging, boosting, stacking, and a few others. We will also explore Random Forests.

```
191
```

### Voting Classifiers

Suppose you have trained a few classifiers, each one achieving about 80% accuracy.
You may have a Logistic Regression classifier, an SVM classifier, a Random Forest
classifier, a K-Nearest Neighbors classifier, and perhaps a few more (see Figure 7-1).

Figure 7-1. Training diverse classifiers

A very simple way to create an even better classifier is to aggregate the predictions of
each classifier and predict the class that gets the most votes. This majority-vote classi‐
fier is called a hard voting classifier (see Figure 7-2).

Figure 7-2. Hard voting classifier predictions

**192 | Chapter 7: Ensemble Learning and Random Forests**

Somewhat surprisingly, this voting classifier often achieves a higher accuracy than the
best classifier in the ensemble. In fact, even if each classifier is a weak learner (mean‐
ing it does only slightly better than random guessing), the ensemble can still be a
strong learner (achieving high accuracy), provided there are a sufficient number of
weak learners and they are sufficiently diverse.

How is this possible? The following analogy can help shed some light on this mystery.
Suppose you have a slightly biased coin that has a 51% chance of coming up heads,
and 49% chance of coming up tails. If you toss it 1,000 times, you will generally get
more or less 510 heads and 490 tails, and hence a majority of heads. If you do the
math, you will find that the probability of obtaining a majority of heads after 1,000
tosses is close to 75%. The more you toss the coin, the higher the probability (e.g.,
with 10,000 tosses, the probability climbs over 97%). This is due to the law of large
numbers: as you keep tossing the coin, the ratio of heads gets closer and closer to the
probability of heads (51%). Figure 7-3 shows 10 series of biased coin tosses. You can
see that as the number of tosses increases, the ratio of heads approaches 51%. Eventu‐
ally all 10 series end up so close to 51% that they are consistently above 50%.

Figure 7-3. The law of large numbers

Similarly, suppose you build an ensemble containing 1,000 classifiers that are individ‐
ually correct only 51% of the time (barely better than random guessing). If you pre‐
dict the majority voted class, you can hope for up to 75% accuracy! However, this is
only true if all classifiers are perfectly independent, making uncorrelated errors,
which is clearly not the case since they are trained on the same data. They are likely to
make the same types of errors, so there will be many majority votes for the wrong
class, reducing the ensemble’s accuracy.

```
Voting Classifiers | 193
```

```
Ensemble methods work best when the predictors are as independ‐
ent from one another as possible. One way to get diverse classifiers
is to train them using very different algorithms. This increases the
chance that they will make very different types of errors, improving
the ensemble’s accuracy.
```

The following code creates and trains a voting classifier in Scikit-Learn, composed of
three diverse classifiers (the training set is the moons dataset, introduced in Chap‐
ter 5):

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
```

```
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
```

```
voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
voting='hard')
voting_clf.fit(X_train, y_train)
```

Let’s look at each classifier’s accuracy on the test set:

```
>>> from sklearn.metrics import accuracy_score
>>> for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
... clf.fit(X_train, y_train)
... y_pred = clf.predict(X_test)
... print (clf.__class__.__name__, accuracy_score(y_test, y_pred))
...
LogisticRegression 0.864
RandomForestClassifier 0.896
SVC 0.888
VotingClassifier 0.904
```

There you have it! The voting classifier slightly outperforms all the individual classifi‐
ers.

If all classifiers are able to estimate class probabilities (i.e., they have a pre
dict_proba() method), then you can tell Scikit-Learn to predict the class with the
highest class probability, averaged over all the individual classifiers. This is called soft
voting. It often achieves higher performance than hard voting because it gives more
weight to highly confident votes. All you need to do is replace voting="hard" with
voting="soft" and ensure that all classifiers can estimate class probabilities. This is
not the case of the SVC class by default, so you need to set its probability hyperpara‐
meter to True (this will make the SVC class use cross-validation to estimate class prob‐
abilities, slowing down training, and it will add a predict_proba() method). If you

**194 | Chapter 7: Ensemble Learning and Random Forests**

```
1 “Bagging Predictors,” L. Breiman (1996).
2 In statistics, resampling with replacement is called bootstrapping.
3 “Pasting small votes for classification in large databases and on-line,” L. Breiman (1999).
```

modify the preceding code to use soft voting, you will find that the voting classifier
achieves over 91.2% accuracy!

### Bagging and Pasting

One way to get a diverse set of classifiers is to use very different training algorithms,
as just discussed. Another approach is to use the same training algorithm for every
predictor, but to train them on different random subsets of the training set. When
sampling is performed with replacement, this method is called bagging^1 (short for
bootstrap aggregating^2 ). When sampling is performed without replacement, it is called
pasting.^3

In other words, both bagging and pasting allow training instances to be sampled sev‐
eral times across multiple predictors, but only bagging allows training instances to be
sampled several times for the same predictor. This sampling and training process is
represented in Figure 7-4.

Figure 7-4. Pasting/bagging training set sampling and training

Once all predictors are trained, the ensemble can make a prediction for a new
instance by simply aggregating the predictions of all predictors. The aggregation
function is typically the statistical mode (i.e., the most frequent prediction, just like a
hard voting classifier) for classification, or the average for regression. Each individual

```
Bagging and Pasting | 195
```

```
4 Bias and variance were introduced in Chapter 4.
5 max_samples can alternatively be set to a float between 0.0 and 1.0, in which case the max number of instances
to sample is equal to the size of the training set times max_samples.
```

predictor has a higher bias than if it were trained on the original training set, but
aggregation reduces both bias and variance.^4 Generally, the net result is that the
ensemble has a similar bias but a lower variance than a single predictor trained on the
original training set.

As you can see in Figure 7-4, predictors can all be trained in parallel, via different
CPU cores or even different servers. Similarly, predictions can be made in parallel.
This is one of the reasons why bagging and pasting are such popular methods: they
scale very well.

#### Bagging and Pasting in Scikit-Learn

Scikit-Learn offers a simple API for both bagging and pasting with the BaggingClas
sifier class (or BaggingRegressor for regression). The following code trains an
ensemble of 500 Decision Tree classifiers,^5 each trained on 100 training instances ran‐
domly sampled from the training set with replacement (this is an example of bagging,
but if you want to use pasting instead, just set bootstrap=False). The n_jobs param‐
eter tells Scikit-Learn the number of CPU cores to use for training and predictions
(–1 tells Scikit-Learn to use all available cores):

```
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
```

```
bag_clf = BaggingClassifier(
DecisionTreeClassifier(), n_estimators=500,
max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

```
The BaggingClassifier automatically performs soft voting
instead of hard voting if the base classifier can estimate class proba‐
bilities (i.e., if it has a predict_proba() method), which is the case
with Decision Trees classifiers.
```

Figure 7-5 compares the decision boundary of a single Decision Tree with the deci‐
sion boundary of a bagging ensemble of 500 trees (from the preceding code), both
trained on the moons dataset. As you can see, the ensemble’s predictions will likely
generalize much better than the single Decision Tree’s predictions: the ensemble has a
comparable bias but a smaller variance (it makes roughly the same number of errors
on the training set, but the decision boundary is less irregular).

**196 | Chapter 7: Ensemble Learning and Random Forests**

```
6 As m grows, this ratio approaches 1 – exp(–1) ≈ 63.212%.
```

Figure 7-5. A single Decision Tree versus a bagging ensemble of 500 trees

Bootstrapping introduces a bit more diversity in the subsets that each predictor is
trained on, so bagging ends up with a slightly higher bias than pasting, but this also
means that predictors end up being less correlated so the ensemble’s variance is
reduced. Overall, bagging often results in better models, which explains why it is gen‐
erally preferred. However, if you have spare time and CPU power you can use cross-
validation to evaluate both bagging and pasting and select the one that works best.

#### Out-of-Bag Evaluation

With bagging, some instances may be sampled several times for any given predictor,
while others may not be sampled at all. By default a BaggingClassifier samples m
training instances with replacement (bootstrap=True), where m is the size of the
training set. This means that only about 63% of the training instances are sampled on
average for each predictor.^6 The remaining 37% of the training instances that are not
sampled are called out-of-bag (oob) instances. Note that they are not the same 37%
for all predictors.

Since a predictor never sees the oob instances during training, it can be evaluated on
these instances, without the need for a separate validation set. You can evaluate the
ensemble itself by averaging out the oob evaluations of each predictor.

In Scikit-Learn, you can set oob*score=True when creating a BaggingClassifier to
request an automatic oob evaluation after training. The following code demonstrates
this. The resulting evaluation score is available through the oob_score* variable:

```
>>> bag_clf = BaggingClassifier(
... DecisionTreeClassifier(), n_estimators=500,
... bootstrap=True, n_jobs=-1, oob_score=True)
...
>>> bag_clf.fit(X_train, y_train)
```

```
Bagging and Pasting | 197
```

```
7 “Ensembles on Random Patches,” G. Louppe and P. Geurts (2012).
8 “The random subspace method for constructing decision forests,” Tin Kam Ho (1998).
```

```
>>> bag_clf.oob_score_
0.90133333333333332
```

According to this oob evaluation, this BaggingClassifier is likely to achieve about
90.1% accuracy on the test set. Let’s verify this:

```
>>> from sklearn.metrics import accuracy_score
>>> y_pred = bag_clf.predict(X_test)
>>> accuracy_score(y_test, y_pred)
0.91200000000000003
```

We get 91.2% accuracy on the test set—close enough!

The oob decision function for each training instance is also available through the
oob*decision_function* variable. In this case (since the base estimator has a pre
dict_proba() method) the decision function returns the class probabilities for each
training instance. For example, the oob evaluation estimates that the first training
instance has a 68.25% probability of belonging to the positive class (and 31.75% of
belonging to the negative class):

```
>>> bag_clf.oob_decision_function_
array([[0.31746032, 0.68253968],
[0.34117647, 0.65882353],
[1. , 0. ],
```

```
[1. , 0. ],
[0.03108808, 0.96891192],
[0.57291667, 0.42708333]])
```

### Random Patches and Random Subspaces

The BaggingClassifier class supports sampling the features as well. This is con‐
trolled by two hyperparameters: max_features and bootstrap_features. They work
the same way as max_samples and bootstrap, but for feature sampling instead of
instance sampling. Thus, each predictor will be trained on a random subset of the
input features.

This is particularly useful when you are dealing with high-dimensional inputs (such
as images). Sampling both training instances and features is called the Random
Patches method.^7 Keeping all training instances (i.e., bootstrap=False and max_sam
ples=1.0) but sampling features (i.e., bootstrap_features=True and/or max_fea
tures smaller than 1.0) is called the Random Subspaces method.^8

**198 | Chapter 7: Ensemble Learning and Random Forests**

9 “Random Decision Forests,” T. Ho (1995).
10 The BaggingClassifier class remains useful if you want a bag of something other than Decision Trees.
11 There are a few notable exceptions: splitter is absent (forced to "random"), presort is absent (forced to
False), max_samples is absent (forced to 1.0), and base_estimator is absent (forced to DecisionTreeClassi
fier with the provided hyperparameters).

```
Sampling features results in even more predictor diversity, trading a bit more bias for
a lower variance.
```

### Random Forests

```
As we have discussed, a Random Forest^9 is an ensemble of Decision Trees, generally
trained via the bagging method (or sometimes pasting), typically with max_samples
set to the size of the training set. Instead of building a BaggingClassifier and pass‐
ing it a DecisionTreeClassifier, you can instead use the RandomForestClassifier
class, which is more convenient and optimized for Decision Trees^10 (similarly, there is
a RandomForestRegressor class for regression tasks). The following code trains a
Random Forest classifier with 500 trees (each limited to maximum 16 nodes), using
all available CPU cores:
from sklearn.ensemble import RandomForestClassifier
```

```
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
```

```
y_pred_rf = rnd_clf.predict(X_test)
With a few exceptions, a RandomForestClassifier has all the hyperparameters of a
DecisionTreeClassifier (to control how trees are grown), plus all the hyperpara‐
meters of a BaggingClassifier to control the ensemble itself.^11
The Random Forest algorithm introduces extra randomness when growing trees;
instead of searching for the very best feature when splitting a node (see Chapter 6), it
searches for the best feature among a random subset of features. This results in a
greater tree diversity, which (once again) trades a higher bias for a lower variance,
generally yielding an overall better model. The following BaggingClassifier is
roughly equivalent to the previous RandomForestClassifier:
bag_clf = BaggingClassifier(
DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)
```

```
Random Forests | 199
```

12 “Extremely randomized trees,” P. Geurts, D. Ernst, L. Wehenkel (2005).

#### Extra-Trees

```
When you are growing a tree in a Random Forest, at each node only a random subset
of the features is considered for splitting (as discussed earlier). It is possible to make
trees even more random by also using random thresholds for each feature rather than
searching for the best possible thresholds (like regular Decision Trees do).
A forest of such extremely random trees is simply called an Extremely Randomized
Trees ensemble^12 (or Extra-Trees for short). Once again, this trades more bias for a
lower variance. It also makes Extra-Trees much faster to train than regular Random
Forests since finding the best possible threshold for each feature at every node is one
of the most time-consuming tasks of growing a tree.
You can create an Extra-Trees classifier using Scikit-Learn’s ExtraTreesClassifier
class. Its API is identical to the RandomForestClassifier class. Similarly, the Extra
TreesRegressor class has the same API as the RandomForestRegressor class.
```

```
It is hard to tell in advance whether a RandomForestClassifier
will perform better or worse than an ExtraTreesClassifier. Gen‐
erally, the only way to know is to try both and compare them using
cross-validation (and tuning the hyperparameters using grid
search).
```

#### Feature Importance

```
Yet another great quality of Random Forests is that they make it easy to measure the
relative importance of each feature. Scikit-Learn measures a feature’s importance by
looking at how much the tree nodes that use that feature reduce impurity on average
(across all trees in the forest). More precisely, it is a weighted average, where each
node’s weight is equal to the number of training samples that are associated with it
(see Chapter 6).
Scikit-Learn computes this score automatically for each feature after training, then it
scales the results so that the sum of all importances is equal to 1. You can access the
result using the feature_importances_ variable. For example, the following code
trains a RandomForestClassifier on the iris dataset (introduced in Chapter 4) and
outputs each feature’s importance. It seems that the most important features are the
petal length (44%) and width (42%), while sepal length and width are rather unim‐
portant in comparison (11% and 2%, respectively).
```

```
200 | Chapter 7: Ensemble Learning and Random Forests
```

```
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
>>> rnd_clf.fit(iris["data"], iris["target"])
>>> for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
... print (name, score)
```

```
sepal length (cm) 0.112492250999
sepal width (cm) 0.0231192882825
petal length (cm) 0.441030464364
petal width (cm) 0.423357996355
```

Similarly, if you train a Random Forest classifier on the MNIST dataset (introduced
in Chapter 3) and plot each pixel’s importance, you get the image represented in
Figure 7-6.

Figure 7-6. MNIST pixel importance (according to a Random Forest classifier)

Random Forests are very handy to get a quick understanding of what features
actually matter, in particular if you need to perform feature selection.

### Boosting

Boosting (originally called hypothesis boosting) refers to any Ensemble method that
can combine several weak learners into a strong learner. The general idea of most
boosting methods is to train predictors sequentially, each trying to correct its prede‐
cessor. There are many boosting methods available, but by far the most popular are

```
Boosting | 201
```

13 “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting,” Yoav Freund,
Robert E. Schapire (1997).
14 This is just for illustrative purposes. SVMs are generally not good base predictors for AdaBoost, because they
are slow and tend to be unstable with AdaBoost.

```
AdaBoost^13 (short for Adaptive Boosting) and Gradient Boosting. Let’s start with Ada‐
Boost.
```

#### AdaBoost

```
One way for a new predictor to correct its predecessor is to pay a bit more attention
to the training instances that the predecessor underfitted. This results in new predic‐
tors focusing more and more on the hard cases. This is the technique used by Ada‐
Boost.
For example, to build an AdaBoost classifier, a first base classifier (such as a Decision
Tree) is trained and used to make predictions on the training set. The relative weight
of misclassified training instances is then increased. A second classifier is trained
using the updated weights and again it makes predictions on the training set, weights
are updated, and so on (see Figure 7-7).
```

```
Figure 7-7. AdaBoost sequential training with instance weight updates
```

```
Figure 7-8 shows the decision boundaries of five consecutive predictors on the
moons dataset (in this example, each predictor is a highly regularized SVM classifier
with an RBF kernel^14 ). The first classifier gets many instances wrong, so their weights
```

```
202 | Chapter 7: Ensemble Learning and Random Forests
```

get boosted. The second classifier therefore does a better job on these instances, and
so on. The plot on the right represents the same sequence of predictors except that
the learning rate is halved (i.e., the misclassified instance weights are boosted half as
much at every iteration). As you can see, this sequential learning technique has some
similarities with Gradient Descent, except that instead of tweaking a single predictor’s
parameters to minimize a cost function, AdaBoost adds predictors to the ensemble,
gradually making it better.

Figure 7-8. Decision boundaries of consecutive predictors

Once all predictors are trained, the ensemble makes predictions very much like bag‐
ging or pasting, except that predictors have different weights depending on their
overall accuracy on the weighted training set.

```
There is one important drawback to this sequential learning techni‐
que: it cannot be parallelized (or only partially), since each predic‐
tor can only be trained after the previous predictor has been
trained and evaluated. As a result, it does not scale as well as bag‐
ging or pasting.
```

Let’s take a closer look at the AdaBoost algorithm. Each instance weight w(i) is initially

set to m^1. A first predictor is trained and its weighted error rate r 1 is computed on the

training set; see Equation 7-1.

```
Equation 7-1. Weighted error rate of the jth predictor
```

```
rj=
```

```
∑
i= 1
yji≠yi
```

```
m
wi
```

```
∑
i= 1
```

```
m
wi
```

```
whereyji is thejthpredictor’s prediction for theithinstance.
```

```
Boosting | 203
```

15 The original AdaBoost algorithm does not use a learning rate hyperparameter.

```
The predictor’s weight αj is then computed using Equation 7-2, where η is the learn‐
ing rate hyperparameter (defaults to 1).^15 The more accurate the predictor is, the
higher its weight will be. If it is just guessing randomly, then its weight will be close to
zero. However, if it is most often wrong (i.e., less accurate than random guessing),
then its weight will be negative.
```

```
Equation 7-2. Predictor weight
```

```
αj=ηlog
```

```
1 −rj
rj
```

```
Next the instance weights are updated using Equation 7-3: the misclassified instances
are boosted.
```

```
Equation 7-3. Weight update rule
fori= 1, 2,⋯,m
```

```
wi
```

```
wi ifyji=yi
```

```
wi exp αj ifyji≠yi
```

```
Then all the instance weights are normalized (i.e., divided by ∑im= 1wi).
```

```
Finally, a new predictor is trained using the updated weights, and the whole process is
repeated (the new predictor’s weight is computed, the instance weights are updated,
then another predictor is trained, and so on). The algorithm stops when the desired
number of predictors is reached, or when a perfect predictor is found.
To make predictions, AdaBoost simply computes the predictions of all the predictors
and weighs them using the predictor weights αj. The predicted class is the one that
receives the majority of weighted votes (see Equation 7-4).
```

```
Equation 7-4. AdaBoost predictions
```

```
y x = argmax
k
```

```
∑
j= 1
yj x =k
```

```
N
αj whereNis the number of predictors.
```

```
204 | Chapter 7: Ensemble Learning and Random Forests
```

16 For more details, see “Multi-Class AdaBoost,” J. Zhu et al. (2006).
17 First introduced in “Arcing the Edge,” L. Breiman (1997), and further developed in the paper “Greedy Func‐
tion Approximation: A Gradient Boosting Machine,” Jerome H. Friedman (1999).

```
Scikit-Learn actually uses a multiclass version of AdaBoost called SAMME^16 (which
stands for Stagewise Additive Modeling using a Multiclass Exponential loss function).
When there are just two classes, SAMME is equivalent to AdaBoost. Moreover, if the
predictors can estimate class probabilities (i.e., if they have a predict_proba()
method), Scikit-Learn can use a variant of SAMME called SAMME.R (the R stands
for “Real”), which relies on class probabilities rather than predictions and generally
performs better.
The following code trains an AdaBoost classifier based on 200 Decision Stumps using
Scikit-Learn’s AdaBoostClassifier class (as you might expect, there is also an Ada
BoostRegressor class). A Decision Stump is a Decision Tree with max_depth=1—in
other words, a tree composed of a single decision node plus two leaf nodes. This is
the default base estimator for the AdaBoostClassifier class:
from sklearn.ensemble import AdaBoostClassifier
```

```
ada_clf = AdaBoostClassifier(
DecisionTreeClassifier(max_depth=1), n_estimators=200,
algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)
```

```
If your AdaBoost ensemble is overfitting the training set, you can
try reducing the number of estimators or more strongly regulariz‐
ing the base estimator.
```

#### Gradient Boosting

```
Another very popular Boosting algorithm is Gradient Boosting.^17 Just like AdaBoost,
Gradient Boosting works by sequentially adding predictors to an ensemble, each one
correcting its predecessor. However, instead of tweaking the instance weights at every
iteration like AdaBoost does, this method tries to fit the new predictor to the residual
errors made by the previous predictor.
Let’s go through a simple regression example using Decision Trees as the base predic‐
tors (of course Gradient Boosting also works great with regression tasks). This is
called Gradient Tree Boosting, or Gradient Boosted Regression Trees (GBRT). First, let’s
fit a DecisionTreeRegressor to the training set (for example, a noisy quadratic train‐
ing set):
```

```
Boosting | 205
```

```
from sklearn.tree import DecisionTreeRegressor
```

```
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
```

Now train a second DecisionTreeRegressor on the residual errors made by the first
predictor:

```
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)
```

Then we train a third regressor on the residual errors made by the second predictor:

```
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)
```

Now we have an ensemble containing three trees. It can make predictions on a new
instance simply by adding up the predictions of all the trees:

```
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

Figure 7-9 represents the predictions of these three trees in the left column, and the
ensemble’s predictions in the right column. In the first row, the ensemble has just one
tree, so its predictions are exactly the same as the first tree’s predictions. In the second
row, a new tree is trained on the residual errors of the first tree. On the right you can
see that the ensemble’s predictions are equal to the sum of the predictions of the first
two trees. Similarly, in the third row another tree is trained on the residual errors of
the second tree. You can see that the ensemble’s predictions gradually get better as
trees are added to the ensemble.

A simpler way to train GBRT ensembles is to use Scikit-Learn’s GradientBoostingRe
gressor class. Much like the RandomForestRegressor class, it has hyperparameters to
control the growth of Decision Trees (e.g., max_depth, min_samples_leaf, and so on),
as well as hyperparameters to control the ensemble training, such as the number of
trees (n_estimators). The following code creates the same ensemble as the previous
one:

```
from sklearn.ensemble import GradientBoostingRegressor
```

```
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)
```

**206 | Chapter 7: Ensemble Learning and Random Forests**

Figure 7-9. Gradient Boosting

The learning_rate hyperparameter scales the contribution of each tree. If you set it
to a low value, such as 0.1, you will need more trees in the ensemble to fit the train‐
ing set, but the predictions will usually generalize better. This is a regularization tech‐
nique called shrinkage. Figure 7-10 shows two GBRT ensembles trained with a low
learning rate: the one on the left does not have enough trees to fit the training set,
while the one on the right has too many trees and overfits the training set.

```
Boosting | 207
```

Figure 7-10. GBRT ensembles with not enough predictors (left) and too many (right)

In order to find the optimal number of trees, you can use early stopping (see Chap‐
ter 4). A simple way to implement this is to use the staged_predict() method: it
returns an iterator over the predictions made by the ensemble at each stage of train‐
ing (with one tree, two trees, etc.). The following code trains a GBRT ensemble with
120 trees, then measures the validation error at each stage of training to find the opti‐
mal number of trees, and finally trains another GBRT ensemble using the optimal
number of trees:

```
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

```
X_train, X_val, y_train, y_val = train_test_split(X, y)
```

```
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)
```

```
errors = [mean_squared_error(y_val, y_pred)
for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)
```

```
gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
```

The validation errors are represented on the left of Figure 7-11, and the best model’s
predictions are represented on the right.

**208 | Chapter 7: Ensemble Learning and Random Forests**

Figure 7-11. Tuning the number of trees using early stopping

It is also possible to implement early stopping by actually stopping training early
(instead of training a large number of trees first and then looking back to find the
optimal number). You can do so by setting warm_start=True, which makes Scikit-
Learn keep existing trees when the fit() method is called, allowing incremental
training. The following code stops training when the validation error does not
improve for five iterations in a row:

```
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
```

```
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
gbrt.n_estimators = n_estimators
gbrt.fit(X_train, y_train)
y_pred = gbrt.predict(X_val)
val_error = mean_squared_error(y_val, y_pred)
if val_error < min_val_error:
min_val_error = val_error
error_going_up = 0
else :
error_going_up += 1
if error_going_up == 5:
break # early stopping
```

The GradientBoostingRegressor class also supports a subsample hyperparameter,
which specifies the fraction of training instances to be used for training each tree. For
example, if subsample=0.25, then each tree is trained on 25% of the training instan‐
ces, selected randomly. As you can probably guess by now, this trades a higher bias
for a lower variance. It also speeds up training considerably. This technique is called
Stochastic Gradient Boosting.

```
Boosting | 209
```

18 “Stacked Generalization,” D. Wolpert (1992).

```
It is possible to use Gradient Boosting with other cost functions.
This is controlled by the loss hyperparameter (see Scikit-Learn’s
documentation for more details).
```

```
It is worth noting that an optimized implementation of Gradient Boosting is available
in the popular python library XGBoost, which stands for Extreme Gradient Boosting.
This package was initially developed by Tianqi Chen as part of the Distributed (Deep)
Machine Learning Community (DMLC), and it aims at being extremely fast, scalable
and portable. In fact, XGBoost is often an important component of the winning
entries in ML competitions. XGBoost’s API is quite similar to Scikit-Learn’s:
import xgboost
```

```
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
XGBoost also offers several nice features, such as automatically taking care of early
stopping:
xgb_reg.fit(X_train, y_train,
eval_set=[(X_val, y_val)], early_stopping_rounds=2)
y_pred = xgb_reg.predict(X_val)
You should definitely check it out!
```

### Stacking

```
The last Ensemble method we will discuss in this chapter is called stacking (short for
stacked generalization).^18 It is based on a simple idea: instead of using trivial functions
(such as hard voting) to aggregate the predictions of all predictors in an ensemble,
why don’t we train a model to perform this aggregation? Figure 7-12 shows such an
ensemble performing a regression task on a new instance. Each of the bottom three
predictors predicts a different value (3.1, 2.7, and 2.9), and then the final predictor
(called a blender, or a meta learner) takes these predictions as inputs and makes the
final prediction (3.0).
```

```
210 | Chapter 7: Ensemble Learning and Random Forests
```

19 Alternatively, it is possible to use out-of-fold predictions. In some contexts this is called stacking, while using a
hold-out set is called blending. However, for many people these terms are synonymous.

```
Figure 7-12. Aggregating predictions using a blending predictor
```

```
To train the blender, a common approach is to use a hold-out set.^19 Let’s see how it
works. First, the training set is split in two subsets. The first subset is used to train the
predictors in the first layer (see Figure 7-13).
```

```
Figure 7-13. Training the first layer
```

```
Next, the first layer predictors are used to make predictions on the second (held-out)
set (see Figure 7-14). This ensures that the predictions are “clean,” since the predictors
never saw these instances during training. Now for each instance in the hold-out set
```

```
Stacking | 211
```

there are three predicted values. We can create a new training set using these predic‐
ted values as input features (which makes this new training set three-dimensional),
and keeping the target values. The blender is trained on this new training set, so it
learns to predict the target value given the first layer’s predictions.

Figure 7-14. Training the blender

It is actually possible to train several different blenders this way (e.g., one using Lin‐
ear Regression, another using Random Forest Regression, and so on): we get a whole
layer of blenders. The trick is to split the training set into three subsets: the first one is
used to train the first layer, the second one is used to create the training set used to
train the second layer (using predictions made by the predictors of the first layer),
and the third one is used to create the training set to train the third layer (using pre‐
dictions made by the predictors of the second layer). Once this is done, we can make
a prediction for a new instance by going through each layer sequentially, as shown in
Figure 7-15.

**212 | Chapter 7: Ensemble Learning and Random Forests**

Figure 7-15. Predictions in a multilayer stacking ensemble

Unfortunately, Scikit-Learn does not support stacking directly, but it is not too hard
to roll out your own implementation (see the following exercises). Alternatively, you
can use an open source implementation such as brew (available at https://github.com/
viisar/brew).

### Exercises

1. If you have trained five different models on the exact same training data, and
   they all achieve 95% precision, is there any chance that you can combine these
   models to get better results? If so, how? If not, why?
2. What is the difference between hard and soft voting classifiers?
3. Is it possible to speed up training of a bagging ensemble by distributing it across
   multiple servers? What about pasting ensembles, boosting ensembles, random
   forests, or stacking ensembles?
4. What is the benefit of out-of-bag evaluation?
5. What makes Extra-Trees more random than regular Random Forests? How can
   this extra randomness help? Are Extra-Trees slower or faster than regular Ran‐
   dom Forests?
6. If your AdaBoost ensemble underfits the training data, what hyperparameters
   should you tweak and how?

```
Exercises | 213
```

7. If your Gradient Boosting ensemble overfits the training set, should you increase
   or decrease the learning rate?
8. Load the MNIST data (introduced in Chapter 3), and split it into a training set, a
   validation set, and a test set (e.g., use 50,000 instances for training, 10,000 for val‐
   idation, and 10,000 for testing). Then train various classifiers, such as a Random
   Forest classifier, an Extra-Trees classifier, and an SVM. Next, try to combine
   them into an ensemble that outperforms them all on the validation set, using a
   soft or hard voting classifier. Once you have found one, try it on the test set. How
   much better does it perform compared to the individual classifiers?
9. Run the individual classifiers from the previous exercise to make predictions on
   the validation set, and create a new training set with the resulting predictions:
   each training instance is a vector containing the set of predictions from all your
   classifiers for an image, and the target is the image’s class. Train a classifier on
   this new training set. Congratulations, you have just trained a blender, and
   together with the classifiers they form a stacking ensemble! Now let’s evaluate the
   ensemble on the test set. For each image in the test set, make predictions with all
   your classifiers, then feed the predictions to the blender to get the ensemble’s pre‐
   dictions. How does it compare to the voting classifier you trained earlier?

Solutions to these exercises are available in ???.

**214 | Chapter 7: Ensemble Learning and Random Forests**

**CHAPTER 8**

**Dimensionality Reduction**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 8 in the final
release of the book.
```

Many Machine Learning problems involve thousands or even millions of features for
each training instance. Not only does this make training extremely slow, it can also
make it much harder to find a good solution, as we will see. This problem is often
referred to as the curse of dimensionality.

Fortunately, in real-world problems, it is often possible to reduce the number of fea‐
tures considerably, turning an intractable problem into a tractable one. For example,
consider the MNIST images (introduced in Chapter 3): the pixels on the image bor‐
ders are almost always white, so you could completely drop these pixels from the
training set without losing much information. Figure 7-6 confirms that these pixels
are utterly unimportant for the classification task. Moreover, two neighboring pixels
are often highly correlated: if you merge them into a single pixel (e.g., by taking the
mean of the two pixel intensities), you will not lose much information.

```
215
```

```
1 Well, four dimensions if you count time, and a few more if you are a string theorist.
```

```
Reducing dimensionality does lose some information (just like
compressing an image to JPEG can degrade its quality), so even
though it will speed up training, it may also make your system per‐
form slightly worse. It also makes your pipelines a bit more com‐
plex and thus harder to maintain. So you should first try to train
your system with the original data before considering using dimen‐
sionality reduction if training is too slow. In some cases, however,
reducing the dimensionality of the training data may filter out
some noise and unnecessary details and thus result in higher per‐
formance (but in general it won’t; it will just speed up training).
```

Apart from speeding up training, dimensionality reduction is also extremely useful
for data visualization (or DataViz). Reducing the number of dimensions down to two
(or three) makes it possible to plot a condensed view of a high-dimensional training
set on a graph and often gain some important insights by visually detecting patterns,
such as clusters. Moreover, DataViz is essential to communicate your conclusions to
people who are not data scientists, in particular decision makers who will use your
results.

In this chapter we will discuss the curse of dimensionality and get a sense of what
goes on in high-dimensional space. Then, we will present the two main approaches to
dimensionality reduction (projection and Manifold Learning), and we will go
through three of the most popular dimensionality reduction techniques: PCA, Kernel
PCA, and LLE.

### The Curse of Dimensionality

We are so used to living in three dimensions^1 that our intuition fails us when we try
to imagine a high-dimensional space. Even a basic 4D hypercube is incredibly hard to
picture in our mind (see Figure 8-1), let alone a 200-dimensional ellipsoid bent in a
1,000-dimensional space.

**216 | Chapter 8: Dimensionality Reduction**

```
2 Watch a rotating tesseract projected into 3D space at https://homl.info/30. Image by Wikipedia user Nerd‐
Boy1392 (Creative Commons BY-SA 3.0). Reproduced from https://en.wikipedia.org/wiki/Tesseract.
3 Fun fact: anyone you know is probably an extremist in at least one dimension (e.g., how much sugar they put
in their coffee), if you consider enough dimensions.
```

Figure 8-1. Point, segment, square, cube, and tesseract (0D to 4D hypercubes)^2

It turns out that many things behave very differently in high-dimensional space. For
example, if you pick a random point in a unit square (a 1 × 1 square), it will have only
about a 0.4% chance of being located less than 0.001 from a border (in other words, it
is very unlikely that a random point will be “extreme” along any dimension). But in a
10,000-dimensional unit hypercube (a 1 × 1 × ⋯ × 1 cube, with ten thousand 1s), this
probability is greater than 99.999999%. Most points in a high-dimensional hypercube
are very close to the border.^3

Here is a more troublesome difference: if you pick two points randomly in a unit
square, the distance between these two points will be, on average, roughly 0.52. If you
pick two random points in a unit 3D cube, the average distance will be roughly 0.66.
But what about two points picked randomly in a 1,000,000-dimensional hypercube?
Well, the average distance, believe it or not, will be about 408.25 (roughly
1, 000, 000/6)! This is quite counterintuitive: how can two points be so far apart
when they both lie within the same unit hypercube? This fact implies that high-
dimensional datasets are at risk of being very sparse: most training instances are
likely to be far away from each other. Of course, this also means that a new instance
will likely be far away from any training instance, making predictions much less relia‐
ble than in lower dimensions, since they will be based on much larger extrapolations.
In short, the more dimensions the training set has, the greater the risk of overfitting
it.

In theory, one solution to the curse of dimensionality could be to increase the size of
the training set to reach a sufficient density of training instances. Unfortunately, in
practice, the number of training instances required to reach a given density grows
exponentially with the number of dimensions. With just 100 features (much less than

```
The Curse of Dimensionality | 217
```

in the MNIST problem), you would need more training instances than atoms in the
observable universe in order for training instances to be within 0.1 of each other on
average, assuming they were spread out uniformly across all dimensions.

### Main Approaches for Dimensionality Reduction

Before we dive into specific dimensionality reduction algorithms, let’s take a look at
the two main approaches to reducing dimensionality: projection and Manifold
Learning.

#### Projection

In most real-world problems, training instances are not spread out uniformly across
all dimensions. Many features are almost constant, while others are highly correlated
(as discussed earlier for MNIST). As a result, all training instances actually lie within
(or close to) a much lower-dimensional subspace of the high-dimensional space. This
sounds very abstract, so let’s look at an example. In Figure 8-2 you can see a 3D data‐
set represented by the circles.

Figure 8-2. A 3D dataset lying close to a 2D subspace

Notice that all training instances lie close to a plane: this is a lower-dimensional (2D)
subspace of the high-dimensional (3D) space. Now if we project every training
instance perpendicularly onto this subspace (as represented by the short lines con‐
necting the instances to the plane), we get the new 2D dataset shown in Figure 8-3.
Ta-da! We have just reduced the dataset’s dimensionality from 3D to 2D. Note that
the axes correspond to new features z 1 and z 2 (the coordinates of the projections on
the plane).

**218 | Chapter 8: Dimensionality Reduction**

Figure 8-3. The new 2D dataset after projection

However, projection is not always the best approach to dimensionality reduction. In
many cases the subspace may twist and turn, such as in the famous Swiss roll toy data‐
set represented in Figure 8-4.

Figure 8-4. Swiss roll dataset

```
Main Approaches for Dimensionality Reduction | 219
```

Simply projecting onto a plane (e.g., by dropping x 3 ) would squash different layers of
the Swiss roll together, as shown on the left of Figure 8-5. However, what you really
want is to unroll the Swiss roll to obtain the 2D dataset on the right of Figure 8-5.

Figure 8-5. Squashing by projecting onto a plane (left) versus unrolling the Swiss roll
(right)

#### Manifold Learning

The Swiss roll is an example of a 2D manifold. Put simply, a 2D manifold is a 2D
shape that can be bent and twisted in a higher-dimensional space. More generally, a
d-dimensional manifold is a part of an n-dimensional space (where d < n) that locally
resembles a d-dimensional hyperplane. In the case of the Swiss roll, d = 2 and n = 3: it
locally resembles a 2D plane, but it is rolled in the third dimension.

Many dimensionality reduction algorithms work by modeling the manifold on which
the training instances lie; this is called Manifold Learning. It relies on the manifold
assumption, also called the manifold hypothesis, which holds that most real-world
high-dimensional datasets lie close to a much lower-dimensional manifold. This
assumption is very often empirically observed.

Once again, think about the MNIST dataset: all handwritten digit images have some
similarities. They are made of connected lines, the borders are white, they are more
or less centered, and so on. If you randomly generated images, only a ridiculously
tiny fraction of them would look like handwritten digits. In other words, the degrees
of freedom available to you if you try to create a digit image are dramatically lower
than the degrees of freedom you would have if you were allowed to generate any
image you wanted. These constraints tend to squeeze the dataset into a lower-
dimensional manifold.

The manifold assumption is often accompanied by another implicit assumption: that
the task at hand (e.g., classification or regression) will be simpler if expressed in the
lower-dimensional space of the manifold. For example, in the top row of Figure 8-6
the Swiss roll is split into two classes: in the 3D space (on the left), the decision

**220 | Chapter 8: Dimensionality Reduction**

boundary would be fairly complex, but in the 2D unrolled manifold space (on the
right), the decision boundary is a simple straight line.

However, this assumption does not always hold. For example, in the bottom row of
Figure 8-6, the decision boundary is located at x 1 = 5. This decision boundary looks
very simple in the original 3D space (a vertical plane), but it looks more complex in
the unrolled manifold (a collection of four independent line segments).

In short, if you reduce the dimensionality of your training set before training a
model, it will usually speed up training, but it may not always lead to a better or sim‐
pler solution; it all depends on the dataset.

Hopefully you now have a good sense of what the curse of dimensionality is and how
dimensionality reduction algorithms can fight it, especially when the manifold
assumption holds. The rest of this chapter will go through some of the most popular
algorithms.

Figure 8-6. The decision boundary may not always be simpler with lower dimensions

```
Main Approaches for Dimensionality Reduction | 221
```

```
4 “On Lines and Planes of Closest Fit to Systems of Points in Space,” K. Pearson (1901).
```

### PCA

Principal Component Analysis (PCA) is by far the most popular dimensionality reduc‐
tion algorithm. First it identifies the hyperplane that lies closest to the data, and then
it projects the data onto it, just like in Figure 8-2.

#### Preserving the Variance

Before you can project the training set onto a lower-dimensional hyperplane, you
first need to choose the right hyperplane. For example, a simple 2D dataset is repre‐
sented on the left of Figure 8-7, along with three different axes (i.e., one-dimensional
hyperplanes). On the right is the result of the projection of the dataset onto each of
these axes. As you can see, the projection onto the solid line preserves the maximum
variance, while the projection onto the dotted line preserves very little variance, and
the projection onto the dashed line preserves an intermediate amount of variance.

Figure 8-7. Selecting the subspace onto which to project

It seems reasonable to select the axis that preserves the maximum amount of var‐
iance, as it will most likely lose less information than the other projections. Another
way to justify this choice is that it is the axis that minimizes the mean squared dis‐
tance between the original dataset and its projection onto that axis. This is the rather
simple idea behind PCA.^4

**222 | Chapter 8: Dimensionality Reduction**

#### Principal Components

PCA identifies the axis that accounts for the largest amount of variance in the train‐
ing set. In Figure 8-7, it is the solid line. It also finds a second axis, orthogonal to the
first one, that accounts for the largest amount of remaining variance. In this 2D
example there is no choice: it is the dotted line. If it were a higher-dimensional data‐
set, PCA would also find a third axis, orthogonal to both previous axes, and a fourth,
a fifth, and so on—as many axes as the number of dimensions in the dataset.

The unit vector that defines the ith axis is called the ith principal component (PC). In
Figure 8-7, the 1st PC is **c 1** and the 2nd PC is **c 2**. In Figure 8-2 the first two PCs are
represented by the orthogonal arrows in the plane, and the third PC would be
orthogonal to the plane (pointing up or down).

```
The direction of the principal components is not stable: if you per‐
turb the training set slightly and run PCA again, some of the new
PCs may point in the opposite direction of the original PCs. How‐
ever, they will generally still lie on the same axes. In some cases, a
pair of PCs may even rotate or swap, but the plane they define will
generally remain the same.
```

So how can you find the principal components of a training set? Luckily, there is a
standard matrix factorization technique called Singular Value Decomposition (SVD)
that can decompose the training set matrix **X** into the matrix multiplication of three
matrices **U** Σ **V** T, where **V** contains all the principal components that we are looking
for, as shown in Equation 8-1.

```
Equation 8-1. Principal components matrix
```

```
V =
```

```
∣ ∣ ∣
c 1 c 2 ⋯ cn
∣ ∣ ∣
```

The following Python code uses NumPy’s svd() function to obtain all the principal
components of the training set, then extracts the first two PCs:

```
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
```

```
PCA | 223
```

```
PCA assumes that the dataset is centered around the origin. As we
will see, Scikit-Learn’s PCA classes take care of centering the data
for you. However, if you implement PCA yourself (as in the pre‐
ceding example), or if you use other libraries, don’t forget to center
the data first.
```

#### Projecting Down to d Dimensions

Once you have identified all the principal components, you can reduce the dimen‐
sionality of the dataset down to d dimensions by projecting it onto the hyperplane
defined by the first d principal components. Selecting this hyperplane ensures that the
projection will preserve as much variance as possible. For example, in Figure 8-2 the
3D dataset is projected down to the 2D plane defined by the first two principal com‐
ponents, preserving a large part of the dataset’s variance. As a result, the 2D projec‐
tion looks very much like the original 3D dataset.

To project the training set onto the hyperplane, you can simply compute the matrix
multiplication of the training set matrix **X** by the matrix **W** d, defined as the matrix
containing the first d principal components (i.e., the matrix composed of the first d
columns of **V** ), as shown in Equation 8-2.

```
Equation 8-2. Projecting the training set down to d dimensions
X d‐proj= XW d
```

The following Python code projects the training set onto the plane defined by the first
two principal components:

```
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
```

There you have it! You now know how to reduce the dimensionality of any dataset
down to any number of dimensions, while preserving as much variance as possible.

#### Using Scikit-Learn

Scikit-Learn’s PCA class implements PCA using SVD decomposition just like we did
before. The following code applies PCA to reduce the dimensionality of the dataset
down to two dimensions (note that it automatically takes care of centering the data):

```
from sklearn.decomposition import PCA
```

```
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
```

After fitting the PCA transformer to the dataset, you can access the principal compo‐
nents using the components\_ variable (note that it contains the PCs as horizontal vec‐

**224 | Chapter 8: Dimensionality Reduction**

tors, so, for example, the first principal component is equal to pca.components\_.T[:,
0]).

#### Explained Variance Ratio

Another very useful piece of information is the explained variance ratio of each prin‐
cipal component, available via the explained*variance_ratio* variable. It indicates
the proportion of the dataset’s variance that lies along the axis of each principal com‐
ponent. For example, let’s look at the explained variance ratios of the first two compo‐
nents of the 3D dataset represented in Figure 8-2:

```
>>> pca.explained_variance_ratio_
array([0.84248607, 0.14631839])
```

This tells you that 84.2% of the dataset’s variance lies along the first axis, and 14.6%
lies along the second axis. This leaves less than 1.2% for the third axis, so it is reason‐
able to assume that it probably carries little information.

#### Choosing the Right Number of Dimensions

Instead of arbitrarily choosing the number of dimensions to reduce down to, it is
generally preferable to choose the number of dimensions that add up to a sufficiently
large portion of the variance (e.g., 95%). Unless, of course, you are reducing dimen‐
sionality for data visualization—in that case you will generally want to reduce the
dimensionality down to 2 or 3.

The following code computes PCA without reducing dimensionality, then computes
the minimum number of dimensions required to preserve 95% of the training set’s
variance:

```
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
```

You could then set n_components=d and run PCA again. However, there is a much
better option: instead of specifying the number of principal components you want to
preserve, you can set n_components to be a float between 0.0 and 1.0, indicating the
ratio of variance you wish to preserve:

```
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
```

Yet another option is to plot the explained variance as a function of the number of
dimensions (simply plot cumsum; see Figure 8-8). There will usually be an elbow in the
curve, where the explained variance stops growing fast. You can think of this as the
intrinsic dimensionality of the dataset. In this case, you can see that reducing the

```
PCA | 225
```

dimensionality down to about 100 dimensions wouldn’t lose too much explained var‐
iance.

Figure 8-8. Explained variance as a function of the number of dimensions

#### PCA for Compression

Obviously after dimensionality reduction, the training set takes up much less space.
For example, try applying PCA to the MNIST dataset while preserving 95% of its var‐
iance. You should find that each instance will have just over 150 features, instead of
the original 784 features. So while most of the variance is preserved, the dataset is
now less than 20% of its original size! This is a reasonable compression ratio, and you
can see how this can speed up a classification algorithm (such as an SVM classifier)
tremendously.

It is also possible to decompress the reduced dataset back to 784 dimensions by
applying the inverse transformation of the PCA projection. Of course this won’t give
you back the original data, since the projection lost a bit of information (within the
5% variance that was dropped), but it will likely be quite close to the original data.
The mean squared distance between the original data and the reconstructed data
(compressed and then decompressed) is called the reconstruction error. For example,
the following code compresses the MNIST dataset down to 154 dimensions, then uses
the inverse_transform() method to decompress it back to 784 dimensions.
Figure 8-9 shows a few digits from the original training set (on the left), and the cor‐
responding digits after compression and decompression. You can see that there is a
slight image quality loss, but the digits are still mostly intact.

```
pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)
```

**226 | Chapter 8: Dimensionality Reduction**

Figure 8-9. MNIST compression preserving 95% of the variance

The equation of the inverse transformation is shown in Equation 8-3.

```
Equation 8-3. PCA inverse transformation, back to the original number of
dimensions
```

```
X recovered= X d‐proj W dT
```

##### Randomized PCA

If you set the svd_solver hyperparameter to "randomized", Scikit-Learn uses a sto‐
chastic algorithm called Randomized PCA that quickly finds an approximation of the
first d principal components. Its computational complexity is O(m × d^2 ) + O(d^3 ),
instead of O(m × n^2 ) + O(n^3 ) for the full SVD approach, so it is dramatically faster
than full SVD when d is much smaller than n:

```
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)
```

By default, svd_solver is actually set to "auto": Scikit-Learn automatically uses the
randomized PCA algorithm if m or n is greater than 500 and d is less than 80% of m
or n, or else it uses the full SVD approach. If you want to force Scikit-Learn to use full
SVD, you can set the svd_solver hyperparameter to "full".

##### Incremental PCA

One problem with the preceding implementations of PCA is that they require the
whole training set to fit in memory in order for the algorithm to run. Fortunately,
Incremental PCA (IPCA) algorithms have been developed: you can split the training
set into mini-batches and feed an IPCA algorithm one mini-batch at a time. This is

```
PCA | 227
```

```
5 Scikit-Learn uses the algorithm described in “Incremental Learning for Robust Visual Tracking,” D. Ross et al.
(2007).
```

useful for large training sets, and also to apply PCA online (i.e., on the fly, as new
instances arrive).

The following code splits the MNIST dataset into 100 mini-batches (using NumPy’s
array_split() function) and feeds them to Scikit-Learn’s IncrementalPCA class^5 to
reduce the dimensionality of the MNIST dataset down to 154 dimensions (just like
before). Note that you must call the partial_fit() method with each mini-batch
rather than the fit() method with the whole training set:

```
from sklearn.decomposition import IncrementalPCA
```

```
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
inc_pca.partial_fit(X_batch)
```

```
X_reduced = inc_pca.transform(X_train)
```

Alternatively, you can use NumPy’s memmap class, which allows you to manipulate a
large array stored in a binary file on disk as if it were entirely in memory; the class
loads only the data it needs in memory, when it needs it. Since the IncrementalPCA
class uses only a small part of the array at any given time, the memory usage remains
under control. This makes it possible to call the usual fit() method, as you can see
in the following code:

```
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
```

```
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
```

#### Kernel PCA

In Chapter 5 we discussed the kernel trick, a mathematical technique that implicitly
maps instances into a very high-dimensional space (called the feature space), enabling
nonlinear classification and regression with Support Vector Machines. Recall that a
linear decision boundary in the high-dimensional feature space corresponds to a
complex nonlinear decision boundary in the original space.

It turns out that the same trick can be applied to PCA, making it possible to perform
complex nonlinear projections for dimensionality reduction. This is called Kernel

**228 | Chapter 8: Dimensionality Reduction**

```
6 “Kernel Principal Component Analysis,” B. Schölkopf, A. Smola, K. Müller (1999).
```

PCA (kPCA).^6 It is often good at preserving clusters of instances after projection, or
sometimes even unrolling datasets that lie close to a twisted manifold.

For example, the following code uses Scikit-Learn’s KernelPCA class to perform kPCA
with an RBF kernel (see Chapter 5 for more details about the RBF kernel and the
other kernels):

```
from sklearn.decomposition import KernelPCA
```

```
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
```

Figure 8-10 shows the Swiss roll, reduced to two dimensions using a linear kernel
(equivalent to simply using the PCA class), an RBF kernel, and a sigmoid kernel
(Logistic).

Figure 8-10. Swiss roll reduced to 2D using kPCA with various kernels

##### Selecting a Kernel and Tuning Hyperparameters

As kPCA is an unsupervised learning algorithm, there is no obvious performance
measure to help you select the best kernel and hyperparameter values. However,
dimensionality reduction is often a preparation step for a supervised learning task
(e.g., classification), so you can simply use grid search to select the kernel and hyper‐
parameters that lead to the best performance on that task. For example, the following
code creates a two-step pipeline, first reducing dimensionality to two dimensions
using kPCA, then applying Logistic Regression for classification. Then it uses Grid
SearchCV to find the best kernel and gamma value for kPCA in order to get the best
classification accuracy at the end of the pipeline:

```
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
```

```
Kernel PCA | 229
```

```
clf = Pipeline([
("kpca", KernelPCA(n_components=2)),
("log_reg", LogisticRegression())
])
```

```
param_grid = [{
"kpca__gamma": np.linspace(0.03, 0.05, 10),
"kpca__kernel": ["rbf", "sigmoid"]
}]
```

```
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
```

The best kernel and hyperparameters are then available through the best*params*
variable:

```
>>> print (grid_search.best_params_)
{'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}
```

Another approach, this time entirely unsupervised, is to select the kernel and hyper‐
parameters that yield the lowest reconstruction error. However, reconstruction is not
as easy as with linear PCA. Here’s why. Figure 8-11 shows the original Swiss roll 3D
dataset (top left), and the resulting 2D dataset after kPCA is applied using an RBF
kernel (top right). Thanks to the kernel trick, this is mathematically equivalent to
mapping the training set to an infinite-dimensional feature space (bottom right)
using the feature map φ, then projecting the transformed training set down to 2D
using linear PCA. Notice that if we could invert the linear PCA step for a given
instance in the reduced space, the reconstructed point would lie in feature space, not
in the original space (e.g., like the one represented by an x in the diagram). Since the
feature space is infinite-dimensional, we cannot compute the reconstructed point,
and therefore we cannot compute the true reconstruction error. Fortunately, it is pos‐
sible to find a point in the original space that would map close to the reconstructed
point. This is called the reconstruction pre-image. Once you have this pre-image, you
can measure its squared distance to the original instance. You can then select the ker‐
nel and hyperparameters that minimize this reconstruction pre-image error.

**230 | Chapter 8: Dimensionality Reduction**

```
7 Scikit-Learn uses the algorithm based on Kernel Ridge Regression described in Gokhan H. Bakır, Jason
Weston, and Bernhard Scholkopf, “Learning to Find Pre-images” (Tubingen, Germany: Max Planck Institute
for Biological Cybernetics, 2004).
```

Figure 8-11. Kernel PCA and the reconstruction pre-image error

You may be wondering how to perform this reconstruction. One solution is to train a
supervised regression model, with the projected instances as the training set and the
original instances as the targets. Scikit-Learn will do this automatically if you set
fit_inverse_transform=True, as shown in the following code:^7

```
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
```

```
By default, fit_inverse_transform=False and KernelPCA has no
inverse_transform() method. This method only gets created
when you set fit_inverse_transform=True.
```

```
Kernel PCA | 231
```

```
8 “Nonlinear Dimensionality Reduction by Locally Linear Embedding,” S. Roweis, L. Saul (2000).
```

You can then compute the reconstruction pre-image error:

```
>>> from sklearn.metrics import mean_squared_error
>>> mean_squared_error(X, X_preimage)
32.786308795766132
```

Now you can use grid search with cross-validation to find the kernel and hyperpara‐
meters that minimize this pre-image reconstruction error.

#### LLE

Locally Linear Embedding (LLE)^8 is another very powerful nonlinear dimensionality
reduction (NLDR) technique. It is a Manifold Learning technique that does not rely
on projections like the previous algorithms. In a nutshell, LLE works by first measur‐
ing how each training instance linearly relates to its closest neighbors (c.n.), and then
looking for a low-dimensional representation of the training set where these local
relationships are best preserved (more details shortly). This makes it particularly
good at unrolling twisted manifolds, especially when there is not too much noise.

For example, the following code uses Scikit-Learn’s LocallyLinearEmbedding class to
unroll the Swiss roll. The resulting 2D dataset is shown in Figure 8-12. As you can
see, the Swiss roll is completely unrolled and the distances between instances are
locally well preserved. However, distances are not preserved on a larger scale: the left
part of the unrolled Swiss roll is stretched, while the right part is squeezed. Neverthe‐
less, LLE did a pretty good job at modeling the manifold.

```
from sklearn.manifold import LocallyLinearEmbedding
```

```
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
```

**232 | Chapter 8: Dimensionality Reduction**

Figure 8-12. Unrolled Swiss roll using LLE

Here’s how LLE works: first, for each training instance **x** (i), the algorithm identifies its
k closest neighbors (in the preceding code k = 10), then tries to reconstruct **x** (i) as a
linear function of these neighbors. More specifically, it finds the weights wi,j such that
the squared distance between **x** (i) and ∑mj= 1wi,j **x** j is as small as possible, assuming wi,j
= 0 if **x** (j) is not one of the k closest neighbors of **x** (i). Thus the first step of LLE is the
constrained optimization problem described in Equation 8-4, where **W** is the weight
matrix containing all the weights wi,j. The second constraint simply normalizes the
weights for each training instance **x** (i).

```
LLE | 233
```

```
Equation 8-4. LLE step 1: linearly modeling local relationships
```

```
W = argmin
W
```

```
∑
i= 1
```

```
m
x i− ∑
j= 1
```

```
m
wi,j x j
```

```
2
```

```
subject to
```

```
wi,j= 0 if x j is not one of thekc.n. of x i
```

```
∑
j= 1
```

```
m
wi,j= 1 fori= 1, 2,⋯,m
```

After this step, the weight matrix **W** (containing the weights wi,j) encodes the local
linear relationships between the training instances. Now the second step is to map the
training instances into a d-dimensional space (where d < n) while preserving these
local relationships as much as possible. If **z** (i) is the image of **x** (i) in this d-dimensional

space, then we want the squared distance between **z** (i) and ∑mj= 1wi,j **z** j to be as small
as possible. This idea leads to the unconstrained optimization problem described in
Equation 8-5. It looks very similar to the first step, but instead of keeping the instan‐
ces fixed and finding the optimal weights, we are doing the reverse: keeping the
weights fixed and finding the optimal position of the instances’ images in the low-
dimensional space. Note that **Z** is the matrix containing all **z** (i).

```
Equation 8-5. LLE step 2: reducing dimensionality while preserving relationships
```

```
Z = argmin
Z
i∑= 1
```

```
m
z i− ∑
j= 1
```

```
m
wi,j z j
```

```
2
```

Scikit-Learn’s LLE implementation has the following computational complexity:
O(m log(m)n log(k)) for finding the k nearest neighbors, O(mnk^3 ) for optimizing the
weights, and O(dm^2 ) for constructing the low-dimensional representations. Unfortu‐
nately, the m^2 in the last term makes this algorithm scale poorly to very large datasets.

#### Other Dimensionality Reduction Techniques

There are many other dimensionality reduction techniques, several of which are
available in Scikit-Learn. Here are some of the most popular:

- Multidimensional Scaling (MDS) reduces dimensionality while trying to preserve
  the distances between the instances (see Figure 8-13).

**234 | Chapter 8: Dimensionality Reduction**

```
9 The geodesic distance between two nodes in a graph is the number of nodes on the shortest path between
these nodes.
```

- Isomap creates a graph by connecting each instance to its nearest neighbors, then
  reduces dimensionality while trying to preserve the geodesic distances^9 between
  the instances.
- t-Distributed Stochastic Neighbor Embedding (t-SNE) reduces dimensionality
  while trying to keep similar instances close and dissimilar instances apart. It is
  mostly used for visualization, in particular to visualize clusters of instances in
  high-dimensional space (e.g., to visualize the MNIST images in 2D).
- Linear Discriminant Analysis (LDA) is actually a classification algorithm, but dur‐
  ing training it learns the most discriminative axes between the classes, and these
  axes can then be used to define a hyperplane onto which to project the data. The
  benefit is that the projection will keep classes as far apart as possible, so LDA is a
  good technique to reduce dimensionality before running another classification
  algorithm such as an SVM classifier.

Figure 8-13. Reducing the Swiss roll to 2D using various techniques

#### Exercises

1. What are the main motivations for reducing a dataset’s dimensionality? What are
   the main drawbacks?
2. What is the curse of dimensionality?
3. Once a dataset’s dimensionality has been reduced, is it possible to reverse the
   operation? If so, how? If not, why?
4. Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?
5. Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained
   variance ratio to 95%. How many dimensions will the resulting dataset have?

```
Exercises | 235
```

6. In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA,
   or Kernel PCA?
7. How can you evaluate the performance of a dimensionality reduction algorithm
   on your dataset?
8. Does it make any sense to chain two different dimensionality reduction algo‐
   rithms?
9. Load the MNIST dataset (introduced in Chapter 3) and split it into a training set
   and a test set (take the first 60,000 instances for training, and the remaining
   10,000 for testing). Train a Random Forest classifier on the dataset and time how
   long it takes, then evaluate the resulting model on the test set. Next, use PCA to
   reduce the dataset’s dimensionality, with an explained variance ratio of 95%.
   Train a new Random Forest classifier on the reduced dataset and see how long it
   takes. Was training much faster? Next evaluate the classifier on the test set: how
   does it compare to the previous classifier?
10. Use t-SNE to reduce the MNIST dataset down to two dimensions and plot the
    result using Matplotlib. You can use a scatterplot using 10 different colors to rep‐
    resent each image’s target class. Alternatively, you can write colored digits at the
    location of each instance, or even plot scaled-down versions of the digit images
    themselves (if you plot all digits, the visualization will be too cluttered, so you
    should either draw a random sample or plot an instance only if no other instance
    has already been plotted at a close distance). You should get a nice visualization
    with well-separated clusters of digits. Try using other dimensionality reduction
    algorithms such as PCA, LLE, or MDS and compare the resulting visualizations.

Solutions to these exercises are available in ???.

**236 | Chapter 8: Dimensionality Reduction**

**CHAPTER 9**

**Unsupervised Learning Techniques**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 9 in the final
release of the book.
```

Although most of the applications of Machine Learning today are based on super‐
vised learning (and as a result, this is where most of the investments go to), the vast
majority of the available data is actually unlabeled: we have the input features **X** , but
we do not have the labels **y**. Yann LeCun famously said that “if intelligence was a cake,
unsupervised learning would be the cake, supervised learning would be the icing on
the cake, and reinforcement learning would be the cherry on the cake”. In other
words, there is a huge potential in unsupervised learning that we have only barely
started to sink our teeth into.

For example, say you want to create a system that will take a few pictures of each item
on a manufacturing production line and detect which items are defective. You can
fairly easily create a system that will take pictures automatically, and this might give
you thousands of pictures every day. You can then build a reasonably large dataset in
just a few weeks. But wait, there are no labels! If you want to train a regular binary
classifier that will predict whether an item is defective or not, you will need to label
every single picture as “defective” or “normal”. This will generally require human
experts to sit down and manually go through all the pictures. This is a long, costly
and tedious task, so it will usually only be done on a small subset of the available pic‐
tures. As a result, the labeled dataset will be quite small, and the classifier’s perfor‐
mance will be disappointing. Moreover, every time the company makes any change to
its products, the whole process will need to be started over from scratch. Wouldn’t it

```
237
```

be great if the algorithm could just exploit the unlabeled data without needing
humans to label every picture? Enter unsupervised learning.

In Chapter 8, we looked at the most common unsupervised learning task: dimension‐
ality reduction. In this chapter, we will look at a few more unsupervised learning tasks
and algorithms:

- Clustering: the goal is to group similar instances together into clusters. This is a
  great tool for data analysis, customer segmentation, recommender systems,
  search engines, image segmentation, semi-supervised learning, dimensionality
  reduction, and more.
- Anomaly detection: the objective is to learn what “normal” data looks like, and
  use this to detect abnormal instances, such as defective items on a production
  line or a new trend in a time series.
- Density estimation: this is the task of estimating the probability density function
  (PDF) of the random process that generated the dataset. This is commonly used
  for anomaly detection: instances located in very low-density regions are likely to
  be anomalies. It is also useful for data analysis and visualization.

Ready for some cake? We will start with clustering, using K-Means and DBSCAN,
and then we will discuss Gaussian mixture models and see how they can be used for
density estimation, clustering, and anomaly detection.

#### Clustering

As you enjoy a hike in the mountains, you stumble upon a plant you have never seen
before. You look around and you notice a few more. They are not perfectly identical,
yet they are sufficiently similar for you to know that they most likely belong to the
same species (or at least the same genus). You may need a botanist to tell you what
species that is, but you certainly don’t need an expert to identify groups of similar-
looking objects. This is called clustering: it is the task of identifying similar instances
and assigning them to clusters, i.e., groups of similar instances.

Just like in classification, each instance gets assigned to a group. However, this is an
unsupervised task. Consider Figure 9-1: on the left is the iris dataset (introduced in
Chapter 4), where each instance’s species (i.e., its class) is represented with a different
marker. It is a labeled dataset, for which classification algorithms such as Logistic
Regression, SVMs or Random Forest classifiers are well suited. On the right is the
same dataset, but without the labels, so you cannot use a classification algorithm any‐
more. This is where clustering algorithms step in: many of them can easily detect the
top left cluster. It is also quite easy to see with our own eyes, but it is not so obvious
that the lower right cluster is actually composed of two distinct sub-clusters. That
said, the dataset actually has two additional features (sepal length and width), not

**238 | Chapter 9: Unsupervised Learning Techniques**

represented here, and clustering algorithms can make good use of all features, so in
fact they identify the three clusters fairly well (e.g., using a Gaussian mixture model,
only 5 instances out of 150 are assigned to the wrong cluster).

Figure 9-1. Classification (left) versus clustering (right)

Clustering is used in a wide variety of applications, including:

- For customer segmentation: you can cluster your customers based on their pur‐
  chases, their activity on your website, and so on. This is useful to understand who
  your customers are and what they need, so you can adapt your products and
  marketing campaigns to each segment. For example, this can be useful in recom‐
  mender systems to suggest content that other users in the same cluster enjoyed.
- For data analysis: when analyzing a new dataset, it is often useful to first discover
  clusters of similar instances, as it is often easier to analyze clusters separately.
- As a dimensionality reduction technique: once a dataset has been clustered, it is
  usually possible to measure each instance’s affinity with each cluster (affinity is
  any measure of how well an instance fits into a cluster). Each instance’s feature
  vector **x** can then be replaced with the vector of its cluster affinities. If there are k
  clusters, then this vector is k dimensional. This is typically much lower dimen‐
  sional than the original feature vector, but it can preserve enough information for
  further processing.
- For anomaly detection (also called outlier detection): any instance that has a low
  affinity to all the clusters is likely to be an anomaly. For example, if you have clus‐
  tered the users of your website based on their behavior, you can detect users with
  unusual behavior, such as an unusual number of requests per second, and so on.
  Anomaly detection is particularly useful in detecting defects in manufacturing, or
  for fraud detection.
- For semi-supervised learning: if you only have a few labels, you could perform
  clustering and propagate the labels to all the instances in the same cluster. This
  can greatly increase the amount of labels available for a subsequent supervised
  learning algorithm, and thus improve its performance.

```
Clustering | 239
```

```
1 “Least square quantization in PCM,” Stuart P. Lloyd. (1982).
```

- For search engines: for example, some search engines let you search for images
  that are similar to a reference image. To build such a system, you would first
  apply a clustering algorithm to all the images in your database: similar images
  would end up in the same cluster. Then when a user provides a reference image,
  all you need to do is to find this image’s cluster using the trained clustering
  model, and you can then simply return all the images from this cluster.
- To segment an image: by clustering pixels according to their color, then replacing
  each pixel’s color with the mean color of its cluster, it is possible to reduce the
  number of different colors in the image considerably. This technique is used in
  many object detection and tracking systems, as it makes it easier to detect the
  contour of each object.

There is no universal definition of what a cluster is: it really depends on the context,
and different algorithms will capture different kinds of clusters. For example, some
algorithms look for instances centered around a particular point, called a centroid.
Others look for continuous regions of densely packed instances: these clusters can
take on any shape. Some algorithms are hierarchical, looking for clusters of clusters.
And the list goes on.

In this section, we will look at two popular clustering algorithms: K-Means and
DBSCAN, and we will show some of their applications, such as non-linear dimen‐
sionality reduction, semi-supervised learning and anomaly detection.

##### K-Means

Consider the unlabeled dataset represented in Figure 9-2: you can clearly see 5 blobs
of instances. The K-Means algorithm is a simple algorithm capable of clustering this
kind of dataset very quickly and efficiently, often in just a few iterations. It was pro‐
posed by Stuart Lloyd at the Bell Labs in 1957 as a technique for pulse-code modula‐
tion, but it was only published outside of the company in 1982, in a paper titled
“Least square quantization in PCM”.^1 By then, in 1965, Edward W. Forgy had pub‐
lished virtually the same algorithm, so K-Means is sometimes referred to as Lloyd-
Forgy.

**240 | Chapter 9: Unsupervised Learning Techniques**

Figure 9-2. An unlabeled dataset composed of five blobs of instances

Let’s train a K-Means clusterer on this dataset. It will try to find each blob’s center and
assign each instance to the closest blob:

```
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)
```

Note that you have to specify the number of clusters k that the algorithm must find.
In this example, it is pretty obvious from looking at the data that k should be set to 5,
but in general it is not that easy. We will discuss this shortly.

Each instance was assigned to one of the 5 clusters. In the context of clustering, an
instance’s label is the index of the cluster that this instance gets assigned to by the
algorithm: this is not to be confused with the class labels in classification (remember
that clustering is an unsupervised learning task). The KMeans instance preserves a
copy of the labels of the instances it was trained on, available via the labels\_ instance
variable:

```
>>> y_pred
array([4, 0, 1, ..., 2, 1, 0], dtype=int32)
>>> y_pred is kmeans.labels_
True
```

We can also take a look at the 5 centroids that the algorithm found:

```
>>> kmeans.cluster_centers_
array([[-2.80389616, 1.80117999],
[ 0.20876306, 2.25551336],
[-2.79290307, 2.79641063],
[-1.46679593, 2.28585348],
[-2.80037642, 1.30082566]])
```

Of course, you can easily assign new instances to the cluster whose centroid is closest:

```
Clustering | 241
```

```
>>> X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
>>> kmeans.predict(X_new)
array([1, 1, 2, 2], dtype=int32)
```

If you plot the cluster’s decision boundaries, you get a Voronoi tessellation (see
Figure 9-3, where each centroid is represented with an X):

Figure 9-3. K-Means decision boundaries (Voronoi tessellation)

The vast majority of the instances were clearly assigned to the appropriate cluster, but
a few instances were probably mislabeled (especially near the boundary between the
top left cluster and the central cluster). Indeed, the K-Means algorithm does not
behave very well when the blobs have very different diameters since all it cares about
when assigning an instance to a cluster is the distance to the centroid.

Instead of assigning each instance to a single cluster, which is called hard clustering, it
can be useful to just give each instance a score per cluster: this is called soft clustering.
For example, the score can be the distance between the instance and the centroid, or
conversely it can be a similarity score (or affinity) such as the Gaussian Radial Basis
Function (introduced in Chapter 5). In the KMeans class, the transform() method
measures the distance from each instance to every centroid:

```
>>> kmeans.transform(X_new)
array([[2.81093633, 0.32995317, 2.9042344 , 1.49439034, 2.88633901],
[5.80730058, 2.80290755, 5.84739223, 4.4759332 , 5.84236351],
[1.21475352, 3.29399768, 0.29040966, 1.69136631, 1.71086031],
[0.72581411, 3.21806371, 0.36159148, 1.54808703, 1.21567622]])
```

In this example, the first instance in X_new is located at a distance of 2.81 from the
first centroid, 0.33 from the second centroid, 2.90 from the third centroid, 1.49 from
the fourth centroid and 2.87 from the fifth centroid. If you have a high-dimensional
dataset and you transform it this way, you end up with a k-dimensional dataset: this
can be a very efficient non-linear dimensionality reduction technique.

**242 | Chapter 9: Unsupervised Learning Techniques**

```
2 This can be proven by pointing out that the mean squared distance between the instances and their closest
centroid can only go down at each step.
```

**The K-Means Algorithm**

So how does the algorithm work? Well it is really quite simple. Suppose you were
given the centroids: you could easily label all the instances in the dataset by assigning
each of them to the cluster whose centroid is closest. Conversely, if you were given all
the instance labels, you could easily locate all the centroids by computing the mean of
the instances for each cluster. But you are given neither the labels nor the centroids,
so how can you proceed? Well, just start by placing the centroids randomly (e.g., by
picking k instances at random and using their locations as centroids). Then label the
instances, update the centroids, label the instances, update the centroids, and so on
until the centroids stop moving. The algorithm is guaranteed to converge in a finite
number of steps (usually quite small), it will not oscillate forever^2. You can see the
algorithm in action in Figure 9-4: the centroids are initialized randomly (top left),
then the instances are labeled (top right), then the centroids are updated (center left),
the instances are relabeled (center right), and so on. As you can see, in just 3 itera‐
tions the algorithm has reached a clustering that seems close to optimal.

Figure 9-4. The K-Means algorithm

```
Clustering | 243
```

```
The computational complexity of the algorithm is generally linear
with regards to the number of instances m, the number of clusters
k and the number of dimensions n. However, this is only true when
the data has a clustering structure. If it does not, then in the worst
case scenario the complexity can increase exponentially with the
number of instances. In practice, however, this rarely happens, and
K-Means is generally one of the fastest clustering algorithms.
```

Unfortunately, although the algorithm is guaranteed to converge, it may not converge
to the right solution (i.e., it may converge to a local optimum): this depends on the
centroid initialization. For example, Figure 9-5 shows two sub-optimal solutions that
the algorithm can converge to if you are not lucky with the random initialization step:

Figure 9-5. Sub-optimal solutions due to unlucky centroid initializations

Let’s look at a few ways you can mitigate this risk by improving the centroid initializa‐
tion.

**Centroid Initialization Methods**

If you happen to know approximately where the centroids should be (e.g., if you ran
another clustering algorithm earlier), then you can set the init hyperparameter to a
NumPy array containing the list of centroids, and set n_init to 1:

```
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)
```

Another solution is to run the algorithm multiple times with different random initial‐
izations and keep the best solution. This is controlled by the n_init hyperparameter:
by default, it is equal to 10, which means that the whole algorithm described earlier
actually runs 10 times when you call fit(), and Scikit-Learn keeps the best solution.
But how exactly does it know which solution is the best? Well of course it uses a per‐
formance metric! It is called the model’s inertia: this is the mean squared distance
between each instance and its closest centroid. It is roughly equal to 223.3 for the
model on the left of Figure 9-5, 237.5 for the model on the right of Figure 9-5, and
211.6 for the model in Figure 9-3. The KMeans class runs the algorithm n_init times
and keeps the model with the lowest inertia: in this example, the model in Figure 9-3
will be selected (unless we are very unlucky with n_init consecutive random initiali‐

**244 | Chapter 9: Unsupervised Learning Techniques**

```
3 “k-means\++: The advantages of careful seeding,” David Arthur and Sergei Vassilvitskii (2006).
4 “Using the Triangle Inequality to Accelerate k-Means,” Charles Elkan (2003).
```

zations). If you are curious, a model’s inertia is accessible via the inertia\_ instance
variable:

```
>>> kmeans.inertia_
211.59853725816856
```

The score() method returns the negative inertia. Why negative? Well, it is because a
predictor’s score() method must always respect the "great is better" rule.

```
>>> kmeans.score(X)
-211.59853725816856
```

An important improvement to the K-Means algorithm, called K-Means+\+, was pro‐
posed in a 2006 paper by David Arthur and Sergei Vassilvitskii:^3 they introduced a
smarter initialization step that tends to select centroids that are distant from one
another, and this makes the K-Means algorithm much less likely to converge to a sub-
optimal solution. They showed that the additional computation required for the
smarter initialization step is well worth it since it makes it possible to drastically
reduce the number of times the algorithm needs to be run to find the optimal solu‐
tion. Here is the K-Means++ initialization algorithm:

- Take one centroid **c** (1), chosen uniformly at random from the dataset.
- Take a new centroid **c** (i), choosing an instance **x** (i) with probability: D𝐀i

```
2
```

```
∑jm= 1D𝐀j
```

```
2
where D( x (i)) is the distance between the instance x (i) and the closest
centroid that was already chosen. This probability distribution ensures that
instances further away from already chosen centroids are much more likely be
selected as centroids.
```

- Repeat the previous step until all k centroids have been chosen.

The KMeans class actually uses this initialization method by default. If you want to
force it to use the original method (i.e., picking k instances randomly to define the
initial centroids), then you can set the init hyperparameter to "random". You will
rarely need to do this.

**Accelerated K-Means and Mini-batch K-Means**

Another important improvement to the K-Means algorithm was proposed in a 2003
paper by Charles Elkan.^4 It considerably accelerates the algorithm by avoiding many
unnecessary distance calculations: this is achieved by exploiting the triangle inequal‐

```
Clustering | 245
```

```
5 The triangle inequality is AC ≤ AB + BC where A, B and C are three points, and AB, AC and BC are the
distances between these points.
6 “Web-Scale K-Means Clustering,” David Sculley (2010).
```

ity (i.e., the straight line is always the shortest^5 ) and by keeping track of lower and
upper bounds for distances between instances and centroids. This is the algorithm
used by default by the KMeans class (but you can force it to use the original algorithm
by setting the algorithm hyperparameter to "full", although you probably will
never need to).

Yet another important variant of the K-Means algorithm was proposed in a 2010
paper by David Sculley.^6 Instead of using the full dataset at each iteration, the algo‐
rithm is capable of using mini-batches, moving the centroids just slightly at each iter‐
ation. This speeds up the algorithm typically by a factor of 3 or 4 and makes it
possible to cluster huge datasets that do not fit in memory. Scikit-Learn implements
this algorithm in the MiniBatchKMeans class. You can just use this class like the
KMeans class:

```
from sklearn.cluster import MiniBatchKMeans
```

```
minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
```

If the dataset does not fit in memory, the simplest option is to use the memmap class, as
we did for incremental PCA in Chapter 8. Alternatively, you can pass one mini-batch
at a time to the partial_fit() method, but this will require much more work, since
you will need to perform multiple initializations and select the best one yourself (see
the notebook for an example).

Although the Mini-batch K-Means algorithm is much faster than the regular K-
Means algorithm, its inertia is generally slightly worse, especially as the number of
clusters increases. You can see this in Figure 9-6: the plot on the left compares the
inertias of Mini-batch K-Means and regular K-Means models trained on the previous
dataset using various numbers of clusters k. The difference between the two curves
remains fairly constant, but this difference becomes more and more significant as k
increases, since the inertia becomes smaller and smaller. However, in the plot on the
right, you can see that Mini-batch K-Means is much faster than regular K-Means, and
this difference increases with k.

**246 | Chapter 9: Unsupervised Learning Techniques**

Figure 9-6. Mini-batch K-Means vs K-Means: worse inertia as k increases (left) but
much faster (right)

**Finding the Optimal Number of Clusters**

So far, we have set the number of clusters k to 5 because it was obvious by looking at
the data that this is the correct number of clusters. But in general, it will not be so
easy to know how to set k, and the result might be quite bad if you set it to the wrong
value. For example, as you can see in Figure 9-7, setting k to 3 or 8 results in fairly
bad models:

Figure 9-7. Bad choices for the number of clusters

You might be thinking that we could just pick the model with the lowest inertia,
right? Unfortunately, it is not that simple. The inertia for k=3 is 653.2, which is much
higher than for k=5 (which was 211.6), but with k=8, the inertia is just 119.1. The
inertia is not a good performance metric when trying to choose k since it keeps get‐
ting lower as we increase k. Indeed, the more clusters there are, the closer each
instance will be to its closest centroid, and therefore the lower the inertia will be. Let’s
plot the inertia as a function of k (see Figure 9-8):

```
Clustering | 247
```

Figure 9-8. Selecting the number of clusters k using the “elbow rule”

As you can see, the inertia drops very quickly as we increase k up to 4, but then it
decreases much more slowly as we keep increasing k. This curve has roughly the
shape of an arm, and there is an “elbow” at k=4 so if we did not know better, it would
be a good choice: any lower value would be dramatic, while any higher value would
not help much, and we might just be splitting perfectly good clusters in half for no
good reason.

This technique for choosing the best value for the number of clusters is rather coarse.
A more precise approach (but also more computationally expensive) is to use the sil‐
houette score, which is the mean silhouette coefficient over all the instances. An instan‐
ce’s silhouette coefficient is equal to (b – a) / max(a, b) where a is the mean distance
to the other instances in the same cluster (it is the mean intra-cluster distance), and b
is the mean nearest-cluster distance, that is the mean distance to the instances of the
next closest cluster (defined as the one that minimizes b, excluding the instance’s own
cluster). The silhouette coefficient can vary between -1 and +1: a coefficient close to
+1 means that the instance is well inside its own cluster and far from other clusters,
while a coefficient close to 0 means that it is close to a cluster boundary, and finally a
coefficient close to -1 means that the instance may have been assigned to the wrong
cluster. To compute the silhouette score, you can use Scikit-Learn’s silhou
ette_score() function, giving it all the instances in the dataset, and the labels they
were assigned:

```
>>> from sklearn.metrics import silhouette_score
>>> silhouette_score(X, kmeans.labels_)
0.655517642572828
```

Let’s compare the silhouette scores for different numbers of clusters (see Figure 9-9):

**248 | Chapter 9: Unsupervised Learning Techniques**

Figure 9-9. Selecting the number of clusters k using the silhouette score

As you can see, this visualization is much richer than the previous one: in particular,
although it confirms that k=4 is a very good choice, it also underlines the fact that
k=5 is quite good as well, and much better than k=6 or 7. This was not visible when
comparing inertias.

An even more informative visualization is obtained when you plot every instance’s
silhouette coefficient, sorted by the cluster they are assigned to and by the value of the
coefficient. This is called a silhouette diagram (see Figure 9-10):

Figure 9-10. Silouhette analysis: comparing the silhouette diagrams for various values of
k

The vertical dashed lines represent the silhouette score for each number of clusters.
When most of the instances in a cluster have a lower coefficient than this score (i.e., if
many of the instances stop short of the dashed line, ending to the left of it), then the
cluster is rather bad since this means its instances are much too close to other clus‐

```
Clustering | 249
```

ters. We can see that when k=3 and when k=6, we get bad clusters. But when k=4 or
k=5, the clusters look pretty good – most instances extend beyond the dashed line, to
the right and closer to 1.0. When k=4, the cluster at index 1 (the third from the top),
is rather big, while when k=5, all clusters have similar sizes, so even though the over‐
all silhouette score from k=4 is slightly greater than for k=5, it seems like a good idea
to use k=5 to get clusters of similar sizes.

##### Limits of K-Means

Despite its many merits, most notably being fast and scalable, K-Means is not perfect.
As we saw, it is necessary to run the algorithm several times to avoid sub-optimal sol‐
utions, plus you need to specify the number of clusters, which can be quite a hassle.
Moreover, K-Means does not behave very well when the clusters have varying sizes,
different densities, or non-spherical shapes. For example, Figure 9-11 shows how K-
Means clusters a dataset containing three ellipsoidal clusters of different dimensions,
densities and orientations:

Figure 9-11. K-Means fails to cluster these ellipsoidal blobs properly

As you can see, neither of these solutions are any good. The solution on the left is
better, but it still chops off 25% of the middle cluster and assigns it to the cluster on
the right. The solution on the right is just terrible, even though its inertia is lower. So
depending on the data, different clustering algorithms may perform better. For exam‐
ple, on these types of elliptical clusters, Gaussian mixture models work great.

```
It is important to scale the input features before you run K-Means,
or else the clusters may be very stretched, and K-Means will per‐
form poorly. Scaling the features does not guarantee that all the
clusters will be nice and spherical, but it generally improves things.
```

Now let’s look at a few ways we can benefit from clustering. We will use K-Means, but
feel free to experiment with other clustering algorithms.

**250 | Chapter 9: Unsupervised Learning Techniques**

##### Using clustering for image segmentation

Image segmentation is the task of partitioning an image into multiple segments. In
semantic segmentation, all pixels that are part of the same object type get assigned to
the same segment. For example, in a self-driving car’s vision system, all pixels that are
part of a pedestrian’s image might be assigned to the “pedestrian” segment (there
would just be one segment containing all the pedestrians). In instance segmentation,
all pixels that are part of the same individual object are assigned to the same segment.
In this case there would be a different segment for each pedestrian. The state of the
art in semantic or instance segmentation today is achieved using complex architec‐
tures based on convolutional neural networks (see Chapter 14). Here, we are going to
do something much simpler: color segmentation. We will simply assign pixels to the
same segment if they have a similar color. In some applications, this may be sufficient,
for example if you want to analyze satellite images to measure how much total forest
area there is in a region, color segmentation may be just fine.

First, let’s load the image (see the upper left image in Figure 9-12) using Matplotlib’s
imread() function:

```
>>> from matplotlib.image import imread # you could also use `imageio.imread()`
>>> image = imread(os.path.join("images","clustering","ladybug.png"))
>>> image.shape
(533, 800, 3)
```

The image is represented as a 3D array: the first dimension’s size is the height, the
second is the width, and the third is the number of color channels, in this case red,
green and blue (RGB). In other words, for each pixel there is a 3D vector containing
the intensities of red, green and blue, each between 0.0 and 1.0 (or between 0 and 255
if you use imageio.imread()). Some images may have less channels, such as gray‐
scale images (one channel), or more channels, such as images with an additional
alpha channel for transparency, or satellite images which often contain channels for
many light frequencies (e.g., infrared). The following code reshapes the array to get a
long list of RGB colors, then it clusters these colors using K-Means. For example, it
may identify a color cluster for all shades of green. Next, for each color (e.g., dark
green), it looks for the mean color of the pixel’s color cluster. For example, all shades
of green may be replaced with the same light green color (assuming the mean color of
the green cluster is light green). Finally it reshapes this long list of colors to get the
same shape as the original image. And we’re done!

```
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
```

This outputs the image shown in the upper right of Figure 9-12. You can experiment
with various numbers of clusters, as shown in the figure. When you use less than 8
clusters, notice that the ladybug’s flashy red color fails to get a cluster of its own: it

```
Clustering | 251
```

gets merged with colors from the environment. This is due to the fact that the lady‐
bug is quite small, much smaller than the rest of the image, so even though its color is
flashy, K-Means fails to dedicate a cluster to it: as mentioned earlier, K-Means prefers
clusters of similar sizes.

Figure 9-12. Image segmentation using K-Means with various numbers of color clusters

That was not too hard, was it? Now let’s look at another application of clustering: pre‐
processing.

##### Using Clustering for Preprocessing

Clustering can be an efficient approach to dimensionality reduction, in particular as a
preprocessing step before a supervised learning algorithm. For example, let’s tackle
the digits dataset which is a simple MNIST-like dataset containing 1,797 grayscale 8×8
images representing digits 0 to 9. First, let’s load the dataset:

```
from sklearn.datasets import load_digits
```

```
X_digits, y_digits = load_digits(return_X_y=True)
```

Now, let’s split it into a training set and a test set:

```
from sklearn.model_selection import train_test_split
```

```
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)
```

Next, let’s fit a Logistic Regression model:

```
from sklearn.linear_model import LogisticRegression
```

```
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

Let’s evaluate its accuracy on the test set:

```
>>> log_reg.score(X_test, y_test)
0.9666666666666667
```

**252 | Chapter 9: Unsupervised Learning Techniques**

Okay, that’s our baseline: 96.7% accuracy. Let’s see if we can do better by using K-
Means as a preprocessing step. We will create a pipeline that will first cluster the
training set into 50 clusters and replace the images with their distances to these 50
clusters, then apply a logistic regression model.

```
Although it is tempting to define the number of clusters to 10,
since there are 10 different digits, it is unlikely to perform well,
because there are several different ways to write each digit.
```

```
from sklearn.pipeline import Pipeline
```

```
pipeline = Pipeline([
("kmeans", KMeans(n_clusters=50)),
("log_reg", LogisticRegression()),
])
pipeline.fit(X_train, y_train)
```

Now let’s evaluate this classification pipeline:

```
>>> pipeline.score(X_test, y_test)
0.9822222222222222
```

How about that? We almost divided the error rate by a factor of 2!

But we chose the number of clusters k completely arbitrarily, we can surely do better.
Since K-Means is just a preprocessing step in a classification pipeline, finding a good
value for k is much simpler than earlier: there’s no need to perform silhouette analysis
or minimize the inertia, the best value of k is simply the one that results in the best
classification performance during cross-validation. Let’s use GridSearchCV to find the
optimal number of clusters:

```
from sklearn.model_selection import GridSearchCV
```

```
param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
```

Let’s look at best value for k, and the performance of the resulting pipeline:

```
>>> grid_clf.best_params_
{'kmeans__n_clusters': 90}
>>> grid_clf.score(X_test, y_test)
0.9844444444444445
```

With k=90 clusters, we get a small accuracy boost, reaching 98.4% accuracy on the
test set. Cool!

```
Clustering | 253
```

##### Using Clustering for Semi-Supervised Learning

Another use case for clustering is in semi-supervised learning, when we have plenty
of unlabeled instances and very few labeled instances. Let’s train a logistic regression
model on a sample of 50 labeled instances from the digits dataset:

```
n_labeled = 50
log_reg = LogisticRegression()
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
```

What is the performance of this model on the test set?

```
>>> log_reg.score(X_test, y_test)
0.8266666666666667
```

The accuracy is just 82.7%: it should come as no surprise that this is much lower than
earlier, when we trained the model on the full training set. Let’s see how we can do
better. First, let’s cluster the training set into 50 clusters, then for each cluster let’s find
the image closest to the centroid. We will call these images the representative images:

```
k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]
```

Figure 9-13 shows these 50 representative images:

Figure 9-13. Fifty representative digit images (one per cluster)

Now let’s look at each image and manually label it:

```
y_representative_digits = np.array([4, 8, 0, 6, 8, 3, ..., 7, 6, 2, 3, 1, 1])
```

Now we have a dataset with just 50 labeled instances, but instead of being completely
random instances, each of them is a representative image of its cluster. Let’s see if the
performance is any better:

```
>>> log_reg = LogisticRegression()
>>> log_reg.fit(X_representative_digits, y_representative_digits)
>>> log_reg.score(X_test, y_test)
0.9244444444444444
```

Wow! We jumped from 82.7% accuracy to 92.4%, although we are still only training
the model on 50 instances. Since it is often costly and painful to label instances, espe‐

**254 | Chapter 9: Unsupervised Learning Techniques**

cially when it has to be done manually by experts, it is a good idea to label representa‐
tive instances rather than just random instances.

But perhaps we can go one step further: what if we propagated the labels to all the
other instances in the same cluster? This is called label propagation:

```
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
```

Now let’s train the model again and look at its performance:

```
>>> log_reg = LogisticRegression()
>>> log_reg.fit(X_train, y_train_propagated)
>>> log_reg.score(X_test, y_test)
0.9288888888888889
```

We got a tiny little accuracy boost. Better than nothing, but not astounding. The
problem is that we propagated each representative instance’s label to all the instances
in the same cluster, including the instances located close to the cluster boundaries,
which are more likely to be mislabeled. Let’s see what happens if we only propagate
the labels to the 20% of the instances that are closest to the centroids:

```
percentile_closest = 20
```

```
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
in_cluster = (kmeans.labels_ == i)
cluster_dist = X_cluster_dist[in_cluster]
cutoff_distance = np.percentile(cluster_dist, percentile_closest)
above_cutoff = (X_cluster_dist > cutoff_distance)
X_cluster_dist[in_cluster & above_cutoff] = -1
```

```
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]
```

Now let’s train the model again on this partially propagated dataset:

```
>>> log_reg = LogisticRegression()
>>> log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
>>> log_reg.score(X_test, y_test)
0.9422222222222222
```

Nice! With just 50 labeled instances (only 5 examples per class on average!), we got
94.2% performance, which is pretty close to the performance of logistic regression on
the fully labeled digits dataset (which was 96.7%). This is because the propagated
labels are actually pretty good, their accuracy is very close to 99%:

```
Clustering | 255
```

```
>>> np.mean(y_train_partially_propagated == y_train[partially_propagated])
0.9896907216494846
```

```
Active Learning
To continue improving your model and your training set, the next step could be to do
a few rounds of active learning: this is when a human expert interacts with the learn‐
ing algorithm, providing labels when the algorithm needs them. There are many dif‐
ferent strategies for active learning, but one of the most common ones is called
uncertainty sampling:
```

- The model is trained on the labeled instances gathered so far, and this model is
  used to make predictions on all the unlabeled instances.
- The instances for which the model is most uncertain (i.e., when its estimated
  probability is lowest) must be labeled by the expert.
- Then you just iterate this process again and again, until the performance
  improvement stops being worth the labeling effort.

```
Other strategies include labeling the instances that would result in the largest model
change, or the largest drop in the model’s validation error, or the instances that differ‐
ent models disagree on (e.g., an SVM, a Random Forest, and so on).
```

Before we move on to Gaussian mixture models, let’s take a look at DBSCAN,
another popular clustering algorithm that illustrates a very different approach based
on local density estimation. This approach allows the algorithm to identify clusters of
arbitrary shapes.

##### DBSCAN

This algorithm defines clusters as continuous regions of high density. It is actually
quite simple:

- For each instance, the algorithm counts how many instances are located within a
  small distance ε (epsilon) from it. This region is called the instance’s ε-
  neighborhood.
- If an instance has at least min_samples instances in its ε-neighborhood (includ‐
  ing itself ), then it is considered a core instance. In other words, core instances are
  those that are located in dense regions.
- All instances in the neighborhood of a core instance belong to the same cluster.
  This may include other core instances, therefore a long sequence of neighboring
  core instances forms a single cluster.

**256 | Chapter 9: Unsupervised Learning Techniques**

- Any instance that is not a core instance and does not have one in its neighbor‐
  hood is considered an anomaly.

This algorithm works well if all the clusters are dense enough, and they are well sepa‐
rated by low-density regions. The DBSCAN class in Scikit-Learn is as simple to use as
you might expect. Let’s test it on the moons dataset, introduced in Chapter 5:

```
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
```

```
X, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```

The labels of all the instances are now available in the labels\_ instance variable:

```
>>> dbscan.labels_
array([ 0, 2, -1, -1, 1, 0, 0, 0, ..., 3, 2, 3, 3, 4, 2, 6, 3])
```

Notice that some instances have a cluster index equal to -1: this means that they are
considered as anomalies by the algorithm. The indices of the core instances are avail‐
able in the core*sample_indices* instance variable, and the core instances them‐
selves are available in the components\_ instance variable:

```
>>> len(dbscan.core_sample_indices_)
808
>>> dbscan.core_sample_indices_
array([ 0, 4, 5, 6, 7, 8, 10, 11, ..., 992, 993, 995, 997, 998, 999])
>>> dbscan.components_
array([[-0.02137124, 0.40618608],
[-0.84192557, 0.53058695],
...
[-0.94355873, 0.3278936 ],
[ 0.79419406, 0.60777171]])
```

This clustering is represented in the left plot of Figure 9-14. As you can see, it identi‐
fied quite a lot of anomalies, plus 7 different clusters. How disappointing! Fortunately,
if we widen each instance’s neighborhood by increasing eps to 0.2, we get the cluster‐
ing on the right, which looks perfect. Let’s continue with this model.

Figure 9-14. DBSCAN clustering using two different neighborhood radiuses

```
Clustering | 257
```

Somewhat surprisingly, the DBSCAN class does not have a predict() method,
although it has a fit_predict() method. In other words, it cannot predict which
cluster a new instance belongs to. The rationale for this decision is that several classi‐
fication algorithms could make sense here, and it is easy enough to train one, for
example a KNeighborsClassifier:

```
from sklearn.neighbors import KNeighborsClassifier
```

```
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
```

Now, given a few new instances, we can predict which cluster they most likely belong
to, and even estimate a probability for each cluster. Note that we only trained them on
the core instances, but we could also have chosen to train them on all the instances,
or all but the anomalies: this choice depends on the final task.

```
>>> X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
>>> knn.predict(X_new)
array([1, 0, 1, 0])
>>> knn.predict_proba(X_new)
array([[0.18, 0.82],
[1. , 0. ],
[0.12, 0.88],
[1. , 0. ]])
```

The decision boundary is represented on Figure 9-15 (the crosses represent the 4
instances in X_new). Notice that since there is no anomaly in the KNN’s training set,
the classifier always chooses a cluster, even when that cluster is far away. However, it
is fairly straightforward to introduce a maximum distance, in which case the two
instances that are far away from both clusters are classified as anomalies. To do this,
we can use the kneighbors() method of the KNeighborsClassifier: given a set of
instances, it returns the distances and the indices of the k nearest neighbors in the
training set (two matrices, each with k columns):

```
>>> y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
>>> y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
>>> y_pred[y_dist > 0.2] = -1
>>> y_pred.ravel()
array([-1, 0, 1, -1])
```

**258 | Chapter 9: Unsupervised Learning Techniques**

Figure 9-15. cluster_classification_diagram

In short, DBSCAN is a very simple yet powerful algorithm, capable of identifying any
number of clusters, of any shape, it is robust to outliers, and it has just two hyper‐
parameters (eps and min_samples). However, if the density varies significantly across
the clusters, it can be impossible for it to capture all the clusters properly. Moreover,
its computational complexity is roughly O(m log m), making it pretty close to linear
with regards to the number of instances. However, Scikit-Learn’s implementation can
require up to O(m^2 ) memory if eps is large.

##### Other Clustering Algorithms

Scikit-Learn implements several more clustering algorithms that you should take a
look at. We cannot cover them all in detail here, but here is a brief overview:

- Agglomerative clustering: a hierarchy of clusters is built from the bottom up.
  Think of many tiny bubbles floating on water and gradually attaching to each
  other until there’s just one big group of bubbles. Similarly, at each iteration
  agglomerative clustering connects the nearest pair of clusters (starting with indi‐
  vidual instances). If you draw a tree with a branch for every pair of clusters that
  merged, you get a binary tree of clusters, where the leaves are the individual
  instances. This approach scales very well to large numbers of instances or clus‐
  ters, it can capture clusters of various shapes, it produces a flexible and informa‐
  tive cluster tree instead of forcing you to choose a particular cluster scale, and it
  can be used with any pairwise distance. It can scale nicely to large numbers of
  instances if you provide a connectivity matrix. This is a sparse m by m matrix
  that indicates which pairs of instances are neighbors (e.g., returned by
  sklearn.neighbors.kneighbors_graph()). Without a connectivity matrix, the
  algorithm does not scale well to large datasets.
- Birch: this algorithm was designed specifically for very large datasets, and it can
  be faster than batch K-Means, with similar results, as long as the number of fea‐
  tures is not too large (<20). It builds a tree structure during training containing

```
Clustering | 259
```

```
just enough information to quickly assign each new instance to a cluster, without
having to store all the instances in the tree: this allows it to use limited memory,
while handle huge datasets.
```

- Mean-shift: this algorithm starts by placing a circle centered on each instance,
  then for each circle it computes the mean of all the instances located within it,
  and it shifts the circle so that it is centered on the mean. Next, it iterates this
  mean-shift step until all the circles stop moving (i.e., until each of them is cen‐
  tered on the mean of the instances it contains). This algorithm shifts the circles
  in the direction of higher density, until each of them has found a local density
  maximum. Finally, all the instances whose circles have settled in the same place
  (or close enough) are assigned to the same cluster. This has some of the same fea‐
  tures as DBSCAN, in particular it can find any number of clusters of any shape, it
  has just one hyperparameter (the radius of the circles, called the bandwidth) and
  it relies on local density estimation. However, it tends to chop clusters into pieces
  when they have internal density variations. Unfortunately, its computational
  complexity is O(m^2 ), so it is not suited for large datasets.
- Affinity propagation: this algorithm uses a voting system, where instances vote for
  similar instances to be their representatives, and once the algorithm converges,
  each representative and its voters form a cluster. This algorithm can detect any
  number of clusters of different sizes. Unfortunately, this algorithm has a compu‐
  tational complexity of O(m^2 ), so it is not suited for large datasets.
- Spectral clustering: this algorithm takes a similarity matrix between the instances
  and creates a low-dimensional embedding from it (i.e., it reduces its dimension‐
  ality), then it uses another clustering algorithm in this low-dimensional space
  (Scikit-Learn’s implementation uses K-Means). Spectral clustering can capture
  complex cluster structures, and it can also be used to cut graphs (e.g., to identify
  clusters of friends on a social network), however it does not scale well to large
  number of instances, and it does not behave well when the clusters have very dif‐
  ferent sizes.

Now let’s dive into Gaussian mixture models, which can be used for density estima‐
tion, clustering and anomaly detection.

#### Gaussian Mixtures

A Gaussian mixture model (GMM) is a probabilistic model that assumes that the
instances were generated from a mixture of several Gaussian distributions whose
parameters are unknown. All the instances generated from a single Gaussian distri‐
bution form a cluster that typically looks like an ellipsoid. Each cluster can have a dif‐
ferent ellipsoidal shape, size, density and orientation, just like in Figure 9-11. When
you observe an instance, you know it was generated from one of the Gaussian distri‐

**260 | Chapter 9: Unsupervised Learning Techniques**

```
7 Phi (φ or φ) is the 21st letter of the Greek alphabet.
8 Most of these notations are standard, but a few additional notations were taken from the Wikipedia article on
plate notation.
```

butions, but you are not told which one, and you do not know what the parameters of
these distributions are.

There are several GMM variants: in the simplest variant, implemented in the Gaus
sianMixture class, you must know in advance the number k of Gaussian distribu‐
tions. The dataset **X** is assumed to have been generated through the following
probabilistic process:

- For each instance, a cluster is picked randomly among k clusters. The probability
  of choosing the jth cluster is defined by the cluster’s weight φ(j).^7 The index of the
  cluster chosen for the ith instance is noted z(i).
- If z(i)=j, meaning the ith instance has been assigned to the jth cluster, the location
  **x** (i) of this instance is sampled randomly from the Gaussian distribution with
  mean **μ** (j) and covariance matrix **Σ** (j). This is noted 𝐀i ∼ 𝐀 μj,Σj.

This generative process can be represented as a graphical model (see Figure 9-16).
This is a graph which represents the structure of the conditional dependencies
between random variables.

Figure 9-16. Gaussian mixture model

Here is how to interpret it:^8

- The circles represent random variables.
- The squares represent fixed values (i.e., parameters of the model).

```
Gaussian Mixtures | 261
```

- The large rectangles are called plates: they indicate that their content is repeated
  several times.
- The number indicated at the bottom right hand side of each plate indicates how
  many times its content is repeated, so there are m random variables z(i) (from z(1)
  to z(m)) and m random variables **x** (i), and k means **μ** (j) and k covariance matrices
  **Σ** (j), but just one weight vector **φ** (containing all the weights φ(1) to φ(k)).
- Each variable z(i) is drawn from the categorical distribution with weights **φ**. Each
  variable **x** (i) is drawn from the normal distribution with the mean and covariance
  matrix defined by its cluster z(i).
- The solid arrows represent conditional dependencies. For example, the probabil‐
  ity distribution for each random variable z(i) depends on the weight vector **φ**.
  Note that when an arrow crosses a plate boundary, it means that it applies to all
  the repetitions of that plate, so for example the weight vector **φ** conditions the
  probability distributions of all the random variables **x** (1) to **x** (m).
- The squiggly arrow from z(i) to **x** (i) represents a switch: depending on the value of
  z(i), the instance **x** (i) will be sampled from a different Gaussian distribution. For
  example, if z(i)=j, then 𝐀i ∼ 𝐀 μj,Σj.
- Shaded nodes indicate that the value is known, so in this case only the random
  variables **x** (i) have known values: they are called observed variables. The unknown
  random variables z(i) are called latent variables.

So what can you do with such a model? Well, given the dataset **X** , you typically want
to start by estimating the weights **φ** and all the distribution parameters **μ** (1) to **μ** (k) and
**Σ** (1) to **Σ** (k). Scikit-Learn’s GaussianMixture class makes this trivial:

```
from sklearn.mixture import GaussianMixture
```

```
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
```

Let’s look at the parameters that the algorithm estimated:

```
>>> gm.weights_
array([0.20965228, 0.4000662 , 0.39028152])
>>> gm.means_
array([[ 3.39909717, 1.05933727],
[-1.40763984, 1.42710194],
[ 0.05135313, 0.07524095]])
>>> gm.covariances_
array([[[ 1.14807234, -0.03270354],
[-0.03270354, 0.95496237]],
```

```
[[ 0.63478101, 0.72969804],
[ 0.72969804, 1.1609872 ]],
```

**262 | Chapter 9: Unsupervised Learning Techniques**

```
[[ 0.68809572, 0.79608475],
[ 0.79608475, 1.21234145]]])
```

Great, it worked fine! Indeed, the weights that were used to generate the data were
0.2, 0.4 and 0.4, and similarly, the means and covariance matrices were very close to
those found by the algorithm. But how? This class relies on the Expectation-
Maximization (EM) algorithm, which has many similarities with the K-Means algo‐
rithm: it also initializes the cluster parameters randomly, then it repeats two steps
until convergence, first assigning instances to clusters (this is called the expectation
step) then updating the clusters (this is called the maximization step). Sounds famil‐
iar? Indeed, in the context of clustering you can think of EM as a generalization of K-
Means which not only finds the cluster centers ( **μ** (1) to **μ** (k)), but also their size, shape
and orientation ( **Σ** (1) to **Σ** (k)), as well as their relative weights (φ(1) to φ(k)). Unlike K-
Means, EM uses soft cluster assignments rather than hard assignments: for each
instance during the expectation step, the algorithm estimates the probability that it
belongs to each cluster (based on the current cluster parameters). Then, during the
maximization step, each cluster is updated using all the instances in the dataset, with
each instance weighted by the estimated probability that it belongs to that cluster.
These probabilities are called the responsibilities of the clusters for the instances. Dur‐
ing the maximization step, each cluster’s update will mostly be impacted by the
instances it is most responsible for.

```
Unfortunately, just like K-Means, EM can end up converging to
poor solutions, so it needs to be run several times, keeping only the
best solution. This is why we set n_init to 10. Be careful: by default
n_init is only set to 1.
```

You can check whether or not the algorithm converged and how many iterations it
took:

```
>>> gm.converged_
True
>>> gm.n_iter_
3
```

Okay, now that you have an estimate of the location, size, shape, orientation and rela‐
tive weight of each cluster, the model can easily assign each instance to the most likely
cluster (hard clustering) or estimate the probability that it belongs to a particular
cluster (soft clustering). For this, just use the predict() method for hard clustering,
or the predict_proba() method for soft clustering:

```
>>> gm.predict(X)
array([2, 2, 1, ..., 0, 0, 0])
>>> gm.predict_proba(X)
array([[2.32389467e-02, 6.77397850e-07, 9.76760376e-01],
[1.64685609e-02, 6.75361303e-04, 9.82856078e-01],
```

```
Gaussian Mixtures | 263
```

```
[2.01535333e-06, 9.99923053e-01, 7.49319577e-05],
...,
[9.99999571e-01, 2.13946075e-26, 4.28788333e-07],
[1.00000000e+00, 1.46454409e-41, 5.12459171e-16],
[1.00000000e+00, 8.02006365e-41, 2.27626238e-15]])
```

It is a generative model, meaning you can actually sample new instances from it (note
that they are ordered by cluster index):

```
>>> X_new, y_new = gm.sample(6)
>>> X_new
array([[ 2.95400315, 2.63680992],
[-1.16654575, 1.62792705],
[-1.39477712, -1.48511338],
[ 0.27221525, 0.690366 ],
[ 0.54095936, 0.48591934],
[ 0.38064009, -0.56240465]])
```

```
>>> y_new
array([0, 1, 2, 2, 2, 2])
```

It is also possible to estimate the density of the model at any given location. This is
achieved using the score_samples() method: for each instance it is given, this
method estimates the log of the probability density function (PDF) at that location.
The greater the score, the higher the density:

```
>>> gm.score_samples(X)
array([-2.60782346, -3.57106041, -3.33003479, ..., -3.51352783,
-4.39802535, -3.80743859])
```

If you compute the exponential of these scores, you get the value of the PDF at the
location of the given instances. These are not probabilities, but probability densities:
they can take on any positive value, not just between 0 and 1. To estimate the proba‐
bility that an instance will fall within a particular region, you would have to integrate
the PDF over that region (if you do so over the entire space of possible instance loca‐
tions, the result will be 1).

Figure 9-17 shows the cluster means, the decision boundaries (dashed lines), and the
density contours of this model:

**264 | Chapter 9: Unsupervised Learning Techniques**

Figure 9-17. Cluster means, decision boundaries and density contours of a trained Gaus‐
sian mixture model

Nice! The algorithm clearly found an excellent solution. Of course, we made its task
easy by actually generating the data using a set of 2D Gaussian distributions (unfortu‐
nately, real life data is not always so Gaussian and low-dimensional), and we also gave
the algorithm the correct number of clusters. When there are many dimensions, or
many clusters, or few instances, EM can struggle to converge to the optimal solution.
You might need to reduce the difficulty of the task by limiting the number of parame‐
ters that the algorithm has to learn: one way to do this is to limit the range of shapes
and orientations that the clusters can have. This can be achieved by imposing con‐
straints on the covariance matrices. To do this, just set the covariance_type hyper‐
parameter to one of the following values:

- "spherical": all clusters must be spherical, but they can have different diameters
  (i.e., different variances).
- "diag": clusters can take on any ellipsoidal shape of any size, but the ellipsoid’s
  axes must be parallel to the coordinate axes (i.e., the covariance matrices must be
  diagonal).
- "tied": all clusters must have the same ellipsoidal shape, size and orientation
  (i.e., all clusters share the same covariance matrix).

By default, covariance_type is equal to "full", which means that each cluster can
take on any shape, size and orientation (it has its own unconstrained covariance
matrix). Figure 9-18 plots the solutions found by the EM algorithm when cova
riance_type is set to "tied" or "spherical“.

```
Gaussian Mixtures | 265
```

Figure 9-18. covariance_type_diagram

```
The computational complexity of training a GaussianMixture
model depends on the number of instances m, the number of
dimensions n, the number of clusters k, and the constraints on the
covariance matrices. If covariance_type is "spherical or "diag",
it is O(kmn), assuming the data has a clustering structure. If cova
riance_type is "tied" or "full", it is O(kmn^2 + kn^3 ), so it will not
scale to large numbers of features.
```

Gaussian mixture models can also be used for anomaly detection. Let’s see how.

##### Anomaly Detection using Gaussian Mixtures

Anomaly detection (also called outlier detection) is the task of detecting instances that
deviate strongly from the norm. These instances are of course called anomalies or
outliers, while the normal instances are called inliers. Anomaly detection is very use‐
ful in a wide variety of applications, for example in fraud detection, or for detecting
defective products in manufacturing, or to remove outliers from a dataset before
training another model, which can significantly improve the performance of the
resulting model.

Using a Gaussian mixture model for anomaly detection is quite simple: any instance
located in a low-density region can be considered an anomaly. You must define what
density threshold you want to use. For example, in a manufacturing company that
tries to detect defective products, the ratio of defective products is usually well-
known. Say it is equal to 4%, then you can set the density threshold to be the value
that results in having 4% of the instances located in areas below that threshold den‐
sity. If you notice that you get too many false positives (i.e., perfectly good products
that are flagged as defective), you can lower the threshold. Conversely, if you have too
many false negatives (i.e., defective products that the system does not flag as defec‐
tive), you can increase the threshold. This is the usual precision/recall tradeoff (see
Chapter 3). Here is how you would identify the outliers using the 4th percentile low‐

**266 | Chapter 9: Unsupervised Learning Techniques**

est density as the threshold (i.e., approximately 4% of the instances will be flagged as
anomalies):

```
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
```

These anomalies are represented as stars on Figure 9-19:

Figure 9-19. Anomaly detection using a Gaussian mixture model

A closely related task is novelty detection: it differs from anomaly detection in that the
algorithm is assumed to be trained on a “clean” dataset, uncontaminated by outliers,
whereas anomaly detection does not make this assumption. Indeed, outlier detection
is often precisely used to clean up a dataset.

```
Gaussian mixture models try to fit all the data, including the outli‐
ers, so if you have too many of them, this will bias the model’s view
of “normality”: some outliers may wrongly be considered as nor‐
mal. If this happens, you can try to fit the model once, use it to
detect and remove the most extreme outliers, then fit the model
again on the cleaned up dataset. Another approach is to use robust
covariance estimation methods (see the EllipticEnvelope class).
```

Just like K-Means, the GaussianMixture algorithm requires you to specify the num‐
ber of clusters. So how can you find it?

##### Selecting the Number of Clusters

With K-Means, you could use the inertia or the silhouette score to select the appro‐
priate number of clusters, but with Gaussian mixtures, it is not possible to use these
metrics because they are not reliable when the clusters are not spherical or have dif‐
ferent sizes. Instead, you can try to find the model that minimizes a theoretical infor‐

```
Gaussian Mixtures | 267
```

mation criterion such as the Bayesian information criterion (BIC) or the Akaike
information criterion (AIC), defined in Equation 9-1.

```
Equation 9-1. Bayesian information criterion (BIC) and Akaike information
criterion (AIC)
```

```
BIC= log mp− 2 log L
```

```
AIC= 2 p− 2 log L
```

- m is the number of instances, as always.
- p is the number of parameters learned by the model.
- L is the maximized value of the likelihood function of the model.

Both the BIC and the AIC penalize models that have more parameters to learn (e.g.,
more clusters), and reward models that fit the data well. They often end up selecting
the same model, but when they differ, the model selected by the BIC tends to be sim‐
pler (fewer parameters) than the one selected by the AIC, but it does not fit the data
quite as well (this is especially true for larger datasets).

```
Likelihood function
The terms “probability” and “likelihood” are often used interchangeably in the
English language, but they have very different meanings in statistics: given a statistical
model with some parameters θ , the word “probability” is used to describe how plausi‐
ble a future outcome x is (knowing the parameter values θ ), while the word “likeli‐
hood” is used to describe how plausible a particular set of parameter values θ are,
after the outcome x is known.
Consider a one-dimensional mixture model of two Gaussian distributions centered at
-4 and +1. For simplicity, this toy model has a single parameter θ that controls the
standard deviations of both distributions. The top left contour plot in Figure 9-20
shows the entire model f(x; θ) as a function of both x and θ. To estimate the probabil‐
ity distribution of a future outcome x, you need to set the model parameter θ. For
example, if you set it to θ=1.3 (the horizontal line), you get the probability density
function f(x; θ=1.3) shown in the lower left plot. Say you want to estimate the proba‐
bility that x will fall between -2 and +2, you must calculate the integral of the PDF on
this range (i.e., the surface of the shaded region). On the other hand, if you have
observed a single instance x=2.5 (the vertical line in the upper left plot), you get the
likelihood function noted ℒ(θ|x=2.5)=f(x=2.5; θ) represented in the upper right plot.
In short, the PDF is a function of x (with θ fixed) while the likelihood function is a
function of θ (with x fixed). It is important to understand that the likelihood function
is not a probability distribution: if you integrate a probability distribution over all
```

**268 | Chapter 9: Unsupervised Learning Techniques**

```
possible values of x, you always get 1, but if you integrate the likelihood function over
all possible values of θ, the result can be any positive value.
```

```
Figure 9-20. A model’s parametric function (top left), and some derived functions: a PDF
(lower left), a likelihood function (top right) and a log likelihood function (lower right)
```

```
Given a dataset X , a common task is to try to estimate the most likely values for the
model parameters. To do this, you must find the values that maximize the likelihood
function, given X. In this example, if you have observed a single instance x=2.5, the
maximum likelihood estimate (MLE) of θ is θ=1.5. If a prior probability distribution g
over θ exists, it is possible to take it into account by maximizing ℒ(θ|x)g(θ) rather
than just maximizing ℒ(θ|x). This is called maximum a-posteriori (MAP) estimation.
Since MAP constrains the parameter values, you can think of it as a regularized ver‐
sion of MLE.
Notice that it is equivalent to maximize the likelihood function or to maximize its
logarithm (represented in the lower right hand side of Figure 9-20): indeed, the loga‐
rithm is a strictly increasing function, so if θ maximizes the log likelihood, it also
maximizes the likelihood. It turns out that it is generally easier to maximize the log
likelihood. For example, if you observed several independent instances x(1) to x(m), you
would need to find the value of θ that maximizes the product of the individual likeli‐
hood functions. But it is equivalent, and much simpler, to maximize the sum (not the
product) of the log likelihood functions, thanks to the magic of the logarithm which
converts products into sums: log(ab)=log(a)+log(b).
```

```
Once you have estimated θ, the value of θ that maximizes the likelihood function,
then you are ready to compute L=ℒθ,𝐀. This is the value which is used to com‐
pute the AIC and BIC: you can think of it as a measure of how well the model fits the
data.
```

To compute the BIC and AIC, just call the bic() or aic() methods:

```
Gaussian Mixtures | 269
```

```
>>> gm.bic(X)
8189.74345832983
>>> gm.aic(X)
8102.518178214792
```

Figure 9-21 shows the BIC for different numbers of clusters k. As you can see, both
the BIC and the AIC are lowest when k=3, so it is most likely the best choice. Note
that we could also search for the best value for the covariance_type hyperparameter.
For example, if it is "spherical" rather than "full", then the model has much fewer
parameters to learn, but it does not fit the data as well.

Figure 9-21. AIC and BIC for different numbers of clusters k

##### Bayesian Gaussian Mixture Models

Rather than manually searching for the optimal number of clusters, it is possible to
use instead the BayesianGaussianMixture class which is capable of giving weights
equal (or close) to zero to unnecessary clusters. Just set the number of clusters n_com
ponents to a value that you have good reason to believe is greater than the optimal
number of clusters (this assumes some minimal knowledge about the problem at
hand), and the algorithm will eliminate the unnecessary clusters automatically. For
example, let’s set the number of clusters to 10 and see what happens:

```
>>> from sklearn.mixture import BayesianGaussianMixture
>>> bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
>>> bgm.fit(X)
>>> np.round(bgm.weights_, 2)
array([0.4 , 0.21, 0.4 , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])
```

Perfect: the algorithm automatically detected that only 3 clusters are needed, and the
resulting clusters are almost identical to the ones in Figure 9-17.

In this model, the cluster parameters (including the weights, means and covariance
matrices) are not treated as fixed model parameters anymore, but as latent random
variables, like the cluster assignments (see Figure 9-22). So **z** now includes both the
cluster parameters and the cluster assignments.

**270 | Chapter 9: Unsupervised Learning Techniques**

Figure 9-22. Bayesian Gaussian mixture model

Prior knowledge about the latent variables **z** can be encoded in a probability distribu‐
tion p( **z** ) called the prior. For example, we may have a prior belief that the clusters are
likely to be few (low concentration), or conversely, that they are more likely to be
plentiful (high concentration). This can be adjusted using the weight_concentra
tion_prior hyperparameter. Setting it to 0.01 or 1000 gives very different clusterings
(see Figure 9-23). However, the more data we have, the less the priors matter. In fact,
to plot diagrams with such large differences, you must use very strong priors and lit‐
tle data.

Figure 9-23. Using different concentration priors

```
The fact that you see only 3 regions in the right plot although there
are 4 centroids is not a bug: the weight of the top-right cluster is
much larger than the weight of the lower-right cluster, so the prob‐
ability that any given point in this region belongs to the top-right
cluster is greater than the probability that it belongs to the lower-
right cluster, even near the lower-right cluster.
```

```
Gaussian Mixtures | 271
```

Bayes’ theorem (Equation 9-2) tells us how to update the probability distribution over
the latent variables after we observe some data **X**. It computes the posterior distribu‐
tion p( **z** | **X** ), which is the conditional probability of **z** given **X**.

```
Equation 9-2. Bayes’ theorem
```

```
p zX = Posterior = Likelihood×PriorEvidence =p X p zX p z
```

Unfortunately, in a Gaussian mixture model (and many other problems), the denomi‐
nator p( **x** ) is intractable, as it requires integrating over all the possible values of **z**
(Equation 9-3). This means considering all possible combinations of cluster parame‐
ters and cluster assignments.

```
Equation 9-3. The evidence p( X ) is often intractable
```

p **X** =∫p **Xz** p **z** d **z**

This is one of the central problems in Bayesian statistics, and there are several
approaches to solving it. One of them is variational inference, which picks a family of
distributions q( **z** ; **λ** ) with its own variational parameters **λ** (lambda), then it optimizes
these parameters to make q( **z** ) a good approximation of p( **z** | **X** ). This is achieved by
finding the value of **λ** that minimizes the KL divergence from q( **z** ) to p( **z** | **X** ), noted
DKL(q‖p). The KL divergence equation is shown in (see Equation 9-4), and it can be
rewritten as the log of the evidence (log p( **X** )) minus the evidence lower bound
(ELBO). Since the log of the evidence does not depend on q, it is a constant term, so
minimizing the KL divergence just requires maximizing the ELBO.

```
Equation 9-4. KL divergence from q( z ) to p( z | X )
```

```
DKLq∥p =𝐀q logpq zzX
```

```
=𝐀q logq z − logp z X
```

```
=𝐀q logq z − logpp zX , X
```

```
=𝐀q logq z − logp z , X + logp X
=𝐀q logq z −𝐀q logp z , X +𝐀q logp X
=𝐀q logp X − 𝐀q logp z , X −𝐀q logq z
= logp X − ELBO
where ELBO =𝐀q log p z , X −𝐀q logq z
```

**272 | Chapter 9: Unsupervised Learning Techniques**

In practice, there are different techniques to maximize the ELBO. In mean field varia‐
tional inference, it is necessary to pick the family of distributions q( **z** ; **λ** ) and the prior
p(z) very carefully to ensure that the equation for the ELBO simplifies to a form that
can actually be computed. Unfortunately, there is no general way to do this, it
depends on the task and requires some mathematical skills. For example, the distribu‐
tions and lower bound equations used in Scikit-Learn’s BayesianGaussianMixture
class are presented in the documentation. From these equations it is possible to derive
update equations for the cluster parameters and assignment variables: these are then
used very much like in the Expectation-Maximization algorithm. In fact, the compu‐
tational complexity of the BayesianGaussianMixture class is similar to that of the
GaussianMixture class (but generally significantly slower). A simpler approach to
maximizing the ELBO is called black box stochastic variational inference (BBSVI): at
each iteration, a few samples are drawn from q and they are used to estimate the gra‐
dients of the ELBO with regards to the variational parameters **λ** , which are then used
in a gradient ascent step. This approach makes it possible to use Bayesian inference
with any kind of model (provided it is differentiable), even deep neural networks: this
is called Bayesian deep learning.

```
If you want to dive deeper into Bayesian statistics, check out the
Bayesian Data Analysis book by Andrew Gelman, John Carlin, Hal
Stern, David Dunson, Aki Vehtari, and Donald Rubin.
```

Gaussian mixture models work great on clusters with ellipsoidal shapes, but if you try
to fit a dataset with different shapes, you may have bad surprises. For example, let’s
see what happens if we use a Bayesian Gaussian mixture model to cluster the moons
dataset (see Figure 9-24):

Figure 9-24. moons_vs_bgm_diagram

Oops, the algorithm desperately searched for ellipsoids, so it found 8 different clus‐
ters instead of 2. The density estimation is not too bad, so this model could perhaps
be used for anomaly detection, but it failed to identify the two moons. Let’s now look
at a few clustering algorithms capable of dealing with arbitrarily shaped clusters.

```
Gaussian Mixtures | 273
```

##### Other Anomaly Detection and Novelty Detection Algorithms

Scikit-Learn also implements a few algorithms dedicated to anomaly detection or
novelty detection:

- Fast-MCD (minimum covariance determinant), implemented by the EllipticEn
  velope class: this algorithm is useful for outlier detection, in particular to
  cleanup a dataset. It assumes that the normal instances (inliers) are generated
  from a single Gaussian distribution (not a mixture), but it also assumes that the
  dataset is contaminated with outliers that were not generated from this Gaussian
  distribution. When it estimates the parameters of the Gaussian distribution (i.e.,
  the shape of the elliptic envelope around the inliers), it is careful to ignore the
  instances that are most likely outliers. This gives a better estimation of the elliptic
  envelope, and thus makes it better at identifying the outliers.
- Isolation forest: this is an efficient algorithm for outlier detection, especially in
  high-dimensional datasets. The algorithm builds a Random Forest in which each
  Decision Tree is grown randomly: at each node, it picks a feature randomly, then
  it picks a random threshold value (between the min and max value) to split the
  dataset in two. The dataset gradually gets chopped into pieces this way, until all
  instances end up isolated from the other instances. An anomaly is usually far
  from other instances, so on average (across all the Decision Trees) it tends to get
  isolated in less steps than normal instances.
- Local outlier factor (LOF): this algorithm is also good for outlier detection. It
  compares the density of instances around a given instance to the density around
  its neighbors. An anomaly is often more isolated than its k nearest neighbors.
- One-class SVM: this algorithm is better suited for novelty detection. Recall that a
  kernelized SVM classifier separates two classes by first (implicitly) mapping all
  the instances to a high-dimensional space, then separating the two classes using a
  linear SVM classifier within this high-dimensional space (see Chapter 5). Since
  we just have one class of instances, the one-class SVM algorithm instead tries to
  separate the instances in high-dimensional space from the origin. In the original
  space, this will correspond to finding a small region that encompasses all the
  instances. If a new instance does not fall within this region, it is an anomaly.
  There are a few hyperparameters to tweak: the usual ones for a kernelized SVM,
  plus a margin hyperparameter that corresponds to the probability of a new
  instance being mistakenly considered as novel, when it is in fact normal. It works
  great, especially with high-dimensional datasets, but just like all SVMs, it does
  not scale to large datasets.

**274 | Chapter 9: Unsupervised Learning Techniques**

### PART II

**Neural Networks and Deep Learning**

```
1 You can get the best of both worlds by being open to biological inspirations without being afraid to create
biologically unrealistic models, as long as they work well.
```

**CHAPTER 10**

**Introduction to Artificial Neural Networks**

**with Keras**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 10 in the final
release of the book.
```

Birds inspired us to fly, burdock plants inspired velcro, and countless more inven‐
tions were inspired by nature. It seems only logical, then, to look at the brain’s archi‐
tecture for inspiration on how to build an intelligent machine. This is the key idea
that sparked artificial neural networks (ANNs). However, although planes were
inspired by birds, they don’t have to flap their wings. Similarly, ANNs have gradually
become quite different from their biological cousins. Some researchers even argue
that we should drop the biological analogy altogether (e.g., by saying “units” rather
than “neurons”), lest we restrict our creativity to biologically plausible systems.^1

ANNs are at the very core of Deep Learning. They are versatile, powerful, and scala‐
ble, making them ideal to tackle large and highly complex Machine Learning tasks,
such as classifying billions of images (e.g., Google Images), powering speech recogni‐
tion services (e.g., Apple’s Siri), recommending the best videos to watch to hundreds
of millions of users every day (e.g., YouTube), or learning to beat the world champion
at the game of Go by playing millions of games against itself (DeepMind’s Alpha‐
Zero).

```
277
```

```
2 “A Logical Calculus of Ideas Immanent in Nervous Activity,” W. McCulloch and W. Pitts (1943).
```

In the first part of this chapter, we will introduce artificial neural networks, starting
with a quick tour of the very first ANN architectures, leading up to Multi-Layer Per‐
ceptrons (MLPs) which are heavily used today (other architectures will be explored in
the next chapters). In the second part, we will look at how to implement neural net‐
works using the popular Keras API. This is a beautifully designed and simple high-
level API for building, training, evaluating and running neural networks. But don’t be
fooled by its simplicity: it is expressive and flexible enough to let you build a wide
variety of neural network architectures. In fact, it will probably be sufficient for most
of your use cases. Moreover, should you ever need extra flexibility, you can always
write custom Keras components using its lower-level API, as we will see in Chap‐
ter 12.

But first, let’s go back in time to see how artificial neural networks came to be!

#### From Biological to Artificial Neurons

Surprisingly, ANNs have been around for quite a while: they were first introduced
back in 1943 by the neurophysiologist Warren McCulloch and the mathematician
Walter Pitts. In their landmark paper,^2 “A Logical Calculus of Ideas Immanent in
Nervous Activity,” McCulloch and Pitts presented a simplified computational model
of how biological neurons might work together in animal brains to perform complex
computations using propositional logic. This was the first artificial neural network
architecture. Since then many other architectures have been invented, as we will see.

The early successes of ANNs until the 1960s led to the widespread belief that we
would soon be conversing with truly intelligent machines. When it became clear that
this promise would go unfulfilled (at least for quite a while), funding flew elsewhere
and ANNs entered a long winter. In the early 1980s there was a revival of interest in
connectionism (the study of neural networks), as new architectures were invented and
better training techniques were developed. But progress was slow, and by the 1990s
other powerful Machine Learning techniques were invented, such as Support Vector
Machines (see Chapter 5). These techniques seemed to offer better results and stron‐
ger theoretical foundations than ANNs, so once again the study of neural networks
entered a long winter.

Finally, we are now witnessing yet another wave of interest in ANNs. Will this wave
die out like the previous ones did? Well, there are a few good reasons to believe that
this wave is different and that it will have a much more profound impact on our lives:

**278 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

- There is now a huge quantity of data available to train neural networks, and
  ANNs frequently outperform other ML techniques on very large and complex
  problems.
- The tremendous increase in computing power since the 1990s now makes it pos‐
  sible to train large neural networks in a reasonable amount of time. This is in
  part due to Moore’s Law, but also thanks to the gaming industry, which has pro‐
  duced powerful GPU cards by the millions.
- The training algorithms have been improved. To be fair they are only slightly dif‐
  ferent from the ones used in the 1990s, but these relatively small tweaks have a
  huge positive impact.
- Some theoretical limitations of ANNs have turned out to be benign in practice.
  For example, many people thought that ANN training algorithms were doomed
  because they were likely to get stuck in local optima, but it turns out that this is
  rather rare in practice (or when it is the case, they are usually fairly close to the
  global optimum).
- ANNs seem to have entered a virtuous circle of funding and progress. Amazing
  products based on ANNs regularly make the headline news, which pulls more
  and more attention and funding toward them, resulting in more and more pro‐
  gress, and even more amazing products.

##### Biological Neurons

Before we discuss artificial neurons, let’s take a quick look at a biological neuron (rep‐
resented in Figure 10-1). It is an unusual-looking cell mostly found in animal cerebral
cortexes (e.g., your brain), composed of a cell body containing the nucleus and most
of the cell’s complex components, and many branching extensions called dendrites,
plus one very long extension called the axon. The axon’s length may be just a few
times longer than the cell body, or up to tens of thousands of times longer. Near its
extremity the axon splits off into many branches called telodendria, and at the tip of
these branches are minuscule structures called synaptic terminals (or simply synap‐
ses), which are connected to the dendrites (or directly to the cell body) of other neu‐
rons. Biological neurons receive short electrical impulses called signals from other
neurons via these synapses. When a neuron receives a sufficient number of signals
from other neurons within a few milliseconds, it fires its own signals.

```
From Biological to Artificial Neurons | 279
```

```
3 Image by Bruce Blaus (Creative Commons 3.0). Reproduced from https://en.wikipedia.org/wiki/Neuron.
4 In the context of Machine Learning, the phrase “neural networks” generally refers to ANNs, not BNNs.
5 Drawing of a cortical lamination by S. Ramon y Cajal (public domain). Reproduced from https://en.wikipe
dia.org/wiki/Cerebral_cortex.
```

Figure 10-1. Biological neuron^3

Thus, individual biological neurons seem to behave in a rather simple way, but they
are organized in a vast network of billions of neurons, each neuron typically connec‐
ted to thousands of other neurons. Highly complex computations can be performed
by a vast network of fairly simple neurons, much like a complex anthill can emerge
from the combined efforts of simple ants. The architecture of biological neural net‐
works (BNN)^4 is still the subject of active research, but some parts of the brain have
been mapped, and it seems that neurons are often organized in consecutive layers, as
shown in Figure 10-2.

Figure 10-2. Multiple layers in a biological neural network (human cortex)^5

**280 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

##### Logical Computations with Neurons

Warren McCulloch and Walter Pitts proposed a very simple model of the biological
neuron, which later became known as an artificial neuron: it has one or more binary
(on/off ) inputs and one binary output. The artificial neuron simply activates its out‐
put when more than a certain number of its inputs are active. McCulloch and Pitts
showed that even with such a simplified model it is possible to build a network of
artificial neurons that computes any logical proposition you want. For example, let’s
build a few ANNs that perform various logical computations (see Figure 10-3),
assuming that a neuron is activated when at least two of its inputs are active.

Figure 10-3. ANNs performing simple logical computations

- The first network on the left is simply the identity function: if neuron A is activa‐
  ted, then neuron C gets activated as well (since it receives two input signals from
  neuron A), but if neuron A is off, then neuron C is off as well.
- The second network performs a logical AND: neuron C is activated only when
  both neurons A and B are activated (a single input signal is not enough to acti‐
  vate neuron C).
- The third network performs a logical OR: neuron C gets activated if either neu‐
  ron A or neuron B is activated (or both).
- Finally, if we suppose that an input connection can inhibit the neuron’s activity
  (which is the case with biological neurons), then the fourth network computes a
  slightly more complex logical proposition: neuron C is activated only if neuron A
  is active and if neuron B is off. If neuron A is active all the time, then you get a
  logical NOT: neuron C is active when neuron B is off, and vice versa.

You can easily imagine how these networks can be combined to compute complex
logical expressions (see the exercises at the end of the chapter).

##### The Perceptron

The Perceptron is one of the simplest ANN architectures, invented in 1957 by Frank
Rosenblatt. It is based on a slightly different artificial neuron (see Figure 10-4) called

```
From Biological to Artificial Neurons | 281
```

```
6 The name Perceptron is sometimes used to mean a tiny network with a single TLU.
```

a threshold logic unit (TLU), or sometimes a linear threshold unit (LTU): the inputs
and output are now numbers (instead of binary on/off values) and each input con‐
nection is associated with a weight. The TLU computes a weighted sum of its inputs
(z = w 1 x 1 + w 2 x 2 + ⋯ + wn xn = **x** T **w** ), then applies a step function to that sum and
outputs the result: h **w** ( **x** ) = step(z), where z = **x** T **w**.

Figure 10-4. Threshold logic unit

The most common step function used in Perceptrons is the Heaviside step function
(see Equation 10-1). Sometimes the sign function is used instead.

```
Equation 10-1. Common step functions used in Perceptrons
```

```
heaviside z =
```

```
0 ifz< 0
1 ifz≥ 0
```

```
sgn z =
```

```
−1 ifz< 0
0 ifz= 0
+1 ifz> 0
```

A single TLU can be used for simple linear binary classification. It computes a linear
combination of the inputs and if the result exceeds a threshold, it outputs the positive
class or else outputs the negative class (just like a Logistic Regression classifier or a
linear SVM). For example, you could use a single TLU to classify iris flowers based on
the petal length and width (also adding an extra bias feature x 0 = 1, just like we did in
previous chapters). Training a TLU in this case means finding the right values for w 0 ,
w 1 , and w 2 (the training algorithm is discussed shortly).

A Perceptron is simply composed of a single layer of TLUs,^6 with each TLU connected
to all the inputs. When all the neurons in a layer are connected to every neuron in the
previous layer (i.e., its input neurons), it is called a fully connected layer or a dense
layer. To represent the fact that each input is sent to every TLU, it is common to draw
special passthrough neurons called input neurons: they just output whatever input
they are fed. All the input neurons form the input layer. Moreover, an extra bias fea‐

**282 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

ture is generally added (x 0 = 1): it is typically represented using a special type of neu‐
ron called a bias neuron, which just outputs 1 all the time. A Perceptron with two
inputs and three outputs is represented in Figure 10-5. This Perceptron can classify
instances simultaneously into three different binary classes, which makes it a multi‐
output classifier.

Figure 10-5. Perceptron diagram

Thanks to the magic of linear algebra, it is possible to efficiently compute the outputs
of a layer of artificial neurons for several instances at once, by using Equation 10-2:

```
Equation 10-2. Computing the outputs of a fully connected layer
h W , bX =φ XW + b
```

- As always, **X** represents the matrix of input features. It has one row per instance,
  one column per feature.
- The weight matrix **W** contains all the connection weights except for the ones
  from the bias neuron. It has one row per input neuron and one column per artifi‐
  cial neuron in the layer.
- The bias vector **b** contains all the connection weights between the bias neuron
  and the artificial neurons. It has one bias term per artificial neuron.
- The function φ is called the activation function: when the artificial neurons are
  TLUs, it is a step function (but we will discuss other activation functions shortly).

So how is a Perceptron trained? The Perceptron training algorithm proposed by
Frank Rosenblatt was largely inspired by Hebb’s rule. In his book The Organization of
Behavior, published in 1949, Donald Hebb suggested that when a biological neuron
often triggers another neuron, the connection between these two neurons grows
stronger. This idea was later summarized by Siegrid Löwel in this catchy phrase:
“Cells that fire together, wire together.” This rule later became known as Hebb’s rule

```
From Biological to Artificial Neurons | 283
```

```
7 Note that this solution is generally not unique: in general when the data are linearly separable, there is an
infinity of hyperplanes that can separate them.
```

(or Hebbian learning); that is, the connection weight between two neurons is
increased whenever they have the same output. Perceptrons are trained using a var‐
iant of this rule that takes into account the error made by the network; it reinforces
connections that help reduce the error. More specifically, the Perceptron is fed one
training instance at a time, and for each instance it makes its predictions. For every
output neuron that produced a wrong prediction, it reinforces the connection
weights from the inputs that would have contributed to the correct prediction. The
rule is shown in Equation 10-3.

```
Equation 10-3. Perceptron learning rule (weight update)
```

```
wi,jnext step=wi,j+ηyj−yjxi
```

- wi, j is the connection weight between the ith input neuron and the jth output neu‐
  ron.
- xi is the ith input value of the current training instance.
- yj is the output of the jth output neuron for the current training instance.
- yj is the target output of the jth output neuron for the current training instance.
- η is the learning rate.

The decision boundary of each output neuron is linear, so Perceptrons are incapable
of learning complex patterns (just like Logistic Regression classifiers). However, if the
training instances are linearly separable, Rosenblatt demonstrated that this algorithm
would converge to a solution.^7 This is called the Perceptron convergence theorem.

Scikit-Learn provides a Perceptron class that implements a single TLU network. It
can be used pretty much as you would expect—for example, on the iris dataset (intro‐
duced in Chapter 4):

```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
```

```
iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width
y = (iris.target == 0).astype(np.int) # Iris Setosa?
```

```
per_clf = Perceptron()
per_clf.fit(X, y)
```

**284 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

```
y_pred = per_clf.predict([[2, 0.5]])
```

You may have noticed the fact that the Perceptron learning algorithm strongly resem‐
bles Stochastic Gradient Descent. In fact, Scikit-Learn’s Perceptron class is equivalent
to using an SGDClassifier with the following hyperparameters: loss="perceptron",
learning_rate="constant", eta0=1 (the learning rate), and penalty=None (no regu‐
larization).

Note that contrary to Logistic Regression classifiers, Perceptrons do not output a class
probability; rather, they just make predictions based on a hard threshold. This is one
of the good reasons to prefer Logistic Regression over Perceptrons.

In their 1969 monograph titled Perceptrons, Marvin Minsky and Seymour Papert
highlighted a number of serious weaknesses of Perceptrons, in particular the fact that
they are incapable of solving some trivial problems (e.g., the Exclusive OR (XOR)
classification problem; see the left side of Figure 10-6). Of course this is true of any
other linear classification model as well (such as Logistic Regression classifiers), but
researchers had expected much more from Perceptrons, and their disappointment
was great, and many researchers dropped neural networks altogether in favor of
higher-level problems such as logic, problem solving, and search.

However, it turns out that some of the limitations of Perceptrons can be eliminated by
stacking multiple Perceptrons. The resulting ANN is called a Multi-Layer Perceptron
(MLP). In particular, an MLP can solve the XOR problem, as you can verify by com‐
puting the output of the MLP represented on the right of Figure 10-6: with inputs (0, 0) or (1, 1) the network outputs 0, and with inputs (0, 1) or (1, 0) it outputs 1. All
connections have a weight equal to 1, except the four connections where the weight is
shown. Try verifying that this network indeed solves the XOR problem!

Figure 10-6. XOR classification problem and an MLP that solves it

```
From Biological to Artificial Neurons | 285
```

```
8 In the 1990s, an ANN with more than two hidden layers was considered deep. Nowadays, it is common to see
ANNs with dozens of layers, or even hundreds, so the definition of “deep” is quite fuzzy.
9 “Learning Internal Representations by Error Propagation,” D. Rumelhart, G. Hinton, R. Williams (1986).
```

##### Multi-Layer Perceptron and Backpropagation

An MLP is composed of one (passthrough) input layer, one or more layers of TLUs,
called hidden layers, and one final layer of TLUs called the output layer (see
Figure 10-7). The layers close to the input layer are usually called the lower layers,
and the ones close to the outputs are usually called the upper layers. Every layer
except the output layer includes a bias neuron and is fully connected to the next layer.

Figure 10-7. Multi-Layer Perceptron

```
The signal flows only in one direction (from the inputs to the out‐
puts), so this architecture is an example of a feedforward neural net‐
work (FNN).
```

When an ANN contains a deep stack of hidden layers^8 , it is called a deep neural net‐
work (DNN). The field of Deep Learning studies DNNs, and more generally models
containing deep stacks of computations. However, many people talk about Deep
Learning whenever neural networks are involved (even shallow ones).

For many years researchers struggled to find a way to train MLPs, without success.
But in 1986, David Rumelhart, Geoffrey Hinton and Ronald Williams published a
groundbreaking paper^9 introducing the backpropagation training algorithm, which is
still used today. In short, it is simply Gradient Descent (introduced in Chapter 4)

**286 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

10 This technique was actually independently invented several times by various researchers in different fields,
starting with P. Werbos in 1974.

```
using an efficient technique for computing the gradients automatically^10 : in just two
passes through the network (one forward, one backward), the backpropagation algo‐
rithm is able to compute the gradient of the network’s error with regards to every sin‐
gle model parameter. In other words, it can find out how each connection weight and
each bias term should be tweaked in order to reduce the error. Once it has these gra‐
dients, it just performs a regular Gradient Descent step, and the whole process is
repeated until the network converges to the solution.
```

```
Automatically computing gradients is called automatic differentia‐
tion, or autodiff. There are various autodiff techniques, with differ‐
ent pros and cons. The one used by backpropagation is called
reverse-mode autodiff. It is fast and precise, and is well suited when
the function to differentiate has many variables (e.g., connection
weights) and few outputs (e.g., one loss). If you want to learn more
about autodiff, check out ???.
```

```
Let’s run through this algorithm in a bit more detail:
```

- It handles one mini-batch at a time (for example containing 32 instances each),
  and it goes through the full training set multiple times. Each pass is called an
  epoch, as we saw in Chapter 4.
- Each mini-batch is passed to the network’s input layer, which just sends it to the
  first hidden layer. The algorithm then computes the output of all the neurons in
  this layer (for every instance in the mini-batch). The result is passed on to the
  next layer, its output is computed and passed to the next layer, and so on until we
  get the output of the last layer, the output layer. This is the forward pass: it is
  exactly like making predictions, except all intermediate results are preserved
  since they are needed for the backward pass.
- Next, the algorithm measures the network’s output error (i.e., it uses a loss func‐
  tion that compares the desired output and the actual output of the network, and
  returns some measure of the error).
- Then it computes how much each output connection contributed to the error.
  This is done analytically by simply applying the chain rule (perhaps the most fun‐
  damental rule in calculus), which makes this step fast and precise.
- The algorithm then measures how much of these error contributions came from
  each connection in the layer below, again using the chain rule—and so on until
  the algorithm reaches the input layer. As we explained earlier, this reverse pass
  efficiently measures the error gradient across all the connection weights in the

```
From Biological to Artificial Neurons | 287
```

```
network by propagating the error gradient backward through the network (hence
the name of the algorithm).
```

- Finally, the algorithm performs a Gradient Descent step to tweak all the connec‐
  tion weights in the network, using the error gradients it just computed.

This algorithm is so important, it’s worth summarizing it again: for each training
instance the backpropagation algorithm first makes a prediction (forward pass),
measures the error, then goes through each layer in reverse to measure the error con‐
tribution from each connection (reverse pass), and finally slightly tweaks the connec‐
tion weights to reduce the error (Gradient Descent step).

```
It is important to initialize all the hidden layers’ connection weights
randomly, or else training will fail. For example, if you initialize all
weights and biases to zero, then all neurons in a given layer will be
perfectly identical, and thus backpropagation will affect them in
exactly the same way, so they will remain identical. In other words,
despite having hundreds of neurons per layer, your model will act
as if it had only one neuron per layer: it won’t be too smart. If
instead you randomly initialize the weights, you break the symme‐
try and allow backpropagation to train a diverse team of neurons.
```

In order for this algorithm to work properly, the authors made a key change to the
MLP’s architecture: they replaced the step function with the logistic function, σ(z) =
1 / (1 + exp(–z)). This was essential because the step function contains only flat seg‐
ments, so there is no gradient to work with (Gradient Descent cannot move on a flat
surface), while the logistic function has a well-defined nonzero derivative every‐
where, allowing Gradient Descent to make some progress at every step. In fact, the
backpropagation algorithm works well with many other activation functions, not just
the logistic function. Two other popular activation functions are:

The hyperbolic tangent function tanh(z) = 2σ(2z) – 1
Just like the logistic function it is S-shaped, continuous, and differentiable, but its
output value ranges from –1 to 1 (instead of 0 to 1 in the case of the logistic func‐
tion), which tends to make each layer’s output more or less centered around 0 at
the beginning of training. This often helps speed up convergence.

The Rectified Linear Unit function: ReLU(z) = max(0, z)
It is continuous but unfortunately not differentiable at z = 0 (the slope changes
abruptly, which can make Gradient Descent bounce around), and its derivative is
0 for z < 0. However, in practice it works very well and has the advantage of being

**288 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

11 Biological neurons seem to implement a roughly sigmoid (S-shaped) activation function, so researchers stuck
to sigmoid functions for a very long time. But it turns out that ReLU generally works better in ANNs. This is
one of the cases where the biological analogy was misleading.

```
fast to compute^11. Most importantly, the fact that it does not have a maximum
output value also helps reduce some issues during Gradient Descent (we will
come back to this in Chapter 11).
These popular activation functions and their derivatives are represented in
Figure 10-8. But wait! Why do we need activation functions in the first place? Well, if
you chain several linear transformations, all you get is a linear transformation. For
example, say f(x) = 2 x + 3 and g(x) = 5 x - 1, then chaining these two linear functions
gives you another linear function: f(g(x)) = 2(5 x - 1) + 3 = 10 x + 1. So if you don’t
have some non-linearity between layers, then even a deep stack of layers is equivalent
to a single layer: you cannot solve very complex problems with that.
```

```
Figure 10-8. Activation functions and their derivatives
```

```
Okay! So now you know where neural nets came from, what their architecture is and
how to compute their outputs, and you also learned about the backpropagation algo‐
rithm. But what exactly can you do with them?
```

##### Regression MLPs

```
First, MLPs can be used for regression tasks. If you want to predict a single value (e.g.,
the price of a house given many of its features), then you just need a single output
neuron: its output is the predicted value. For multivariate regression (i.e., to predict
multiple values at once), you need one output neuron per output dimension. For
example, to locate the center of an object on an image, you need to predict 2D coordi‐
nates, so you need two output neurons. If you also want to place a bounding box
around the object, then you need two more numbers: the width and the height of the
object. So you end up with 4 output neurons.
```

```
From Biological to Artificial Neurons | 289
```

In general, when building an MLP for regression, you do not want to use any activa‐
tion function for the output neurons, so they are free to output any range of values.
However, if you want to guarantee that the output will always be positive, then you
can use the ReLU activation function, or the softplus activation function in the output
layer. Finally, if you want to guarantee that the predictions will fall within a given
range of values, then you can use the logistic function or the hyperbolic tangent, and
scale the labels to the appropriate range: 0 to 1 for the logistic function, or –1 to 1 for
the hyperbolic tangent.

The loss function to use during training is typically the mean squared error, but if you
have a lot of outliers in the training set, you may prefer to use the mean absolute
error instead. Alternatively, you can use the Huber loss, which is a combination of
both.

```
The Huber loss is quadratic when the error is smaller than a thres‐
hold δ (typically 1), but linear when the error is larger than δ. This
makes it less sensitive to outliers than the mean squared error, and
it is often more precise and converges faster than the mean abso‐
lute error.
```

Table 10-1 summarizes the typical architecture of a regression MLP.

Table 10-1. Typical Regression MLP Architecture

```
Hyperparameter Typical Value
# input neurons One per input feature (e.g., 28 x 28 = 784 for MNIST)
# hidden layers Depends on the problem. Typically 1 to 5.
# neurons per hidden layer Depends on the problem. Typically 10 to 100.
# output neurons 1 per prediction dimension
Hidden activation ReLU (or SELU, see Chapter 11)
Output activation None or ReLU/Softplus (if positive outputs) or Logistic/Tanh (if bounded outputs)
Loss function MSE or MAE/Huber (if outliers)
```

##### Classification MLPs

MLPs can also be used for classification tasks. For a binary classification problem,
you just need a single output neuron using the logistic activation function: the output
will be a number between 0 and 1, which you can interpret as the estimated probabil‐
ity of the positive class. Obviously, the estimated probability of the negative class is
equal to one minus that number.

MLPs can also easily handle multilabel binary classification tasks (see Chapter 3). For
example, you could have an email classification system that predicts whether each
incoming email is ham or spam, and simultaneously predicts whether it is an urgent

**290 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

or non-urgent email. In this case, you would need two output neurons, both using
the logistic activation function: the first would output the probability that the email is
spam and the second would output the probability that it is urgent. More generally,
you would dedicate one output neuron for each positive class. Note that the output
probabilities do not necessarily add up to one. This lets the model output any combi‐
nation of labels: you can have non-urgent ham, urgent ham, non-urgent spam, and
perhaps even urgent spam (although that would probably be an error).

If each instance can belong only to a single class, out of 3 or more possible classes
(e.g., classes 0 through 9 for digit image classification), then you need to have one
output neuron per class, and you should use the softmax activation function for the
whole output layer (see Figure 10-9). The softmax function (introduced in Chapter 4)
will ensure that all the estimated probabilities are between 0 and 1 and that they add
up to one (which is required if the classes are exclusive). This is called multiclass clas‐
sification.

Figure 10-9. A modern MLP (including ReLU and softmax) for classification

Regarding the loss function, since we are predicting probability distributions, the
cross-entropy (also called the log loss, see Chapter 4) is generally a good choice.

Table 10-2 summarizes the typical architecture of a classification MLP.

Table 10-2. Typical Classification MLP Architecture

```
Hyperparameter Binary classification Multilabel binary classification Multiclass classification
Input and hidden layers Same as regression Same as regression Same as regression
# output neurons 1 1 per label 1 per class
Output layer activation Logistic Logistic Softmax
```

```
From Biological to Artificial Neurons | 291
```

12 Project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

```
Hyperparameter Binary classification Multilabel binary classification Multiclass classification
Loss function Cross-Entropy Cross-Entropy Cross-Entropy
```

```
Before we go on, I recommend you go through exercise 1, at the
end of this chapter. You will play with various neural network
architectures and visualize their outputs using the TensorFlow Play‐
ground. This will be very useful to better understand MLPs, for
example the effects of all the hyperparameters (number of layers
and neurons, activation functions, and more).
```

```
Now you have all the concepts you need to start implementing MLPs with Keras!
```

#### Implementing MLPs with Keras

```
Keras is a high-level Deep Learning API that allows you to easily build, train, evaluate
and execute all sorts of neural networks. Its documentation (or specification) is avail‐
able at https://keras.io. The reference implementation is simply called Keras as well, so
to avoid any confusion we will call it keras-team (since it is available at https://
github.com/keras-team/keras). It was developed by François Chollet as part of a
research project^12 and released as an open source project in March 2015. It quickly
gained popularity owing to its ease-of-use, flexibility and beautiful design. To per‐
form the heavy computations required by neural networks, keras-team relies on a
computation backend. At the present, you can choose from three popular open
source deep learning libraries: TensorFlow, Microsoft Cognitive Toolkit (CNTK) or
Theano.
Moreover, since late 2016, other implementations have been released. You can now
run Keras on Apache MXNet, Apple’s Core ML, Javascript or Typescript (to run Keras
code in a web browser), or PlaidML (which can run on all sorts of GPU devices, not
just Nvidia). Moreover, TensorFlow itself now comes bundled with its own Keras
implementation called tf.keras. It only supports TensorFlow as the backend, but it has
the advantage of offering some very useful extra features (see Figure 10-10): for
example, it supports TensorFlow’s Data API which makes it quite easy to load and
preprocess data efficiently. For this reason, we will use tf.keras in this book. However,
in this chapter we will not use any of the TensorFlow-specific features, so the code
should run fine on other Keras implementations as well (at least in Python), with only
minor modifications, such as changing the imports.
```

```
292 | Chapter 10: Introduction to Artificial Neural Networks with Keras
```

Figure 10-10. Two Keras implementations: keras-team (left) and tf.keras (right)

As tf.keras is bundled with TensorFlow, let’s install TensorFlow!

##### Installing TensorFlow 2

Assuming you installed Jupyter and Scikit-Learn by following the installation instruc‐
tions in Chapter 2, you can simply use pip to install TensorFlow. If you created an
isolated environment using virtualenv, you first need to activate it:

```
$ cd $ML_PATH # Your ML working directory (e.g., $HOME/ml)
$ source env/bin/activate # on Linux or MacOSX
$ .\env\Scripts\activate # on Windows
```

Next, install TensorFlow 2 (if you are not using a virtualenv, you will need adminis‐
trator rights, or to add the --user option):

```
$ python3 -m pip install --upgrade tensorflow
```

```
For GPU support, you need to install tensorflow-gpu instead of
tensorflow, and there are other libraries to install. See https://
tensorflow.org/install/gpu for more details.
```

To test your installation, open a Python shell or a Jupyter notebook, then import Ten‐
sorFlow and tf.keras, and print their versions:

```
>>> import tensorflow as tf
>>> from tensorflow import keras
>>> tf.__version__
'2.0.0'
>>> keras.__version__
'2.2.4-tf'
```

```
Implementing MLPs with Keras | 293
```

The second version is the version of the Keras API implemented by tf.keras. Note that
it ends with -tf, highlighting the fact that tf.keras implements the Keras API, plus
some extra TensorFlow-specific features.

Now let’s use tf.keras! Let’s start by building a simple image classifier.

##### Building an Image Classifier Using the Sequential API

First, we need to load a dataset. We will tackle Fashion MNIST, which is a drop-in
replacement of MNIST (introduced in Chapter 3). It has the exact same format as
MNIST (70,000 grayscale images of 28×28 pixels each, with 10 classes), but the
images represent fashion items rather than handwritten digits, so each class is more
diverse and the problem turns out to be significantly more challenging than MNIST.
For example, a simple linear model reaches about 92% accuracy on MNIST, but only
about 83% on Fashion MNIST.

**Using Keras to Load the Dataset**

Keras provides some utility functions to fetch and load common datasets, including
MNIST, Fashion MNIST, the original California housing dataset, and more. Let’s load
Fashion MNIST:

```
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
```

When loading MNIST or Fashion MNIST using Keras rather than Scikit-Learn, one
important difference is that every image is represented as a 28×28 array rather than a
1D array of size 784. Moreover, the pixel intensities are represented as integers (from
0 to 255) rather than floats (from 0.0 to 255.0). Here is the shape and data type of the
training set:

```
>>> X_train_full.shape
(60000, 28, 28)
>>> X_train_full.dtype
dtype('uint8')
```

Note that the dataset is already split into a training set and a test set, but there is no
validation set, so let’s create one. Moreover, since we are going to train the neural net‐
work using Gradient Descent, we must scale the input features. For simplicity, we just
scale the pixel intensities down to the 0-1 range by dividing them by 255.0 (this also
converts them to floats):

```
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
```

With MNIST, when the label is equal to 5, it means that the image represents the
handwritten digit 5. Easy. However, for Fashion MNIST, we need the list of class
names to know what we are dealing with:

**294 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

```
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

For example, the first image in the training set represents a coat:

```
>>> class_names[y_train[0]]
'Coat'
```

Figure 10-11 shows a few samples from the Fashion MNIST dataset:

Figure 10-11. Samples from Fashion MNIST

**Creating the Model Using the Sequential API**

Now let’s build the neural network! Here is a classification MLP with two hidden lay‐
ers:

```
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
```

Let’s go through this code line by line:

- The first line creates a Sequential model. This is the simplest kind of Keras
  model, for neural networks that are just composed of a single stack of layers, con‐
  nected sequentially. This is called the sequential API.
- Next, we build the first layer and add it to the model. It is a Flatten layer whose
  role is simply to convert each input image into a 1D array: if it receives input data
  X, it computes X.reshape(-1, 1). This layer does not have any parameters, it is
  just there to do some simple preprocessing. Since it is the first layer in the model,
  you should specify the input_shape: this does not include the batch size, only the
  shape of the instances. Alternatively, you could add a keras.layers.InputLayer
  as the first layer, setting shape=[28,28].
- Next we add a Dense hidden layer with 300 neurons. It will use the ReLU activa‐
  tion function. Each Dense layer manages its own weight matrix, containing all the
  connection weights between the neurons and their inputs. It also manages a vec‐

```
Implementing MLPs with Keras | 295
```

```
tor of bias terms (one per neuron). When it receives some input data, it computes
Equation 10-2.
```

- Next we add a second Dense hidden layer with 100 neurons, also using the ReLU
  activation function.
- Finally, we add a Dense output layer with 10 neurons (one per class), using the
  softmax activation function (because the classes are exclusive).

```
Specifying activation="relu" is equivalent to activa
tion=keras.activations.relu. Other activation functions are
available in the keras.activations package, we will use many of
them in this book. See https://keras.io/activations/ for the full list.
```

Instead of adding the layers one by one as we just did, you can pass a list of layers
when creating the Sequential model:

```
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(300, activation="relu"),
keras.layers.Dense(100, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])
```

```
Using Code Examples From keras.io
Code examples documented on keras.io will work fine with tf.keras, but you need to
change the imports. For example, consider this keras.io code:
from keras.layers import Dense
output_layer = Dense(10)
You must change the imports like this:
from tensorflow.keras.layers import Dense
output_layer = Dense(10)
Or simply use full paths, if you prefer:
from tensorflow import keras
output_layer = keras.layers.Dense(10)
This is more verbose, but I use this approach in this book so you can easily see which
packages to use, and to avoid confusion between standard classes and custom classes.
In production code, I use the previous approach, as do most people.
```

**296 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

13 You can also generate an image of your model using keras.utils.plot_model().

```
The model’s summary() method displays all the model’s layers^13 , including each layer’s
name (which is automatically generated unless you set it when creating the layer), its
output shape (None means the batch size can be anything), and its number of parame‐
ters. The summary ends with the total number of parameters, including trainable and
non-trainable parameters. Here we only have trainable parameters (we will see exam‐
ples of non-trainable parameters in Chapter 11):
>>> model.summary()
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
flatten_1 (Flatten) (None, 784) 0
_________________________________________________________________
dense_3 (Dense) (None, 300) 235500
_________________________________________________________________
dense_4 (Dense) (None, 100) 30100
_________________________________________________________________
dense_5 (Dense) (None, 10) 1010
=================================================================
Total params: 266,610
Trainable params: 266,610
Non-trainable params: 0
Note that Dense layers often have a lot of parameters. For example, the first hidden
layer has 784 × 300 connection weights, plus 300 bias terms, which adds up to
235,500 parameters! This gives the model quite a lot of flexibility to fit the training
data, but it also means that the model runs the risk of overfitting, especially when you
do not have a lot of training data. We will come back to this later.
You can easily get a model’s list of layers, to fetch a layer by its index, or you can fetch
it by name:
>>> model.layers
[<tensorflow.python.keras.layers.core.Flatten at 0x132414e48>,
<tensorflow.python.keras.layers.core.Dense at 0x1324149b0>,
<tensorflow.python.keras.layers.core.Dense at 0x1356ba8d0>,
<tensorflow.python.keras.layers.core.Dense at 0x13240d240>]
>>> model.layers[1].name
'dense_3'
>>> model.get_layer('dense_3').name
'dense_3'
All the parameters of a layer can be accessed using its get_weights() and
set_weights() method. For a Dense layer, this includes both the connection weights
and the bias terms:
```

```
Implementing MLPs with Keras | 297
```

```
>>> weights, biases = hidden1.get_weights()
>>> weights
array([[ 0.03854964, -0.04054524, 0.00599282, ..., 0.02566582,
0.01032123, 0.06914985],
...,
[ 0.02632413, -0.05105981, -0.00332005, ..., 0.04175945,
0.0443138 , -0.05558084]], dtype=float32)
>>> weights.shape
(784, 300)
>>> biases
array([0., 0., 0., 0., 0., 0., 0., 0., 0., ..., 0., 0., 0.], dtype=float32)
>>> biases.shape
(300,)
```

Notice that the Dense layer initialized the connection weights randomly (which is
needed to break symmetry, as we discussed earlier), and the biases were just initial‐
ized to zeros, which is fine. If you ever want to use a different initialization method,
you can set kernel_initializer (kernel is another name for the matrix of connec‐
tion weights) or bias_initializer when creating the layer. We will discuss initializ‐
ers further in Chapter 11, but if you want the full list, see https://keras.io/initializers/.

```
The shape of the weight matrix depends on the number of inputs.
This is why it is recommended to specify the input_shape when
creating the first layer in a Sequential model. However, if you do
not specify the input shape, it’s okay: Keras will simply wait until it
knows the input shape before it actually builds the model. This will
happen either when you feed it actual data (e.g., during training),
or when you call its build() method. Until the model is really
built, the layers will not have any weights, and you will not be able
to do certain things (such as print the model summary or save the
model), so if you know the input shape when creating the model, it
is best to specify it.
```

**Compiling the Model**

After a model is created, you must call its compile() method to specify the loss func‐
tion and the optimizer to use. Optionally, you can also specify a list of extra metrics to
compute during training and evaluation:

```
model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd",
metrics=["accuracy"])
```

**298 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

```
Using loss="sparse_categorical_crossentropy" is equivalent to
loss=keras.losses.sparse_categorical_crossentropy. Simi‐
larly, optimizer="sgd" is equivalent to optimizer=keras.optimiz
ers.SGD() and metrics=["accuracy"] is equivalent to
metrics=[keras.metrics.sparse_categorical_accuracy] (when
using this loss). We will use many other losses, optimizers and met‐
rics in this book, but for the full lists see https://keras.io/losses/,
https://keras.io/optimizers/ and https://keras.io/metrics/.
```

This requires some explanation. First, we use the "sparse_categorical_crossen
tropy" loss because we have sparse labels (i.e., for each instance there is just a target
class index, from 0 to 9 in this case), and the classes are exclusive. If instead we had
one target probability per class for each instance (such as one-hot vectors, e.g. [0.,
0., 0., 1., 0., 0., 0., 0., 0., 0.] to represent class 3), then we would need
to use the "categorical_crossentropy" loss instead. If we were doing binary classi‐
fication (with one or more binary labels), then we would use the "sigmoid" (i.e.,
logistic) activation function in the output layer instead of the "softmax" activation
function, and we would use the "binary_crossentropy" loss.

```
If you want to convert sparse labels (i.e., class indices) to one-hot
vector labels, you can use the keras.utils.to_categorical()
function. To go the other way round, you can just use the np.arg
max() function with axis=1.
```

Secondly, regarding the optimizer, "sgd" simply means that we will train the model
using simple Stochastic Gradient Descent. In other words, Keras will perform the
backpropagation algorithm described earlier (i.e., reverse-mode autodiff + Gradient
Descent). We will discuss more efficient optimizers in Chapter 11 (they improve the
Gradient Descent part, not the autodiff ).

Finally, since this is a classifier, it’s useful to measure its "accuracy" during training
and evaluation.

**Training and Evaluating the Model**

Now the model is ready to be trained. For this we simply need to call its fit()
method. We pass it the input features (X_train) and the target classes (y_train), as
well as the number of epochs to train (or else it would default to just 1, which would
definitely not be enough to converge to a good solution). We also pass a validation set
(this is optional): Keras will measure the loss and the extra metrics on this set at the
end of each epoch, which is very useful to see how well the model really performs: if
the performance on the training set is much better than on the validation set, your

```
Implementing MLPs with Keras | 299
```

model is probably overfitting the training set (or there is a bug, such as a data mis‐
match between the training set and the validation set):

```
>>> history = model.fit(X_train, y_train, epochs=30,
... validation_data=(X_valid, y_valid))
...
Train on 55000 samples, validate on 5000 samples
Epoch 1/30
55000/55000 [==========] - 3s 55us/sample - loss: 1.4948 - acc: 0.5757
```

- val_loss: 1.0042 - val_acc: 0.7166
  Epoch 2/30
  55000/55000 [==========] - 3s 55us/sample - loss: 0.8690 - acc: 0.7318
- val_loss: 0.7549 - val_acc: 0.7616
  [...]
  Epoch 50/50
  55000/55000 [==========] - 4s 72us/sample - loss: 0.3607 - acc: 0.8752
- val_loss: 0.3706 - val_acc: 0.8728

And that’s it! The neural network is trained. At each epoch during training, Keras dis‐
plays the number of instances processed so far (along with a progress bar), the mean
training time per sample, the loss and accuracy (or any other extra metrics you asked
for), both on the training set and the validation set. You can see that the training loss
went down, which is a good sign, and the validation accuracy reached 87.28% after 50
epochs, not too far from the training accuracy, so there does not seem to be much
overfitting going on.

```
Instead of passing a validation set using the validation_data
argument, you could instead set validation_split to the ratio of
the training set that you want Keras to use for validation (e.g., 0.1).
```

If the training set was very skewed, with some classes being overrepresented and oth‐
ers underrepresented, it would be useful to set the class_weight argument when
calling the fit() method, giving a larger weight to underrepresented classes, and a
lower weight to overrepresented classes. These weights would be used by Keras when
computing the loss. If you need per-instance weights instead, you can set the sam
ple_weight argument (it supersedes class_weight). This could be useful for exam‐
ple if some instances were labeled by experts while others were labeled using a
crowdsourcing platform: you might want to give more weight to the former. You can
also provide sample weights (but not class weights) for the validation set by adding
them as a third item in the validation_data tuple.

The fit() method returns a History object containing the training parameters (his
tory.params), the list of epochs it went through (history.epoch), and most impor‐
tantly a dictionary (history.history) containing the loss and extra metrics it
measured at the end of each epoch on the training set and on the validation set (if

**300 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

any). If you create a Pandas DataFrame using this dictionary and call its plot()
method, you get the learning curves shown in Figure 10-12:

```
import pandas as pd
```

```
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
```

Figure 10-12. Learning Curves

You can see that both the training and validation accuracy steadily increase during
training, while the training and validation loss decrease. Good! Moreover, the valida‐
tion curves are quite close to the training curves, which means that there is not too
much overfitting. In this particular case, the model performed better on the valida‐
tion set than on the training set at the beginning of training: this sometimes happens
by chance (especially when the validation set is fairly small). However, the training set
performance ends up beating the validation performance, as is generally the case
when you train for long enough. You can tell that the model has not quite converged
yet, as the validation loss is still going down, so you should probably continue train‐
ing. It’s as simple as calling the fit() method again, since Keras just continues train‐
ing where it left off (you should be able to reach close to 89% validation accuracy).

If you are not satisfied with the performance of your model, you should go back and
tune the model’s hyperparameters, for example the number of layers, the number of
neurons per layer, the types of activation functions we use for each hidden layer, the

```
Implementing MLPs with Keras | 301
```

number of training epochs, the batch size (it can be set in the fit() method using the
batch_size argument, which defaults to 32). We will get back to hyperparameter
tuning at the end of this chapter. Once you are satisfied with your model’s validation
accuracy, you should evaluate it on the test set to estimate the generalization error
before you deploy the model to production. You can easily do this using the evalu
ate() method (it also supports several other arguments, such as batch_size or sam
ple_weight, please check the documentation for more details):

```
>>> model.evaluate(X_test, y_test)
8832/10000 [==========================] - ETA: 0s - loss: 0.4074 - acc: 0.8540
[0.40738476498126985, 0.854]
```

As we saw in Chapter 2, it is common to get slightly lower performance on the test set
than on the validation set, because the hyperparameters are tuned on the validation
set, not the test set (however, in this example, we did not do any hyperparameter tun‐
ing, so the lower accuracy is just bad luck). Remember to resist the temptation to
tweak the hyperparameters on the test set, or else your estimate of the generalization
error will be too optimistic.

**Using the Model to Make Predictions**

Next, we can use the model’s predict() method to make predictions on new instan‐
ces. Since we don’t have actual new instances, we will just use the first 3 instances of
the test set:

```
>>> X_new = X_test[:3]
>>> y_proba = model.predict(X_new)
>>> y_proba.round(2)
array([[0. , 0. , 0. , 0. , 0. , 0.09, 0. , 0.12, 0. , 0.79],
[0. , 0. , 0.94, 0. , 0.02, 0. , 0.04, 0. , 0. , 0. ],
[0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]],
dtype=float32)
```

As you can see, for each instance the model estimates one probability per class, from
class 0 to class 9. For example, for the first image it estimates that the probability of
class 9 (ankle boot) is 79%, the probability of class 7 (sneaker) is 12%, the probability
of class 5 (sandal) is 9%, and the other classes are negligible. In other words, it
“believes” it’s footwear, probably ankle boots, but it’s not entirely sure, it might be
sneakers or sandals instead. If you only care about the class with the highest estima‐
ted probability (even if that probability is quite low) then you can use the pre
dict_classes() method instead:

```
>>> y_pred = model.predict_classes(X_new)
>>> y_pred
array([9, 2, 1])
>>> np.array(class_names)[y_pred]
array(['Ankle boot', 'Pullover', 'Trouser'], dtype='<U11')
```

And the classifier actually classified all three images correctly:

**302 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

```
>>> y_new = y_test[:3]
>>> y_new
array([9, 2, 1])
```

Now you know how to build, train, evaluate and use a classification MLP using the
Sequential API. But what about regression?

##### Building a Regression MLP Using the Sequential API

Let’s switch to the California housing problem and tackle it using a regression neural
network. For simplicity, we will use Scikit-Learn’s fetch_california_housing()
function to load the data: this dataset is simpler than the one we used in Chapter 2,
since it contains only numerical features (there is no ocean_proximity feature), and
there is no missing value. After loading the data, we split it into a training set, a vali‐
dation set and a test set, and we scale all the features:

```
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

```
housing = fetch_california_housing()
```

```
X_train_full, X_test, y_train_full, y_test = train_test_split(
housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
X_train_full, y_train_full)
```

```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
```

Building, training, evaluating and using a regression MLP using the Sequential API to
make predictions is quite similar to what we did for classification. The main differ‐
ences are the fact that the output layer has a single neuron (since we only want to
predict a single value) and uses no activation function, and the loss function is the
mean squared error. Since the dataset is quite noisy, we just use a single hidden layer
with fewer neurons than before, to avoid overfitting:

```
model = keras.models.Sequential([
keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=20,
validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3] # pretend these are new instances
y_pred = model.predict(X_new)
```

```
Implementing MLPs with Keras | 303
```

14 “Wide & Deep Learning for Recommender Systems,” Heng-Tze Cheng et al. (2016).

```
As you can see, the Sequential API is quite easy to use. However, although sequential
models are extremely common, it is sometimes useful to build neural networks with
more complex topologies, or with multiple inputs or outputs. For this purpose, Keras
offers the Functional API.
```

##### Building Complex Models Using the Functional API

```
One example of a non-sequential neural network is a Wide & Deep neural network.
This neural network architecture was introduced in a 2016 paper by Heng-Tze Cheng
et al.^14. It connects all or part of the inputs directly to the output layer, as shown in
Figure 10-13. This architecture makes it possible for the neural network to learn both
deep patterns (using the deep path) and simple rules (through the short path). In
contrast, a regular MLP forces all the data to flow through the full stack of layers, thus
simple patterns in the data may end up being distorted by this sequence of transfor‐
mations.
```

```
304 | Chapter 10: Introduction to Artificial Neural Networks with Keras
```

Figure 10-13. Wide and Deep Neural Network

Let’s build such a neural network to tackle the California housing problem:

```
input = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()[input, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input], outputs=[output])
```

Let’s go through each line of this code:

- First, we need to create an Input object. This is needed because we may have
  multiple inputs, as we will see later.
- Next, we create a Dense layer with 30 neurons and using the ReLU activation
  function. As soon as it is created, notice that we call it like a function, passing it
  the input. This is why this is called the Functional API. Note that we are just tell‐

```
Implementing MLPs with Keras | 305
```

```
ing Keras how it should connect the layers together, no actual data is being pro‐
cessed yet.
```

- We then create a second hidden layer, and again we use it as a function. Note
  however that we pass it the output of the first hidden layer.
- Next, we create a Concatenate() layer, and once again we immediately use it like
  a function, to concatenate the input and the output of the second hidden layer
  (you may prefer the keras.layers.concatenate() function, which creates a Con
  catenate layer and immediately calls it with the given inputs).
- Then we create the output layer, with a single neuron and no activation function,
  and we call it like a function, passing it the result of the concatenation.
- Lastly, we create a Keras Model, specifying which inputs and outputs to use.

Once you have built the Keras model, everything is exactly like earlier, so no need to
repeat it here: you must compile the model, train it, evaluate it and use it to make
predictions.

But what if you want to send a subset of the features through the wide path, and a
different subset (possibly overlapping) through the deep path (see Figure 10-14)? In
this case, one solution is to use multiple inputs. For example, suppose we want to
send 5 features through the deep path (features 0 to 4), and 6 features through the
wide path (features 2 to 7):

```
input_A = keras.layers.Input(shape=[5])
input_B = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
```

**306 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

Figure 10-14. Handling Multiple Inputs

The code is self-explanatory. Note that we specified inputs=[input_A, input_B]
when creating the model. Now we can compile the model as usual, but when we call
the fit() method, instead of passing a single input matrix X_train, we must pass a
pair of matrices (X_train_A, X_train_B): one per input. The same is true for
X_valid, and also for X_test and X_new when you call evaluate() or predict():

```
model.compile(loss="mse", optimizer="sgd")
```

```
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
```

```
history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))
```

```
Implementing MLPs with Keras | 307
```

There are also many use cases in which you may want to have multiple outputs:

- The task may demand it, for example you may want to locate and classify the
  main object in a picture. This is both a regression task (finding the coordinates of
  the object’s center, as well as its width and height) and a classification task.
- Similarly, you may have multiple independent tasks to perform based on the
  same data. Sure, you could train one neural network per task, but in many cases
  you will get better results on all tasks by training a single neural network with
  one output per task. This is because the neural network can learn features in the
  data that are useful across tasks.
- Another use case is as a regularization technique (i.e., a training constraint whose
  objective is to reduce overfitting and thus improve the model’s ability to general‐
  ize). For example, you may want to add some auxiliary outputs in a neural net‐
  work architecture (see Figure 10-15) to ensure that the underlying part of the
  network learns something useful on its own, without relying on the rest of the
  network.

Figure 10-15. Handling Multiple Outputs – Auxiliary Output for Regularization

Adding extra outputs is quite easy: just connect them to the appropriate layers and
add them to your model’s list of outputs. For example, the following code builds the
network represented in Figure 10-15:

**308 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

```
[...] # Same as above, up to the main output layer
output = keras.layers.Dense(1)(concat)
aux_output = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs=[input_A, input_B],
outputs=[output, aux_output])
```

Each output will need its own loss function, so when we compile the model we
should pass a list of losses (if we pass a single loss, Keras will assume that the same
loss must be used for all outputs). By default, Keras will compute all these losses and
simply add them up to get the final loss used for training. However, we care much
more about the main output than about the auxiliary output (as it is just used for reg‐
ularization), so we want to give the main output’s loss a much greater weight. Fortu‐
nately, it is possible to set all the loss weights when compiling the model:

```
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer="sgd")
```

Now when we train the model, we need to provide some labels for each output. In
this example, the main output and the auxiliary output should try to predict the same
thing, so they should use the same labels. So instead of passing y_train, we just need
to pass (y_train, y_train) (and the same goes for y_valid and y_test):

```
history = model.fit(
[X_train_A, X_train_B], [y_train, y_train], epochs=20,
validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
```

When we evaluate the model, Keras will return the total loss, as well as all the individ‐
ual losses:

```
total_loss, main_loss, aux_loss = model.evaluate(
[X_test_A, X_test_B], [y_test, y_test])
```

Similarly, the predict() method will return predictions for each output:

```
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
```

As you can see, you can build any sort of architecture you want quite easily with the
Functional API. Let’s look at one last way you can build Keras models.

##### Building Dynamic Models Using the Subclassing API

Both the Sequential API and the Functional API are declarative: you start by declar‐
ing which layers you want to use and how they should be connected, and only then
can you start feeding the model some data for training or inference. This has many
advantages: the model can easily be saved, cloned, shared, its structure can be dis‐
played and analyzed, the framework can infer shapes and check types, so errors can
be caught early (i.e., before any data ever goes through the model). It’s also fairly easy
to debug, since the whole model is just a static graph of layers. But the flip side is just
that: it’s static. Some models involve loops, varying shapes, conditional branching,
and other dynamic behaviors. For such cases, or simply if you prefer a more impera‐
tive programming style, the Subclassing API is for you.

```
Implementing MLPs with Keras | 309
```

15 Keras models have an output attribute, so we cannot use that name for the main output layer, which is why
we renamed it to main_output.

```
Simply subclass the Model class, create the layers you need in the constructor, and use
them to perform the computations you want in the call() method. For example, cre‐
ating an instance of the following WideAndDeepModel class gives us an equivalent
model to the one we just built with the Functional API. You can then compile it, eval‐
uate it and use it to make predictions, exactly like we just did.
class WideAndDeepModel (keras.models.Model):
def __init__ (self, units=30, activation="relu", **kwargs):
super(). __init__ (**kwargs) # handles standard args (e.g., name)
self.hidden1 = keras.layers.Dense(units, activation=activation)
self.hidden2 = keras.layers.Dense(units, activation=activation)
self.main_output = keras.layers.Dense(1)
self.aux_output = keras.layers.Dense(1)
```

```
def call(self, inputs):
input_A, input_B = inputs
hidden1 = self.hidden1(input_B)
hidden2 = self.hidden2(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
main_output = self.main_output(concat)
aux_output = self.aux_output(hidden2)
return main_output, aux_output
```

```
model = WideAndDeepModel()
This example looks very much like the Functional API, except we do not need to cre‐
ate the inputs, we just use the input argument to the call() method, and we separate
the creation of the layers^15 in the constructor from their usage in the call() method.
However, the big difference is that you can do pretty much anything you want in the
call() method: for loops, if statements, low-level TensorFlow operations, your
imagination is the limit (see Chapter 12)! This makes it a great API for researchers
experimenting with new ideas.
However, this extra flexibility comes at a cost: your model’s architecture is hidden
within the call() method, so Keras cannot easily inspect it, it cannot save or clone it,
and when you call the summary() method, you only get a list of layers, without any
information on how they are connected to each other. Moreover, Keras cannot check
types and shapes ahead of time, and it is easier to make mistakes. So unless you really
need that extra flexibility, you should probably stick to the Sequential API or the
Functional API.
```

```
310 | Chapter 10: Introduction to Artificial Neural Networks with Keras
```

```
Keras models can be used just like regular layers, so you can easily
compose them to build complex architectures.
```

Now that you know how to build and train neural nets using Keras, you will want to
save them!

##### Saving and Restoring a Model

Saving a trained Keras model is as simple as it gets:

```
model.save("my_keras_model.h5")
```

Keras will save both the model’s architecture (including every layer’s hyperparame‐
ters) and the value of all the model parameters for every layer (e.g., connection
weights and biases), using the HDF5 format. It also saves the optimizer (including its
hyperparameters and any state it may have).

You will typically have a script that trains a model and saves it, and one or more
scripts (or web services) that load the model and use it to make predictions. Loading
the model is just as easy:

```
model = keras.models.load_model("my_keras_model.h5")
```

```
This will work when using the Sequential API or the Functional
API, but unfortunately not when using Model subclassing. How‐
ever, you can use save_weights() and load_weights() to at least
save and restore the model parameters (but you will need to save
and restore everything else yourself ).
```

But what if training lasts several hours? This is quite common, especially when train‐
ing on large datasets. In this case, you should not only save your model at the end of
training, but also save checkpoints at regular intervals during training. But how can
you tell the fit() method to save checkpoints? The answer is: using callbacks.

##### Using Callbacks

The fit() method accepts a callbacks argument that lets you specify a list of objects
that Keras will call during training at the start and end of training, at the start and end
of each epoch and even before and after processing each batch. For example, the Mod
elCheckpoint callback saves checkpoints of your model at regular intervals during
training, by default at the end of each epoch:

```
Implementing MLPs with Keras | 311
```

```
[...] # build and compile the model
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])
```

Moreover, if you use a validation set during training, you can set
save_best_only=True when creating the ModelCheckpoint. In this case, it will only
save your model when its performance on the validation set is the best so far. This
way, you do not need to worry about training for too long and overfitting the training
set: simply restore the last model saved after training, and this will be the best model
on the validation set. This is a simple way to implement early stopping (introduced in
Chapter 4):

```
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
validation_data=(X_valid, y_valid),
callbacks=[checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5") # rollback to best model
```

Another way to implement early stopping is to simply use the EarlyStopping call‐
back. It will interrupt training when it measures no progress on the validation set for
a number of epochs (defined by the patience argument), and it will optionally roll
back to the best model. You can combine both callbacks to both save checkpoints of
your model (in case your computer crashes), and actually interrupt training early
when there is no more progress (to avoid wasting time and resources):

```
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[checkpoint_cb, early_stopping_cb])
```

The number of epochs can be set to a large value since training will stop automati‐
cally when there is no more progress. Moreover, there is no need to restore the best
model saved in this case since the EarlyStopping callback will keep track of the best
weights and restore them for us at the end of training.

```
There are many other callbacks available in the keras.callbacks
package. See https://keras.io/callbacks/.
```

If you need extra control, you can easily write your own custom callbacks. For exam‐
ple, the following custom callback will display the ratio between the validation loss
and the training loss during training (e.g., to detect overfitting):

**312 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

```
class PrintValTrainRatioCallback (keras.callbacks.Callback):
def on_epoch_end(self, epoch, logs):
print (" \n val/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
```

As you might expect, you can implement on_train_begin(), on_train_end(),
on_epoch_begin(), on_epoch_begin(), on_batch_end() and on_batch_end().
Moreover, callbacks can also be used during evaluation and predictions, should you
ever need them (e.g., for debugging). In this case, you should implement
on_test_begin(), on_test_end(), on_test_batch_begin(), or
on_test_batch_end() (called by evaluate()), or on_predict_begin(), on_pre
dict_end(), on_predict_batch_begin(), or on_predict_batch_end() (called by
predict()).

Now let’s take a look at one more tool you should definitely have in your toolbox
when using tf.keras: TensorBoard.

#### Visualization Using TensorBoard

TensorBoard is a great interactive visualization tool that you can use to view the
learning curves during training, compare learning curves between multiple runs, vis‐
ualize the computation graph, analyze training statistics, view images generated by
your model, visualize complex multidimensional data projected down to 3D and
automatically clustered for you, and more! This tool is installed automatically when
you install TensorFlow, so you already have it!

To use it, you must modify your program so that it outputs the data you want to visu‐
alize to special binary log files called event files. Each binary data record is called a
summary. The TensorBoard server will monitor the log directory, and it will automat‐
ically pick up the changes and update the visualizations: this allows you to visualize
live data (with a short delay), such as the learning curves during training. In general,
you want to point the TensorBoard server to a root log directory, and configure your
program so that it writes to a different subdirectory every time it runs. This way, the
same TensorBoard server instance will allow you to visualize and compare data from
multiple runs of your program, without getting everything mixed up.

So let’s start by defining the root log directory we will use for our TensorBoard logs,
plus a small function that will generate a subdirectory path based on the current date
and time, so that it is different at every run. You may want to include extra informa‐
tion in the log directory name, such as hyperparameter values that you are testing, to
make it easier to know what you are looking at in TensorBoard:

```
root_logdir = os.path.join(os.curdir, "my_logs")
```

```
def get_run_logdir():
import time
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
return os.path.join(root_logdir, run_id)
```

```
Implementing MLPs with Keras | 313
```

```
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_01_16-11_28_43'
```

Next, the good news is that Keras provides a nice TensorBoard callback:

```
[...] # Build and compile your model
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
validation_data=(X_valid, y_valid),
callbacks=[tensorboard_cb])
```

And that’s all there is to it! It could hardly be easier to use. If you run this code, the
TensorBoard callback will take care of creating the log directory for you (along with
its parent directories if needed), and during training it will create event files and write
summaries to them. After running the program a second time (perhaps changing
some hyperparameter value), you will end up with a directory structure similar to
this one:

```
my_logs
├── run_2019_01_16-16_51_02
│ └── events.out.tfevents.1547628669.mycomputer.local.v2
└── run_2019_01_16-16_56_50
└── events.out.tfevents.1547629020.mycomputer.local.v2
```

Next you need to start the TensorBoard server. If you installed TensorFlow within a
virtualenv, you should activate it. Next, run the following command at the root of the
project (or from anywhere else as long as you point to the appropriate log directory).
If your shell cannot find the tensorboard script, then you must update your PATH
environment variable so that it contains the directory in which the script was
installed (alternatively, you can just replace tensorboard with python3 -m tensor
board.main).

```
$ tensorboard --logdir=./my_logs --port=6006
TensorBoard 2.0.0 at http://mycomputer.local:6006 (Press CTRL+C to quit)
```

Finally, open up a web browser to [http://localhost:6006.](http://localhost:6006.) You should see TensorBoard’s
web interface. Click on the SCALARS tab to view the learning curves (see
Figure 10-16). Notice that the training loss went down nicely during both runs, but
the second run went down much faster. Indeed, we used a larger learning rate by set‐
ting optimizer=keras.optimizers.SGD(lr=0.05) instead of optimizer="sgd",
which defaults to a learning rate of 0.001.

**314 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

Figure 10-16. Visualizing Learning Curves with TensorBoard

Unfortunately, at the time of writing, no other data is exported by the TensorBoard
callback, but this issue will probably be fixed by the time you read these lines. In Ten‐
sorFlow 1, this callback exported the computation graph and many useful statistics:
type help(keras.callbacks.TensorBoard) to see all the options.

Let’s summarize what you learned so far in this chapter: we saw where neural nets
came from, what an MLP is and how you can use it for classification and regression,
how to build MLPs using tf.keras’s Sequential API, or more complex architectures
using the Functional API or Model Subclassing, you learned how to save and restore a
model, use callbacks for checkpointing, early stopping, and more, and finally how to
use TensorBoard for visualization. You can already go ahead and use neural networks
to tackle many problems! However, you may wonder how to choose the number of
hidden layers, the number of neurons in the network, and all the other hyperparame‐
ters. Let’s look at this now.

### Fine-Tuning Neural Network Hyperparameters

The flexibility of neural networks is also one of their main drawbacks: there are many
hyperparameters to tweak. Not only can you use any imaginable network architec‐
ture, but even in a simple MLP you can change the number of layers, the number of
neurons per layer, the type of activation function to use in each layer, the weight initi‐

```
Fine-Tuning Neural Network Hyperparameters | 315
```

alization logic, and much more. How do you know what combination of hyperpara‐
meters is the best for your task?

One option is to simply try many combinations of hyperparameters and see which
one works best on the validation set (or using K-fold cross-validation). For this, one
approach is simply use GridSearchCV or RandomizedSearchCV to explore the hyper‐
parameter space, as we did in Chapter 2. For this, we need to wrap our Keras models
in objects that mimic regular Scikit-Learn regressors. The first step is to create a func‐
tion that will build and compile a Keras model, given a set of hyperparameters:

```
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
model = keras.models.Sequential()
options = {"input_shape": input_shape}
for layer in range(n_hidden):
model.add(keras.layers.Dense(n_neurons, activation="relu", **options))
options = {}
model.add(keras.layers.Dense(1, **options))
optimizer = keras.optimizers.SGD(learning_rate)
model.compile(loss="mse", optimizer=optimizer)
return model
```

This function creates a simple Sequential model for univariate regression (only one
output neuron), with the given input shape and the given number of hidden layers
and neurons, and it compiles it using an SGD optimizer configured with the given
learning rate. The options dict is used to ensure that the first layer is properly given
the input shape (note that if n_hidden=0, the first layer will be the output layer). It is
good practice to provide reasonable defaults to as many hyperparameters as you can,
as Scikit-Learn does.

Next, let’s create a KerasRegressor based on this build_model() function:

```
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
```

The KerasRegressor object is a thin wrapper around the Keras model built using
build_model(). Since we did not specify any hyperparameter when creating it, it will
just use the default hyperparameters we defined in build_model(). Now we can use
this object like a regular Scikit-Learn regressor: we can train it using its fit()
method, then evaluate it using its score() method, and use it to make predictions
using its predict() method. Note that any extra parameter you pass to the fit()
method will simply get passed to the underlying Keras model. Also note that the
score will be the opposite of the MSE because Scikit-Learn wants scores, not losses
(i.e., higher should be better).

```
keras_reg.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)
```

**316 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

However, we do not actually want to train and evaluate a single model like this, we
want to train hundreds of variants and see which one performs best on the validation
set. Since there are many hyperparameters, it is preferable to use a randomized search
rather than grid search (as we discussed in Chapter 2). Let’s try to explore the number
of hidden layers, the number of neurons and the learning rate:

```
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
```

```
param_distribs = {
"n_hidden": [0, 1, 2, 3],
"n_neurons": np.arange(1, 100),
"learning_rate": reciprocal(3e-4, 3e-2),
}
```

```
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[keras.callbacks.EarlyStopping(patience=10)])
```

As you can see, this is identical to what we did in Chapter 2, with the exception that
we pass extra parameters to the fit() method: they simply get relayed to the under‐
lying Keras models. Note that RandomizedSearchCV uses K-fold cross-validation, so it
does not use X_valid and y_valid. These are just used for early stopping.

The exploration may last many hours depending on the hardware, the size of the
dataset, the complexity of the model and the value of n_iter and cv. When it is over,
you can access the best parameters found, the best score, and the trained Keras model
like this:

```
>>> rnd_search_cv.best_params_
{'learning_rate': 0.0033625641252688094, 'n_hidden': 2, 'n_neurons': 42}
>>> rnd_search_cv.best_score_
-0.3189529188278931
>>> model = rnd_search_cv.best_estimator_.model
```

You can now save this model, evaluate it on the test set, and if you are satisfied with
its performance, deploy it to production. Using randomized search is not too hard,
and it works well for many fairly simple problems. However, when training is slow
(e.g., for more complex problems with larger datasets), this approach will only
explore a tiny portion of the hyperparameter space. You can partially alleviate this
problem by assisting the search process manually: first run a quick random search
using wide ranges of hyperparameter values, then run another search using smaller
ranges of values centered on the best ones found during the first run, and so on. This
will hopefully zoom in to a good set of hyperparameters. However, this is very time
consuming, and probably not the best use of your time.

Fortunately, there are many techniques to explore a search space much more effi‐
ciently than randomly. Their core idea is simple: when a region of the space turns out

```
Fine-Tuning Neural Network Hyperparameters | 317
```

16 “Population Based Training of Neural Networks,” Max Jaderberg et al. (2017).

```
to be good, it should be explored more. This takes care of the “zooming” process for
you and leads to much better solutions in much less time. Here are a few Python
libraries you can use to optimize hyperparameters:
```

- Hyperopt: a popular Python library for optimizing over all sorts of complex
  search spaces (including real values such as the learning rate, or discrete values
  such as the number of layers).
- Hyperas, kopt or Talos: optimizing hyperparameters for Keras model (the first
  two are based on Hyperopt).
- Scikit-Optimize (skopt): a general-purpose optimization library. The Bayes
  SearchCV class performs Bayesian optimization using an interface similar to Grid
  SearchCV.
- Spearmint: a Bayesian optimization library.
- Sklearn-Deap: a hyperparameter optimization library based on evolutionary
  algorithms, also with a GridSearchCV-like interface.
- And many more!

```
Moreover, many companies offer services for hyperparameter optimization. For
example Google Cloud ML Engine has a hyperparameter tuning service. Other com‐
panies provide APIs for hyperparameter optimization, such as Arimo, SigOpt, Oscar
and many more.
Hyperparameter tuning is still an active area of research. Evolutionary algorithms are
making a comeback lately. For example, check out DeepMind’s excellent 2017 paper^16 ,
where they jointly optimize a population of models and their hyperparameters. Goo‐
gle also used an evolutionary approach, not just to search for hyperparameters, but
also to look for the best neural network architecture for the problem. They call this
AutoML, and it is already available as a cloud service. Perhaps the days of building
neural networks manually will soon be over? Check out Google’s post on this topic. In
fact, evolutionary algorithms have also been used successfully to train individual neu‐
ral networks, replacing the ubiquitous Gradient Descent! See this 2017 post by Uber
where they introduce their Deep Neuroevolution technique.
Despite all this exciting progress, and all these tools and services, it still helps to have
an idea of what values are reasonable for each hyperparameter, so you can build a
quick prototype, and restrict the search space. Here are a few guidelines for choosing
the number of hidden layers and neurons in an MLP, and selecting good values for
some of the main hyperparameters.
```

```
318 | Chapter 10: Introduction to Artificial Neural Networks with Keras
```

#### Number of Hidden Layers

For many problems, you can just begin with a single hidden layer and you will get
reasonable results. It has actually been shown that an MLP with just one hidden layer
can model even the most complex functions provided it has enough neurons. For a
long time, these facts convinced researchers that there was no need to investigate any
deeper neural networks. But they overlooked the fact that deep networks have a much
higher parameter efficiency than shallow ones: they can model complex functions
using exponentially fewer neurons than shallow nets, allowing them to reach much
better performance with the same amount of training data.

To understand why, suppose you are asked to draw a forest using some drawing soft‐
ware, but you are forbidden to use copy/paste. You would have to draw each tree
individually, branch per branch, leaf per leaf. If you could instead draw one leaf,
copy/paste it to draw a branch, then copy/paste that branch to create a tree, and
finally copy/paste this tree to make a forest, you would be finished in no time. Real-
world data is often structured in such a hierarchical way and Deep Neural Networks
automatically take advantage of this fact: lower hidden layers model low-level struc‐
tures (e.g., line segments of various shapes and orientations), intermediate hidden
layers combine these low-level structures to model intermediate-level structures (e.g.,
squares, circles), and the highest hidden layers and the output layer combine these
intermediate structures to model high-level structures (e.g., faces).

Not only does this hierarchical architecture help DNNs converge faster to a good sol‐
ution, it also improves their ability to generalize to new datasets. For example, if you
have already trained a model to recognize faces in pictures, and you now want to
train a new neural network to recognize hairstyles, then you can kickstart training by
reusing the lower layers of the first network. Instead of randomly initializing the
weights and biases of the first few layers of the new neural network, you can initialize
them to the value of the weights and biases of the lower layers of the first network.
This way the network will not have to learn from scratch all the low-level structures
that occur in most pictures; it will only have to learn the higher-level structures (e.g.,
hairstyles). This is called transfer learning.

In summary, for many problems you can start with just one or two hidden layers and
it will work just fine (e.g., you can easily reach above 97% accuracy on the MNIST
dataset using just one hidden layer with a few hundred neurons, and above 98% accu‐
racy using two hidden layers with the same total amount of neurons, in roughly the
same amount of training time). For more complex problems, you can gradually ramp
up the number of hidden layers, until you start overfitting the training set. Very com‐
plex tasks, such as large image classification or speech recognition, typically require
networks with dozens of layers (or even hundreds, but not fully connected ones, as
we will see in Chapter 14), and they need a huge amount of training data. However,
you will rarely have to train such networks from scratch: it is much more common to

```
Fine-Tuning Neural Network Hyperparameters | 319
```

17 By Vincent Vanhoucke in his Deep Learning class on Udacity.com.

```
reuse parts of a pretrained state-of-the-art network that performs a similar task.
Training will be a lot faster and require much less data (we will discuss this in Chap‐
ter 11).
```

#### Number of Neurons per Hidden Layer

```
Obviously the number of neurons in the input and output layers is determined by the
type of input and output your task requires. For example, the MNIST task requires 28
x 28 = 784 input neurons and 10 output neurons.
As for the hidden layers, it used to be a common practice to size them to form a pyra‐
mid, with fewer and fewer neurons at each layer—the rationale being that many low-
level features can coalesce into far fewer high-level features. For example, a typical
neural network for MNIST may have three hidden layers, the first with 300 neurons,
the second with 200, and the third with 100. However, this practice has been largely
abandoned now, as it seems that simply using the same number of neurons in all hid‐
den layers performs just as well in most cases, or even better, and there is just one
hyperparameter to tune instead of one per layer—for example, all hidden layers could
simply have 150 neurons. However, depending on the dataset, it can sometimes help
to make the first hidden layer bigger than the others.
Just like for the number of layers, you can try increasing the number of neurons grad‐
ually until the network starts overfitting. In general you will get more bang for the
buck by increasing the number of layers than the number of neurons per layer.
Unfortunately, as you can see, finding the perfect amount of neurons is still somewhat
of a dark art.
A simpler approach is to pick a model with more layers and neurons than you
actually need, then use early stopping to prevent it from overfitting (and other regu‐
larization techniques, such as dropout, as we will see in Chapter 11). This has been
dubbed the “stretch pants” approach:^17 instead of wasting time looking for pants that
perfectly match your size, just use large stretch pants that will shrink down to the
right size.
```

#### Learning Rate, Batch Size and Other Hyperparameters

```
The number of hidden layers and neurons are not the only hyperparameters you can
tweak in an MLP. Here are some of the most important ones, and some tips on how
to set them:
```

- The learning rate is arguably the most important hyperparameter. In general, the
  optimal learning rate is about half of the maximum learning rate (i.e., the learn‐

```
320 | Chapter 10: Introduction to Artificial Neural Networks with Keras
```

18 “Practical recommendations for gradient-based training of deep architectures,” Yoshua Bengio (2012).

```
ing rate above which the training algorithm diverges, as we saw in Chapter 4). So
a simple approach for tuning the learning rate is to start with a large value that
makes the training algorithm diverge, then divide this value by 3 and try again,
and repeat until the training algorithm stops diverging. At that point, you gener‐
ally won’t be too far from the optimal learning rate. That said, it is sometimes
useful to reduce the learning rate during training: we will discuss this in Chap‐
ter 11.
```

- Choosing a better optimizer than plain old Mini-batch Gradient Descent (and
  tuning its hyperparameters) is also quite important. We will discuss this in Chap‐
  ter 11.
- The batch size can also have a significant impact on your model’s performance
  and the training time. In general the optimal batch size will be lower than 32 (in
  April 2018, Yann Lecun even tweeted "Friends don’t let friends use mini-batches
  larger than 32“). A small batch size ensures that each training iteration is very
  fast, and although a large batch size will give a more precise estimate of the gradi‐
  ents, in practice this does not matter much since the optimization landscape is
  quite complex and the direction of the true gradients do not point precisely in
  the direction of the optimum. However, having a batch size greater than 10 helps
  take advantage of hardware and software optimizations, in particular for matrix
  multiplications, so it will speed up training. Moreover, if you use Batch Normal‐
  ization (see Chapter 11), the batch size should not be too small (in general no less
  than 20).
- We discussed the choice of the activation function earlier in this chapter: in gen‐
  eral, the ReLU activation function will be a good default for all hidden layers. For
  the output layer, it really depends on your task.
- In most cases, the number of training iterations does not actually need to be
  tweaked: just use early stopping instead.

```
For more best practices, make sure to read Yoshua Bengio’s great 2012 paper^18 , which
presents many practical recommendations for deep networks.
This concludes this introduction to artificial neural networks and their implementa‐
tion with Keras. In the next few chapters, we will discuss techniques to train very
deep nets, we will see how to customize your models using TensorFlow’s lower-level
API and how to load and preprocess data efficiently using the Data API, and we will
dive into other popular neural network architectures: convolutional neural networks
for image processing, recurrent neural networks for sequential data, autoencoders for
```

```
Fine-Tuning Neural Network Hyperparameters | 321
```

19 A few extra ANN architectures are presented in ???.

```
representation learning, and generative adversarial networks to model and generate
data.^19
```

### Exercises

1. Visit the TensorFlow Playground at https://playground.tensorflow.org/
   - Layers and patterns: try training the default neural network by clicking the run
     button (top left). Notice how it quickly finds a good solution for the classifica‐
     tion task. Notice that the neurons in the first hidden layer have learned simple
     patterns, while the neurons in the second hidden layer have learned to com‐
     bine the simple patterns of the first hidden layer into more complex patterns.
     In general, the more layers, the more complex the patterns can be.
   - Activation function: try replacing the Tanh activation function with the ReLU
     activation function, and train the network again. Notice that it finds a solution
     even faster, but this time the boundaries are linear. This is due to the shape of
     the ReLU function.
   - Local minima: modify the network architecture to have just one hidden layer
     with three neurons. Train it multiple times (to reset the network weights, click
     the reset button next to the play button). Notice that the training time varies a
     lot, and sometimes it even gets stuck in a local minimum.
   - Too small: now remove one neuron to keep just 2. Notice that the neural net‐
     work is now incapable of finding a good solution, even if you try multiple
     times. The model has too few parameters and it systematically underfits the
     training set.
   - Large enough: next, set the number of neurons to 8 and train the network sev‐
     eral times. Notice that it is now consistently fast and never gets stuck. This
     highlights an important finding in neural network theory: large neural net‐
     works almost never get stuck in local minima, and even when they do these
     local optima are almost as good as the global optimum. However, they can still
     get stuck on long plateaus for a long time.
   - Deep net and vanishing gradients: now change the dataset to be the spiral (bot‐
     tom right dataset under “DATA”). Change the network architecture to have 4
     hidden layers with 8 neurons each. Notice that training takes much longer, and
     often gets stuck on plateaus for long periods of time. Also notice that the neu‐
     rons in the highest layers (i.e. on the right) tend to evolve faster than the neu‐
     rons in the lowest layers (i.e. on the left). This problem, called the “vanishing
     gradients” problem, can be alleviated using better weight initialization and

```
322 | Chapter 10: Introduction to Artificial Neural Networks with Keras
```

```
other techniques, better optimizers (such as AdaGrad or Adam), or using
Batch Normalization.
```

- More: go ahead and play with the other parameters to get a feel of what they
  do. In fact, you should definitely play with this UI for at least one hour, it will
  grow your intuitions about neural networks significantly.

2. Draw an ANN using the original artificial neurons (like the ones in Figure 10-3)
   that computes A ⊕ B (where ⊕ represents the XOR operation). Hint: A ⊕ B = (A
   ∧ ¬ B) ∨ (¬ A ∧ B).
3. Why is it generally preferable to use a Logistic Regression classifier rather than a
   classical Perceptron (i.e., a single layer of threshold logic units trained using the
   Perceptron training algorithm)? How can you tweak a Perceptron to make it
   equivalent to a Logistic Regression classifier?
4. Why was the logistic activation function a key ingredient in training the first
   MLPs?
5. Name three popular activation functions. Can you draw them?
6. Suppose you have an MLP composed of one input layer with 10 passthrough
   neurons, followed by one hidden layer with 50 artificial neurons, and finally one
   output layer with 3 artificial neurons. All artificial neurons use the ReLU activa‐
   tion function.

- What is the shape of the input matrix **X**?
- What about the shape of the hidden layer’s weight vector **W** h, and the shape of
  its bias vector **b** h?
- What is the shape of the output layer’s weight vector **W** o, and its bias vector **b** o?
- What is the shape of the network’s output matrix **Y**?
- Write the equation that computes the network’s output matrix **Y** as a function
  of **X** , **W** h, **b** h, **W** o and **b** o.

7. How many neurons do you need in the output layer if you want to classify email
   into spam or ham? What activation function should you use in the output layer?
   If instead you want to tackle MNIST, how many neurons do you need in the out‐
   put layer, using what activation function? Answer the same questions for getting
   your network to predict housing prices as in Chapter 2.
8. What is backpropagation and how does it work? What is the difference between
   backpropagation and reverse-mode autodiff?
9. Can you list all the hyperparameters you can tweak in an MLP? If the MLP over‐
   fits the training data, how could you tweak these hyperparameters to try to solve
   the problem?

```
Exercises | 323
```

10. Train a deep MLP on the MNIST dataset and see if you can get over 98% preci‐
    sion. Try adding all the bells and whistles (i.e., save checkpoints, use early stop‐
    ping, plot learning curves using TensorBoard, and so on).

Solutions to these exercises are available in ???.

**324 | Chapter 10: Introduction to Artificial Neural Networks with Keras**

**CHAPTER 11**

**Training Deep Neural Networks**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 11 in the final
release of the book.
```

In Chapter 10 we introduced artificial neural networks and trained our first deep
neural networks. But they were very shallow nets, with just a few hidden layers. What
if you need to tackle a very complex problem, such as detecting hundreds of types of
objects in high-resolution images? You may need to train a much deeper DNN, per‐
haps with 10 layers or much more, each containing hundreds of neurons, connected
by hundreds of thousands of connections. This would not be a walk in the park:

- First, you would be faced with the tricky vanishing gradients problem (or the
  related exploding gradients problem) that affects deep neural networks and makes
  lower layers very hard to train.
- Second, you might not have enough training data for such a large network, or it
  might be too costly to label.
- Third, training may be extremely slow.
- Fourth, a model with millions of parameters would severely risk overfitting the
  training set, especially if there are not enough training instances, or they are too
  noisy.

In this chapter, we will go through each of these problems in turn and present techni‐
ques to solve them. We will start by explaining the vanishing gradients problem and
exploring some of the most popular solutions to this problem. Next, we will look at
transfer learning and unsupervised pretraining, which can help you tackle complex

```
325
```

```
1 “Understanding the Difficulty of Training Deep Feedforward Neural Networks,” X. Glorot, Y Bengio (2010).
```

tasks even when you have little labeled data. Then we will discuss various optimizers
that can speed up training large models tremendously compared to plain Gradient
Descent. Finally, we will go through a few popular regularization techniques for large
neural networks.

With these tools, you will be able to train very deep nets: welcome to Deep Learning!

### Vanishing/Exploding Gradients Problems

As we discussed in Chapter 10, the backpropagation algorithm works by going from
the output layer to the input layer, propagating the error gradient on the way. Once
the algorithm has computed the gradient of the cost function with regards to each
parameter in the network, it uses these gradients to update each parameter with a
Gradient Descent step.

Unfortunately, gradients often get smaller and smaller as the algorithm progresses
down to the lower layers. As a result, the Gradient Descent update leaves the lower
layer connection weights virtually unchanged, and training never converges to a good
solution. This is called the vanishing gradients problem. In some cases, the opposite
can happen: the gradients can grow bigger and bigger, so many layers get insanely
large weight updates and the algorithm diverges. This is the exploding gradients prob‐
lem, which is mostly encountered in recurrent neural networks (see ???). More gener‐
ally, deep neural networks suffer from unstable gradients; different layers may learn at
widely different speeds.

Although this unfortunate behavior has been empirically observed for quite a while
(it was one of the reasons why deep neural networks were mostly abandoned for a
long time), it is only around 2010 that significant progress was made in understand‐
ing it. A paper titled “Understanding the Difficulty of Training Deep Feedforward
Neural Networks” by Xavier Glorot and Yoshua Bengio^1 found a few suspects, includ‐
ing the combination of the popular logistic sigmoid activation function and the
weight initialization technique that was most popular at the time, namely random ini‐
tialization using a normal distribution with a mean of 0 and a standard deviation of 1.
In short, they showed that with this activation function and this initialization scheme,
the variance of the outputs of each layer is much greater than the variance of its
inputs. Going forward in the network, the variance keeps increasing after each layer
until the activation function saturates at the top layers. This is actually made worse by
the fact that the logistic function has a mean of 0.5, not 0 (the hyperbolic tangent
function has a mean of 0 and behaves slightly better than the logistic function in deep
networks).

**326 | Chapter 11: Training Deep Neural Networks**

```
2 Here’s an analogy: if you set a microphone amplifier’s knob too close to zero, people won’t hear your voice, but
if you set it too close to the max, your voice will be saturated and people won’t understand what you are say‐
ing. Now imagine a chain of such amplifiers: they all need to be set properly in order for your voice to come
out loud and clear at the end of the chain. Your voice has to come out of each amplifier at the same amplitude
as it came in.
```

Looking at the logistic activation function (see Figure 11-1), you can see that when
inputs become large (negative or positive), the function saturates at 0 or 1, with a
derivative extremely close to 0. Thus when backpropagation kicks in, it has virtually
no gradient to propagate back through the network, and what little gradient exists
keeps getting diluted as backpropagation progresses down through the top layers, so
there is really nothing left for the lower layers.

Figure 11-1. Logistic activation function saturation

#### Glorot and He Initialization

In their paper, Glorot and Bengio propose a way to significantly alleviate this prob‐
lem. We need the signal to flow properly in both directions: in the forward direction
when making predictions, and in the reverse direction when backpropagating gradi‐
ents. We don’t want the signal to die out, nor do we want it to explode and saturate.
For the signal to flow properly, the authors argue that we need the variance of the
outputs of each layer to be equal to the variance of its inputs,^2 and we also need the
gradients to have equal variance before and after flowing through a layer in the
reverse direction (please check out the paper if you are interested in the mathematical
details). It is actually not possible to guarantee both unless the layer has an equal
number of inputs and neurons (these numbers are called the fan-in and fan-out of the
layer), but they proposed a good compromise that has proven to work very well in
practice: the connection weights of each layer must be initialized randomly as

```
Vanishing/Exploding Gradients Problems | 327
```

```
3 Such as “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” K.
He et al. (2015).
```

described in Equation 11-1, where f anavg= f anin+f anout/2. This initialization
strategy is called Xavier initialization (after the author’s first name) or Glorot initiali‐
zation (after his last name).

```
Equation 11-1. Glorot initialization (when using the logistic activation function)
```

```
Normal distribution with mean 0 and variance σ^2 =fan^1
avg
```

```
Or a uniform distribution between −r and  +r, with r= fan^3
avg
```

If you just replace fanavg with fanin in Equation 11-1, you get an initialization strategy
that was actually already proposed by Yann LeCun in the 1990s, called LeCun initiali‐
zation, which was even recommended in the 1998 book Neural Networks: Tricks of the
Trade by Genevieve Orr and Klaus-Robert Müller (Springer). It is equivalent to
Glorot initialization when fanin = fanout. It took over a decade for researchers to realize
just how important this trick really is. Using Glorot initialization can speed up train‐
ing considerably, and it is one of the tricks that led to the current success of Deep
Learning.

Some papers^3 have provided similar strategies for different activation functions.
These strategies differ only by the scale of the variance and whether they use fanavg or

fanin, as shown in Table 11-1 (for the uniform distribution, just compute r= 3 σ^2 ).
The initialization strategy for the ReLU activation function (and its variants, includ‐
ing the ELU activation described shortly) is sometimes called He initialization (after
the last name of its author). The SELU activation function will be explained later in
this chapter. It should be used with LeCun initialization (preferably with a normal
distribution, as we will see).

Table 11-1. Initialization parameters for each type of activation function

```
Initialization Activation functions σ² (Normal)
Glorot None, Tanh, Logistic, Softmax 1 / fanavg
He ReLU & variants 2 / fanin
LeCun SELU 1 / fanin
```

By default, Keras uses Glorot initialization with a uniform distribution. You can
change this to He initialization by setting kernel_initializer="he_uniform" or ker
nel_initializer="he_normal" when creating a layer, like this:

**328 | Chapter 11: Training Deep Neural Networks**

```
4 Unless it is part of the first hidden layer, a dead neuron may sometimes come back to life: gradient descent
may indeed tweak neurons in the layers below in such a way that the weighted sum of the dead neuron’s
inputs is positive again.
5 “Empirical Evaluation of Rectified Activations in Convolution Network,” B. Xu et al. (2015).
```

```
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
```

If you want He initialization with a uniform distribution, but based on fanavg rather
than fanin, you can use the VarianceScaling initializer like this:

```
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
distribution='uniform')
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)
```

#### Nonsaturating Activation Functions

One of the insights in the 2010 paper by Glorot and Bengio was that the vanishing/
exploding gradients problems were in part due to a poor choice of activation func‐
tion. Until then most people had assumed that if Mother Nature had chosen to use
roughly sigmoid activation functions in biological neurons, they must be an excellent
choice. But it turns out that other activation functions behave much better in deep
neural networks, in particular the ReLU activation function, mostly because it does
not saturate for positive values (and also because it is quite fast to compute).

Unfortunately, the ReLU activation function is not perfect. It suffers from a problem
known as the dying ReLUs: during training, some neurons effectively die, meaning
they stop outputting anything other than 0. In some cases, you may find that half of
your network’s neurons are dead, especially if you used a large learning rate. A neu‐
ron dies when its weights get tweaked in such a way that the weighted sum of its
inputs are negative for all instances in the training set. When this happens, it just
keeps outputting 0s, and gradient descent does not affect it anymore since the gradi‐
ent of the ReLU function is 0 when its input is negative.^4

To solve this problem, you may want to use a variant of the ReLU function, such as
the leaky ReLU. This function is defined as LeakyReLUα(z) = max(αz, z) (see
Figure 11-2). The hyperparameter α defines how much the function “leaks”: it is the
slope of the function for z < 0, and is typically set to 0.01. This small slope ensures
that leaky ReLUs never die; they can go into a long coma, but they have a chance to
eventually wake up. A 2015 paper^5 compared several variants of the ReLU activation
function and one of its conclusions was that the leaky variants always outperformed
the strict ReLU activation function. In fact, setting α = 0.2 (huge leak) seemed to
result in better performance than α = 0.01 (small leak). They also evaluated the
randomized leaky ReLU (RReLU), where α is picked randomly in a given range during
training, and it is fixed to an average value during testing. It also performed fairly well
and seemed to act as a regularizer (reducing the risk of overfitting the training set).

```
Vanishing/Exploding Gradients Problems | 329
```

```
6 “Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs),” D. Clevert, T. Unterthiner,
S. Hochreiter (2015).
```

Finally, they also evaluated the parametric leaky ReLU (PReLU), where α is authorized
to be learned during training (instead of being a hyperparameter, it becomes a
parameter that can be modified by backpropagation like any other parameter). This
was reported to strongly outperform ReLU on large image datasets, but on smaller
datasets it runs the risk of overfitting the training set.

Figure 11-2. Leaky ReLU

Last but not least, a 2015 paper by Djork-Arné Clevert et al.^6 proposed a new activa‐
tion function called the exponential linear unit (ELU) that outperformed all the ReLU
variants in their experiments: training time was reduced and the neural network per‐
formed better on the test set. It is represented in Figure 11-3, and Equation 11-2
shows its definition.

```
Equation 11-2. ELU activation function
```

```
ELUα z =
```

```
α exp z − 1 ifz< 0
z ifz≥ 0
```

**330 | Chapter 11: Training Deep Neural Networks**

```
7 “Self-Normalizing Neural Networks, " G. Klambauer, T. Unterthiner and A. Mayr (2017).
```

Figure 11-3. ELU activation function

It looks a lot like the ReLU function, with a few major differences:

- First it takes on negative values when z < 0, which allows the unit to have an
  average output closer to 0. This helps alleviate the vanishing gradients problem,
  as discussed earlier. The hyperparameter α defines the value that the ELU func‐
  tion approaches when z is a large negative number. It is usually set to 1, but you
  can tweak it like any other hyperparameter if you want.
- Second, it has a nonzero gradient for z < 0, which avoids the dead neurons prob‐
  lem.
- Third, if α is equal to 1 then the function is smooth everywhere, including
  around z = 0, which helps speed up Gradient Descent, since it does not bounce as
  much left and right of z = 0.

The main drawback of the ELU activation function is that it is slower to compute
than the ReLU and its variants (due to the use of the exponential function), but dur‐
ing training this is compensated by the faster convergence rate. However, at test time
an ELU network will be slower than a ReLU network.

Moreover, in a 2017 paper^7 by Günter Klambauer et al., called “Self-Normalizing
Neural Networks”, the authors showed that if you build a neural network composed
exclusively of a stack of dense layers, and if all hidden layers use the SELU activation
function (which is just a scaled version of the ELU activation function, as its name
suggests), then the network will self-normalize: the output of each layer will tend to
preserve mean 0 and standard deviation 1 during training, which solves the vanish‐
ing/exploding gradients problem. As a result, this activation function often outper‐

```
Vanishing/Exploding Gradients Problems | 331
```

forms other activation functions very significantly for such neural nets (especially
deep ones). However, there are a few conditions for self-normalization to happen:

- The input features must be standardized (mean 0 and standard deviation 1).
- Every hidden layer’s weights must also be initialized using LeCun normal initiali‐
  zation. In Keras, this means setting kernel_initializer="lecun_normal".
- The network’s architecture must be sequential. Unfortunately, if you try to use
  SELU in non-sequential architectures, such as recurrent networks (see ???) or
  networks with skip connections (i.e., connections that skip layers, such as in wide
  & deep nets), self-normalization will not be guaranteed, so SELU will not neces‐
  sarily outperform other activation functions.
- The paper only guarantees self-normalization if all layers are dense. However, in
  practice the SELU activation function seems to work great with convolutional
  neural nets as well (see Chapter 14).

```
So which activation function should you use for the hidden layers
of your deep neural networks? Although your mileage will vary, in
general SELU > ELU > leaky ReLU (and its variants) > ReLU > tanh
> logistic. If the network’s architecture prevents it from self-
normalizing, then ELU may perform better than SELU (since SELU
is not smooth at z = 0). If you care a lot about runtime latency, then
you may prefer leaky ReLU. If you don’t want to tweak yet another
hyperparameter, you may just use the default α values used by
Keras (e.g., 0.3 for the leaky ReLU). If you have spare time and
computing power, you can use cross-validation to evaluate other
activation functions, in particular RReLU if your network is over‐
fitting, or PReLU if you have a huge training set.
```

To use the leaky ReLU activation function, you must create a LeakyReLU instance like
this:

```
leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
layer = keras.layers.Dense(10, activation=leaky_relu,
kernel_initializer="he_normal")
```

For PReLU, just replace LeakyRelu(alpha=0.2) with PReLU(). There is currently no
official implementation of RReLU in Keras, but you can fairly easily implement your
own (see the exercises at the end of Chapter 12).

For SELU activation, just set activation="selu" and kernel_initial
izer="lecun_normal" when creating a layer:

```
layer = keras.layers.Dense(10, activation="selu",
kernel_initializer="lecun_normal")
```

**332 | Chapter 11: Training Deep Neural Networks**

```
8 “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” S. Ioffe
and C. Szegedy (2015).
```

#### Batch Normalization

Although using He initialization along with ELU (or any variant of ReLU) can signifi‐
cantly reduce the vanishing/exploding gradients problems at the beginning of train‐
ing, it doesn’t guarantee that they won’t come back during training.

In a 2015 paper,^8 Sergey Ioffe and Christian Szegedy proposed a technique called
Batch Normalization (BN) to address the vanishing/exploding gradients problems.
The technique consists of adding an operation in the model just before or after the
activation function of each hidden layer, simply zero-centering and normalizing each
input, then scaling and shifting the result using two new parameter vectors per layer:
one for scaling, the other for shifting. In other words, this operation lets the model
learn the optimal scale and mean of each of the layer’s inputs. In many cases, if you
add a BN layer as the very first layer of your neural network, you do not need to
standardize your training set (e.g., using a StandardScaler): the BN layer will do it
for you (well, approximately, since it only looks at one batch at a time, and it can also
rescale and shift each input feature).

In order to zero-center and normalize the inputs, the algorithm needs to estimate
each input’s mean and standard deviation. It does so by evaluating the mean and stan‐
dard deviation of each input over the current mini-batch (hence the name “Batch
Normalization”). The whole operation is summarized in Equation 11-3.

```
Equation 11-3. Batch Normalization algorithm
```

```
1. μ B=m^1
B
```

```
∑
i= 1
```

```
mB
x i
```

```
2. σ B^2 =m^1
B
```

```
∑
i= 1
```

```
mB
x i− μ B^2
```

```
3. x i =
```

```
x i− μ B
σ B^2 +𝐀
```

```
4. z i = γ ⊗ x i+ β
```

- **μ** B is the vector of input means, evaluated over the whole mini-batch B (it con‐
  tains one mean per input).

```
Vanishing/Exploding Gradients Problems | 333
```

- **σ** B is the vector of input standard deviations, also evaluated over the whole mini-
  batch (it contains one standard deviation per input).
- mB is the number of instances in the mini-batch.
- **x** (i) is the vector of zero-centered and normalized inputs for instance i.
- **γ** is the output scale parameter vector for the layer (it contains one scale parame‐
  ter per input).
- ⊗ represents element-wise multiplication (each input is multiplied by its corre‐
  sponding output scale parameter).
- **β** is the output shift (offset) parameter vector for the layer (it contains one offset
  parameter per input). Each input is offset by its corresponding shift parameter.
- ε is a tiny number to avoid division by zero (typically 10–5). This is called a
  smoothing term.
- **z** (i) is the output of the BN operation: it is a rescaled and shifted version of the
  inputs.

So during training, BN just standardizes its inputs then rescales and offsets them.
Good! What about at test time? Well it is not that simple. Indeed, we may need to
make predictions for individual instances rather than for batches of instances: in this
case, we will have no way to compute each input’s mean and standard deviation.
Moreover, even if we do have a batch of instances, it may be too small, or the instan‐
ces may not be independent and identically distributed (IID), so computing statistics
over the batch instances would be unreliable (during training, the batches should not
be too small, if possible more than 30 instances, and all instances should be IID, as we
saw in Chapter 4). One solution could be to wait until the end of training, then run
the whole training set through the neural network, and compute the mean and stan‐
dard deviation of each input of the BN layer. These “final” input means and standard
deviations can then be used instead of the batch input means and standard deviations
when making predictions. However, it is often preferred to estimate these final statis‐
tics during training using a moving average of the layer’s input means and standard
deviations. To sum up, four parameter vectors are learned in each batch-normalized
layer: **γ** (the ouput scale vector) and **β** (the output offset vector) are learned through
regular backpropagation, and **μ** (the final input mean vector), and **σ** (the final input
standard deviation vector) are estimated using an exponential moving average. Note
that **μ** and **σ** are estimated during training, but they are not used at all during train‐
ing, only after training (to replace the batch input means and standard deviations in
Equation 11-3).

The authors demonstrated that this technique considerably improved all the deep
neural networks they experimented with, leading to a huge improvement in the
ImageNet classification task (ImageNet is a large database of images classified into
many classes and commonly used to evaluate computer vision systems). The vanish‐

**334 | Chapter 11: Training Deep Neural Networks**

ing gradients problem was strongly reduced, to the point that they could use saturat‐
ing activation functions such as the tanh and even the logistic activation function.
The networks were also much less sensitive to the weight initialization. They were
able to use much larger learning rates, significantly speeding up the learning process.
Specifically, they note that “Applied to a state-of-the-art image classification model,
Batch Normalization achieves the same accuracy with 14 times fewer training steps,
and beats the original model by a significant margin. [...] Using an ensemble of
batch-normalized networks, we improve upon the best published result on ImageNet
classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding
the accuracy of human raters.” Finally, like a gift that keeps on giving, Batch Normal‐
ization also acts like a regularizer, reducing the need for other regularization techni‐
ques (such as dropout, described later in this chapter).

Batch Normalization does, however, add some complexity to the model (although it
can remove the need for normalizing the input data, as we discussed earlier). More‐
over, there is a runtime penalty: the neural network makes slower predictions due to
the extra computations required at each layer. So if you need predictions to be
lightning-fast, you may want to check how well plain ELU + He initialization perform
before playing with Batch Normalization.

```
You may find that training is rather slow, because each epoch takes
much more time when you use batch normalization. However, this
is usually counterbalanced by the fact that convergence is much
faster with BN, so it will take fewer epochs to reach the same per‐
formance. All in all, wall time will usually be smaller (this is the
time measured by the clock on your wall).
```

**Implementing Batch Normalization with Keras**

As with most things with Keras, implementing Batch Normalization is quite simple.
Just add a BatchNormalization layer before or after each hidden layer’s activation
function, and optionally add a BN layer as well as the first layer in your model. For
example, this model applies BN after every hidden layer and as the first layer in the
model (after flattening the input images):

```
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.BatchNormalization(),
keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
keras.layers.BatchNormalization(),
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
keras.layers.BatchNormalization(),
keras.layers.Dense(10, activation="softmax")
])
```

```
Vanishing/Exploding Gradients Problems | 335
```

```
9 However, they are estimated during training, based on the training data, so arguably they are trainable. In
Keras, “Non-trainable” really means “untouched by backpropagation”.
```

That’s all! In this tiny example with just two hidden layers, it’s unlikely that Batch
Normalization will have a very positive impact, but for deeper networks it can make a
tremendous difference.

Let’s zoom in a bit. If you display the model summary, you can see that each BN layer
adds 4 parameters per input: **γ** , **β** , **μ** and **σ** (for example, the first BN layer adds 3136
parameters, which is 4 times 784). The last two parameters, **μ** and **σ** , are the moving
averages, they are not affected by backpropagation, so Keras calls them “Non-
trainable”^9 (if you count the total number of BN parameters, 3136 + 1200 + 400, and
divide by two, you get 2,368, which is the total number of non-trainable params in
this model).

```
>>> model.summary()
Model: "sequential_3"
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
flatten_3 (Flatten) (None, 784) 0
_________________________________________________________________
batch_normalization_v2 (Batc (None, 784) 3136
_________________________________________________________________
dense_50 (Dense) (None, 300) 235500
_________________________________________________________________
batch_normalization_v2_1 (Ba (None, 300) 1200
_________________________________________________________________
dense_51 (Dense) (None, 100) 30100
_________________________________________________________________
batch_normalization_v2_2 (Ba (None, 100) 400
_________________________________________________________________
dense_52 (Dense) (None, 10) 1010
=================================================================
Total params: 271,346
Trainable params: 268,978
Non-trainable params: 2,368
```

Let’s look at the parameters of the first BN layer. Two are trainable (by backprop), and
two are not:

```
>>> [(var.name, var.trainable) for var in model.layers[1].variables]
[('batch_normalization_v2/gamma:0', True),
('batch_normalization_v2/beta:0', True),
('batch_normalization_v2/moving_mean:0', False),
('batch_normalization_v2/moving_variance:0', False)]
```

Now when you create a BN layer in Keras, it also creates two operations that will be
called by Keras at each iteration during training. These operations will update the

**336 | Chapter 11: Training Deep Neural Networks**

moving averages. Since we are using the TensorFlow backend, these operations are
TensorFlow operations (we will discuss TF operations in Chapter 12).

```
>>> model.layers[1].updates
[<tf.Operation 'cond_2/Identity' type=Identity>,
<tf.Operation 'cond_3/Identity' type=Identity>]
```

The authors of the BN paper argued in favor of adding the BN layers before the acti‐
vation functions, rather than after (as we just did). There is some debate about this, as
it seems to depend on the task. So that’s one more thing you can experiment with to
see which option works best on your dataset. To add the BN layers before the activa‐
tion functions, we must remove the activation function from the hidden layers, and
add them as separate layers after the BN layers. Moreover, since a Batch Normaliza‐
tion layer includes one offset parameter per input, you can remove the bias term from
the previous layer (just pass use_bias=False when creating it):

```
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.BatchNormalization(),
keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
keras.layers.BatchNormalization(),
keras.layers.Activation("elu"),
keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
keras.layers.Activation("elu"),
keras.layers.BatchNormalization(),
keras.layers.Dense(10, activation="softmax")
])
```

The BatchNormalization class has quite a few hyperparameters you can tweak. The
defaults will usually be fine, but you may occasionally need to tweak the momentum.
This hyperparameter is used when updating the exponential moving averages: given a
new value **v** (i.e., a new vector of input means or standard deviations computed over
the current batch), the running average 𝐀 is updated using the following equation:

```
v v × momentum + v ×1 − momentum
```

A good momentum value is typically close to 1—for example, 0.9, 0.99, or 0.999 (you
want more 9s for larger datasets and smaller mini-batches).

Another important hyperparameter is axis: it determines which axis should be nor‐
malized. It defaults to –1, meaning that by default it will normalize the last axis (using
the means and standard deviations computed across the other axes). For example,
when the input batch is 2D (i.e., the batch shape is [batch size, features]), this means
that each input feature will be normalized based on the mean and standard deviation
computed across all the instances in the batch. For example, the first BN layer in the
previous code example will independently normalize (and rescale and shift) each of
the 784 input features. However, if we move the first BN layer before the Flatten

```
Vanishing/Exploding Gradients Problems | 337
```

10 “Fixup Initialization: Residual Learning Without Normalization,” Hongyi Zhang, Yann N. Dauphin, Tengyu
Ma (2019).
11 “On the difficulty of training recurrent neural networks,” R. Pascanu et al. (2013).

```
layer, then the input batches will be 3D, with shape [batch size, height, width], there‐
fore the BN layer will compute 28 means and 28 standard deviations (one per column
of pixels, computed across all instances in the batch, and all rows in the column), and
it will normalize all pixels in a given column using the same mean and standard devi‐
ation. There will also be just 28 scale parameters and 28 shift parameters. If instead
you still want to treat each of the 784 pixels independently, then you should set
axis=[1, 2].
Notice that the BN layer does not perform the same computation during training and
after training: it uses batch statistics during training, and the “final” statistics after
training (i.e., the final value of the moving averages). Let’s take a peek at the source
code of this class to see how this is handled:
class BatchNormalization (Layer):
[...]
def call(self, inputs, training=None):
if training is None:
training = keras.backend.learning_phase()
[...]
The call() method is the one that actually performs the computations, and as you
can see it has an extra training argument: if it is None it falls back to keras.back
end.learning_phase(), which returns 1 during training (the fit() method ensures
that). Otherwise, it returns 0. If you ever need to write a custom layer, and it needs to
behave differently during training and testing, simply use the same pattern (we will
discuss custom layers in Chapter 12).
Batch Normalization has become one of the most used layers in deep neural net‐
works, to the point that it is often omitted in the diagrams, as it is assumed that BN is
added after every layer. However, a very recent paper^10 by Hongyi Zhang et al. may
well change this: the authors show that by using a novel fixed-update (fixup) weight
initialization technique, they manage to train a very deep neural network (10,000 lay‐
ers!) without BN, achieving state-of-the-art performance on complex image classifi‐
cation tasks.
```

#### Gradient Clipping

```
Another popular technique to lessen the exploding gradients problem is to simply
clip the gradients during backpropagation so that they never exceed some threshold.
This is called Gradient Clipping.^11 This technique is most often used in recurrent neu‐
```

```
338 | Chapter 11: Training Deep Neural Networks
```

ral networks, as Batch Normalization is tricky to use in RNNs, as we will see in ???.
For other types of networks, BN is usually sufficient.

In Keras, implementing Gradient Clipping is just a matter of setting the clipvalue or
clipnorm argument when creating an optimizer. For example:

```
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)
```

This will clip every component of the gradient vector to a value between –1.0 and 1.0.
This means that all the partial derivatives of the loss (with regards to each and every
trainable parameter) will be clipped between –1.0 and 1.0. The threshold is a hyper‐
parameter you can tune. Note that it may change the orientation of the gradient vec‐
tor: for example, if the original gradient vector is [0.9, 100.0], it points mostly in the
direction of the second axis, but once you clip it by value, you get [0.9, 1.0], which
points roughly in the diagonal between the two axes. In practice however, this
approach works well. If you want to ensure that Gradient Clipping does not change
the direction of the gradient vector, you should clip by norm by setting clipnorm
instead of clipvalue. This will clip the whole gradient if its ℓ 2 norm is greater than
the threshold you picked. For example, if you set clipnorm=1.0, then the vector [0.9,
100.0] will be clipped to [0.00899964, 0.9999595], preserving its orientation, but
almost eliminating the first component. If you observe that the gradients explode
during training (you can track the size of the gradients using TensorBoard), you may
want to try both clipping by value and clipping by norm, with different threshold,
and see which option performs best on the validation set.

### Reusing Pretrained Layers

It is generally not a good idea to train a very large DNN from scratch: instead, you
should always try to find an existing neural network that accomplishes a similar task
to the one you are trying to tackle (we will discuss how to find them in Chapter 14),
then just reuse the lower layers of this network: this is called transfer learning. It will
not only speed up training considerably, but will also require much less training data.

For example, suppose that you have access to a DNN that was trained to classify pic‐
tures into 100 different categories, including animals, plants, vehicles, and everyday
objects. You now want to train a DNN to classify specific types of vehicles. These
tasks are very similar, even partly overlapping, so you should try to reuse parts of the
first network (see Figure 11-4).

```
Reusing Pretrained Layers | 339
```

Figure 11-4. Reusing pretrained layers

```
If the input pictures of your new task don’t have the same size as
the ones used in the original task, you will usually have to add a
preprocessing step to resize them to the size expected by the origi‐
nal model. More generally, transfer learning will work best when
the inputs have similar low-level features.
```

The output layer of the original model should usually be replaced since it is most
likely not useful at all for the new task, and it may not even have the right number of
outputs for the new task.

Similarly, the upper hidden layers of the original model are less likely to be as useful
as the lower layers, since the high-level features that are most useful for the new task
may differ significantly from the ones that were most useful for the original task. You
want to find the right number of layers to reuse.

```
The more similar the tasks are, the more layers you want to reuse
(starting with the lower layers). For very similar tasks, you can try
keeping all the hidden layers and just replace the output layer.
```

Try freezing all the reused layers first (i.e., make their weights non-trainable, so gradi‐
ent descent won’t modify them), then train your model and see how it performs.
Then try unfreezing one or two of the top hidden layers to let backpropagation tweak
them and see if performance improves. The more training data you have, the more

**340 | Chapter 11: Training Deep Neural Networks**

layers you can unfreeze. It is also useful to reduce the learning rate when you unfreeze
reused layers: this will avoid wrecking their fine-tuned weights.

If you still cannot get good performance, and you have little training data, try drop‐
ping the top hidden layer(s) and freeze all remaining hidden layers again. You can
iterate until you find the right number of layers to reuse. If you have plenty of train‐
ing data, you may try replacing the top hidden layers instead of dropping them, and
even add more hidden layers.

#### Transfer Learning With Keras

Let’s look at an example. Suppose the fashion MNIST dataset only contained 8 classes,
for example all classes except for sandals and shirts. Someone built and trained a
Keras model on that set and got reasonably good performance (>90% accuracy). Let’s
call this model A. You now want to tackle a different task: you have images of sandals
and shirts, and you want to train a binary classifier (positive=shirts, negative=san‐
dals). However, your dataset is quite small, you only have 200 labeled images. When
you train a new model for this task (let’s call it model B), with the same architecture
as model A, it performs reasonably well (97.2% accuracy), but since it’s a much easier
task (there are just 2 classes), you were hoping for more. While drinking your morn‐
ing coffee, you realize that your task is quite similar to task A, so perhaps transfer
learning can help? Let’s find out!

First, you need to load model A, and create a new model based on the model A’s lay‐
ers. Let’s reuse all layers except for the output layer:

```
model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
```

Note that model_A and model_B_on_A now share some layers. When you train
model_B_on_A, it will also affect model_A. If you want to avoid that, you need to clone
model_A before you reuse its layers. To do this, you must clone model A’s architecture,
then copy its weights (since clone_model() does not clone the weights):

```
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
```

Now we could just train model_B_on_A for task B, but since the new output layer was
initialized randomly, it will make large errors, at least during the first few epochs, so
there will be large error gradients that may wreck the reused weights. To avoid this,
one approach is to freeze the reused layers during the first few epochs, giving the new
layer some time to learn reasonable weights. To do this, simply set every layer’s train
able attribute to False and compile the model:

```
for layer in model_B_on_A.layers[:-1]:
layer.trainable = False
```

```
Reusing Pretrained Layers | 341
```

```
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",
metrics=["accuracy"])
```

```
You must always compile your model after you freeze or unfreeze
layers.
```

Next, we can train the model for a few epochs, then unfreeze the reused layers (which
requires compiling the model again) and continue training to fine-tune the reused
layers for task B. After unfreezing the reused layers, it is usually a good idea to reduce
the learning rate, once again to avoid damaging the reused weights:

```
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
validation_data=(X_valid_B, y_valid_B))
```

```
for layer in model_B_on_A.layers[:-1]:
layer.trainable = True
```

```
optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-3
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
validation_data=(X_valid_B, y_valid_B))
```

So, what’s the final verdict? Well this model’s test accuracy is 99.25%, which means
that transfer learning reduced the error rate from 2.8% down to almost 0.7%! That’s a
factor of 4!

```
>>> model_B_on_A.evaluate(X_test_B, y_test_B)
[0.06887910133600235, 0.9925]
```

Are you convinced? Well you shouldn’t be: I cheated! :) I tried many configurations
until I found one that demonstrated a strong improvement. If you try to change the
classes or the random seed, you will see that the improvement generally drops, or
even vanishes or reverses. What I did is called “torturing the data until it confesses”.
When a paper just looks too positive, you should be suspicious: perhaps the flashy
new technique does not help much (in fact, it may even degrade performance), but
the authors tried many variants and reported only the best results (which may be due
to shear luck), without mentioning how many failures they encountered on the way.
Most of the time, this is not malicious at all, but it is part of the reason why so many
results in Science can never be reproduced.

So why did I cheat? Well it turns out that transfer learning does not work very well
with small dense networks: it works best with deep convolutional neural networks, so
we will revisit transfer learning in Chapter 14, using the same techniques (and this
time there will be no cheating, I promise!).

**342 | Chapter 11: Training Deep Neural Networks**

#### Unsupervised Pretraining

Suppose you want to tackle a complex task for which you don’t have much labeled
training data, but unfortunately you cannot find a model trained on a similar task.
Don’t lose all hope! First, you should of course try to gather more labeled training
data, but if this is too hard or too expensive, you may still be able to perform unsuper‐
vised pretraining (see Figure 11-5). It is often rather cheap to gather unlabeled train‐
ing examples, but quite expensive to label them. If you can gather plenty of unlabeled
training data, you can try to train the layers one by one, starting with the lowest layer
and then going up, using an unsupervised feature detector algorithm such as Restric‐
ted Boltzmann Machines (RBMs; see ???) or autoencoders (see ???). Each layer is
trained on the output of the previously trained layers (all layers except the one being
trained are frozen). Once all layers have been trained this way, you can add the output
layer for your task, and fine-tune the final network using supervised learning (i.e.,
with the labeled training examples). At this point, you can unfreeze all the pretrained
layers, or just some of the upper ones.

Figure 11-5. Unsupervised pretraining

This is a rather long and tedious process, but it often works well; in fact, it is this
technique that Geoffrey Hinton and his team used in 2006 and which led to the
revival of neural networks and the success of Deep Learning. Until 2010, unsuper‐
vised pretraining (typically using RBMs) was the norm for deep nets, and it was only
after the vanishing gradients problem was alleviated that it became much more com‐

```
Reusing Pretrained Layers | 343
```

mon to train DNNs purely using supervised learning. However, unsupervised pre‐
training (today typically using autoencoders rather than RBMs) is still a good option
when you have a complex task to solve, no similar model you can reuse, and little
labeled training data but plenty of unlabeled training data.

#### Pretraining on an Auxiliary Task

If you do not have much labeled training data, one last option is to train a first neural
network on an auxiliary task for which you can easily obtain or generate labeled
training data, then reuse the lower layers of that network for your actual task. The
first neural network’s lower layers will learn feature detectors that will likely be reusa‐
ble by the second neural network.

For example, if you want to build a system to recognize faces, you may only have a
few pictures of each individual—clearly not enough to train a good classifier. Gather‐
ing hundreds of pictures of each person would not be practical. However, you could
gather a lot of pictures of random people on the web and train a first neural network
to detect whether or not two different pictures feature the same person. Such a net‐
work would learn good feature detectors for faces, so reusing its lower layers would
allow you to train a good face classifier using little training data.

For natural language processing (NLP) applications, you can easily download millions
of text documents and automatically generate labeled data from it. For example, you
could randomly mask out some words and train a model to predict what the missing
words are (e.g., it should predict that the missing word in the sentence “What \_\_\_
you saying?” is probably “are” or “were”). If you can train a model to reach good per‐
formance on this task, then it will already know quite a lot about language, and you
can certainly reuse it for your actual task, and fine-tune it on your labeled data (we
will discuss more pretraining tasks in ???).

```
Self-supervised learning is when you automatically generate the
labels from the data itself, then you train a model on the resulting
“labeled” dataset using supervised learning techniques. Since this
approach requires no human labeling whatsoever, it is best classi‐
fied as a form of unsupervised learning.
```

### Faster Optimizers

Training a very large deep neural network can be painfully slow. So far we have seen
four ways to speed up training (and reach a better solution): applying a good initiali‐
zation strategy for the connection weights, using a good activation function, using
Batch Normalization, and reusing parts of a pretrained network (possibly built on an
auxiliary task or using unsupervised learning). Another huge speed boost comes from
using a faster optimizer than the regular Gradient Descent optimizer. In this section

**344 | Chapter 11: Training Deep Neural Networks**

12 “Some methods of speeding up the convergence of iteration methods,” B. Polyak (1964).

```
we will present the most popular ones: Momentum optimization, Nesterov Acceler‐
ated Gradient, AdaGrad, RMSProp, and finally Adam and Nadam optimization.
```

#### Momentum Optimization

```
Imagine a bowling ball rolling down a gentle slope on a smooth surface: it will start
out slowly, but it will quickly pick up momentum until it eventually reaches terminal
velocity (if there is some friction or air resistance). This is the very simple idea behind
Momentum optimization, proposed by Boris Polyak in 1964.^12 In contrast, regular
Gradient Descent will simply take small regular steps down the slope, so it will take
much more time to reach the bottom.
Recall that Gradient Descent simply updates the weights θ by directly subtracting the
gradient of the cost function J( θ ) with regards to the weights (∇ θ J( θ )) multiplied by
the learning rate η. The equation is: θ ← θ – η∇ θ J( θ ). It does not care about what the
earlier gradients were. If the local gradient is tiny, it goes very slowly.
Momentum optimization cares a great deal about what previous gradients were: at
each iteration, it subtracts the local gradient from the momentum vector m (multi‐
plied by the learning rate η), and it updates the weights by simply adding this
momentum vector (see Equation 11-4). In other words, the gradient is used for accel‐
eration, not for speed. To simulate some sort of friction mechanism and prevent the
momentum from growing too large, the algorithm introduces a new hyperparameter
β, simply called the momentum, which must be set between 0 (high friction) and 1
(no friction). A typical momentum value is 0.9.
```

```
Equation 11-4. Momentum algorithm
1. m β m −η∇ θ J θ
2. θ θ + m
```

```
You can easily verify that if the gradient remains constant, the terminal velocity (i.e.,
the maximum size of the weight updates) is equal to that gradient multiplied by the
learning rate η multiplied by 1 −^1 β (ignoring the sign). For example, if β = 0.9, then the
terminal velocity is equal to 10 times the gradient times the learning rate, so Momen‐
tum optimization ends up going 10 times faster than Gradient Descent! This allows
Momentum optimization to escape from plateaus much faster than Gradient Descent.
In particular, we saw in Chapter 4 that when the inputs have very different scales the
cost function will look like an elongated bowl (see Figure 4-7). Gradient Descent goes
down the steep slope quite fast, but then it takes a very long time to go down the val‐
```

```
Faster Optimizers | 345
```

13 “A Method for Unconstrained Convex Minimization Problem with the Rate of Convergence O(1/k^2 ),” Yurii
Nesterov (1983).

```
ley. In contrast, Momentum optimization will roll down the valley faster and faster
until it reaches the bottom (the optimum). In deep neural networks that don’t use
Batch Normalization, the upper layers will often end up having inputs with very dif‐
ferent scales, so using Momentum optimization helps a lot. It can also help roll past
local optima.
```

```
Due to the momentum, the optimizer may overshoot a bit, then
come back, overshoot again, and oscillate like this many times
before stabilizing at the minimum. This is one of the reasons why it
is good to have a bit of friction in the system: it gets rid of these
oscillations and thus speeds up convergence.
```

```
Implementing Momentum optimization in Keras is a no-brainer: just use the SGD
optimizer and set its momentum hyperparameter, then lie back and profit!
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
The one drawback of Momentum optimization is that it adds yet another hyperpara‐
meter to tune. However, the momentum value of 0.9 usually works well in practice
and almost always goes faster than regular Gradient Descent.
```

#### Nesterov Accelerated Gradient

```
One small variant to Momentum optimization, proposed by Yurii Nesterov in 1983,^13
is almost always faster than vanilla Momentum optimization. The idea of Nesterov
Momentum optimization, or Nesterov Accelerated Gradient (NAG), is to measure the
gradient of the cost function not at the local position but slightly ahead in the direc‐
tion of the momentum (see Equation 11-5). The only difference from vanilla
Momentum optimization is that the gradient is measured at θ + β m rather than at θ.
```

```
Equation 11-5. Nesterov Accelerated Gradient algorithm
1. m β m −η∇ θ J θ +β m
2. θ θ + m
```

```
This small tweak works because in general the momentum vector will be pointing in
the right direction (i.e., toward the optimum), so it will be slightly more accurate to
use the gradient measured a bit farther in that direction rather than using the gradi‐
ent at the original position, as you can see in Figure 11-6 (where ∇ 1 represents the
gradient of the cost function measured at the starting point θ , and ∇ 2 represents the
```

```
346 | Chapter 11: Training Deep Neural Networks
```

14 “Adaptive Subgradient Methods for Online Learning and Stochastic Optimization,” J. Duchi et al. (2011).

```
gradient at the point located at θ + β m ). As you can see, the Nesterov update ends up
slightly closer to the optimum. After a while, these small improvements add up and
NAG ends up being significantly faster than regular Momentum optimization. More‐
over, note that when the momentum pushes the weights across a valley, ∇ 1 continues
to push further across the valley, while ∇ 2 pushes back toward the bottom of the val‐
ley. This helps reduce oscillations and thus converges faster.
NAG will almost always speed up training compared to regular Momentum optimi‐
zation. To use it, simply set nesterov=True when creating the SGD optimizer:
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
```

```
Figure 11-6. Regular versus Nesterov Momentum optimization
```

#### AdaGrad

```
Consider the elongated bowl problem again: Gradient Descent starts by quickly going
down the steepest slope, then slowly goes down the bottom of the valley. It would be
nice if the algorithm could detect this early on and correct its direction to point a bit
more toward the global optimum.
The AdaGrad algorithm^14 achieves this by scaling down the gradient vector along the
steepest dimensions (see Equation 11-6):
```

```
Equation 11-6. AdaGrad algorithm
1. s s +∇ θ J θ ⊗∇ θ J θ
2. θ θ −η∇ θ J θ ⊘ s +𝐀
```

```
Faster Optimizers | 347
```

The first step accumulates the square of the gradients into the vector **s** (recall that the
⊗ symbol represents the element-wise multiplication). This vectorized form is equiv‐
alent to computing si ← si + (∂ J( **θ** ) / ∂ θi)^2 for each element si of the vector **s** ; in other
words, each si accumulates the squares of the partial derivative of the cost function
with regards to parameter θi. If the cost function is steep along the ith dimension, then
si will get larger and larger at each iteration.

The second step is almost identical to Gradient Descent, but with one big difference:
the gradient vector is scaled down by a factor of 𝐀+𝐀 (the ⊘ symbol represents the
element-wise division, and ε is a smoothing term to avoid division by zero, typically
set to 10–10). This vectorized form is equivalent to computing
θi θi−η∂J **θ** / ∂θi/ si+𝐀 for all parameters θi (simultaneously).

In short, this algorithm decays the learning rate, but it does so faster for steep dimen‐
sions than for dimensions with gentler slopes. This is called an adaptive learning rate.
It helps point the resulting updates more directly toward the global optimum (see
Figure 11-7). One additional benefit is that it requires much less tuning of the learn‐
ing rate hyperparameter η.

Figure 11-7. AdaGrad versus Gradient Descent

AdaGrad often performs well for simple quadratic problems, but unfortunately it
often stops too early when training neural networks. The learning rate gets scaled
down so much that the algorithm ends up stopping entirely before reaching the
global optimum. So even though Keras has an Adagrad optimizer, you should not use
it to train deep neural networks (it may be efficient for simpler tasks such as Linear
Regression, though). However, understanding Adagrad is helpful to grasp the other
adaptive learning rate optimizers.

**348 | Chapter 11: Training Deep Neural Networks**

15 This algorithm was created by Geoffrey Hinton and Tijmen Tieleman in 2012, and presented by Geoffrey
Hinton in his Coursera class on neural networks (slides: https://homl.info/57; video: https://homl.info/58).
Amusingly, since the authors did not write a paper to describe it, researchers often cite “slide 29 in lecture 6”
in their papers.
16 “Adam: A Method for Stochastic Optimization,” D. Kingma, J. Ba (2015).
17 These are estimations of the mean and (uncentered) variance of the gradients. The mean is often called the
first moment, while the variance is often called the second moment, hence the name of the algorithm.

#### RMSProp

```
Although AdaGrad slows down a bit too fast and ends up never converging to the
global optimum, the RMSProp algorithm^15 fixes this by accumulating only the gradi‐
ents from the most recent iterations (as opposed to all the gradients since the begin‐
ning of training). It does so by using exponential decay in the first step (see Equation
11-7).
```

```
Equation 11-7. RMSProp algorithm
1. s β s + 1 −β∇ θ J θ ⊗∇ θ J θ
2. θ θ −η∇ θ J θ ⊘ s +𝐀
```

```
The decay rate β is typically set to 0.9. Yes, it is once again a new hyperparameter, but
this default value often works well, so you may not need to tune it at all.
As you might expect, Keras has an RMSProp optimizer:
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
Except on very simple problems, this optimizer almost always performs much better
than AdaGrad. In fact, it was the preferred optimization algorithm of many research‐
ers until Adam optimization came around.
```

#### Adam and Nadam Optimization

```
Adam,^16 which stands for adaptive moment estimation, combines the ideas of Momen‐
tum optimization and RMSProp: just like Momentum optimization it keeps track of
an exponentially decaying average of past gradients, and just like RMSProp it keeps
track of an exponentially decaying average of past squared gradients (see Equation
11-8).^17
```

```
Faster Optimizers | 349
```

```
Equation 11-8. Adam algorithm
1. m β 1 m − 1 −β 1 ∇ θ J θ
2. s β 2 s + 1 −β 2 ∇ θ J θ ⊗∇ θ J θ
```

```
3. m m
1 −β 1 t
```

```
4. s s
1 −β 2 t
5. θ θ +η m ⊘ s +𝐀
```

- t represents the iteration number (starting at 1).

If you just look at steps 1, 2, and 5, you will notice Adam’s close similarity to both
Momentum optimization and RMSProp. The only difference is that step 1 computes
an exponentially decaying average rather than an exponentially decaying sum, but
these are actually equivalent except for a constant factor (the decaying average is just
1 – β 1 times the decaying sum). Steps 3 and 4 are somewhat of a technical detail: since
**m** and **s** are initialized at 0, they will be biased toward 0 at the beginning of training,
so these two steps will help boost **m** and **s** at the beginning of training.

The momentum decay hyperparameter β 1 is typically initialized to 0.9, while the scal‐
ing decay hyperparameter β 2 is often initialized to 0.999. As earlier, the smoothing
term ε is usually initialized to a tiny number such as 10–7. These are the default values
for the Adam class (to be precise, epsilon defaults to None, which tells Keras to use
keras.backend.epsilon(), which defaults to 10–7; you can change it using
keras.backend.set_epsilon()).

```
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
```

Since Adam is an adaptive learning rate algorithm (like AdaGrad and RMSProp), it
requires less tuning of the learning rate hyperparameter η. You can often use the
default value η = 0.001, making Adam even easier to use than Gradient Descent.

```
If you are starting to feel overwhelmed by all these different techni‐
ques, and wondering how to choose the right ones for your task,
don’t worry: some practical guidelines are provided at the end of
this chapter.
```

Finally, two variants of Adam are worth mentioning:

**350 | Chapter 11: Training Deep Neural Networks**

18 “Incorporating Nesterov Momentum into Adam,” Timothy Dozat (2015).
19 “The Marginal Value of Adaptive Gradient Methods in Machine Learning,” A. C. Wilson et al. (2017).

- Adamax, introduced in the same paper as Adam: notice that in step 2 of Equation
  11-8, Adam accumulates the squares of the gradients in **s** (with a greater weight
  for more recent weights). In step 5, if we ignore ε and steps 3 and 4 (which are
  technical details anyway), Adam just scales down the parameter updates by the
  square root of **s**. In short, Adam scales down the parameter updates by the ℓ 2
  norm of the time-decayed gradients (recall that the ℓ 2 norm is the square root of
  the sum of squares). Adamax just replaces the ℓ 2 norm with the ℓ∞ norm (a fancy
  way of saying the max). Specifically, it replaces step 2 in Equation 11-8 with
  𝐀 max β 2 𝐀,∇θJθ , it drops step 4, and in step 5 it scales down the gradient
  updates by a factor of **s** , which is just the max of the time-decayed gradients. In
  practice, this can make Adamax more stable than Adam, but this really depends
  on the dataset, and in general Adam actually performs better. So it’s just one
  more optimizer you can try if you experience problems with Adam on some task.
- Nadam optimization^18 is more important: it is simply Adam optimization plus
  the Nesterov trick, so it will often converge slightly faster than Adam. In his
  report, Timothy Dozat compares many different optimizers on various tasks, and
  finds that Nadam generally outperforms Adam, but is sometimes outperformed
  by RMSProp.

```
Adaptive optimization methods (including RMSProp, Adam and
Nadam optimization) are often great, converging fast to a good sol‐
ution. However, a 2017 paper^19 by Ashia C. Wilson et al. showed
that they can lead to solutions that generalize poorly on some data‐
sets. So when you are disappointed by your model’s performance,
try using plain Nesterov Accelerated Gradient instead: your dataset
may just be allergic to adaptive gradients. Also check out the latest
research, it is moving fast (e.g., AdaBound).
```

```
All the optimization techniques discussed so far only rely on the first-order partial
derivatives (Jacobians). The optimization literature contains amazing algorithms
based on the second-order partial derivatives (the Hessians, which are the partial
derivatives of the Jacobians). Unfortunately, these algorithms are very hard to apply
to deep neural networks because there are n^2 Hessians per output (where n is the
number of parameters), as opposed to just n Jacobians per output. Since DNNs typi‐
cally have tens of thousands of parameters, the second-order optimization algorithms
```

```
Faster Optimizers | 351
```

20 “Primal-Dual Subgradient Methods for Convex Problems,” Yurii Nesterov (2005).
21 “Ad Click Prediction: a View from the Trenches,” H. McMahan et al. (2013).

```
often don’t even fit in memory, and even when they do, computing the Hessians is
just too slow.
```

```
Training Sparse Models
All the optimization algorithms just presented produce dense models, meaning that
most parameters will be nonzero. If you need a blazingly fast model at runtime, or if
you need it to take up less memory, you may prefer to end up with a sparse model
instead.
One trivial way to achieve this is to train the model as usual, then get rid of the tiny
weights (set them to 0). However, this will typically not lead to a very sparse model,
and it may degrade the model’s performance.
A better option is to apply strong ℓ 1 regularization during training, as it pushes the
optimizer to zero out as many weights as it can (as discussed in Chapter 4 about Lasso
Regression).
However, in some cases these techniques may remain insufficient. One last option is
to apply Dual Averaging, often called Follow The Regularized Leader (FTRL), a techni‐
que proposed by Yurii Nesterov.^20 When used with ℓ 1 regularization, this technique
often leads to very sparse models. Keras implements a variant of FTRL called FTRL-
Proximal^21 in the FTRL optimizer.
```

#### Learning Rate Scheduling

```
Finding a good learning rate can be tricky. If you set it way too high, training may
actually diverge (as we discussed in Chapter 4). If you set it too low, training will
eventually converge to the optimum, but it will take a very long time. If you set it
slightly too high, it will make progress very quickly at first, but it will end up dancing
around the optimum, never really settling down. If you have a limited computing
budget, you may have to interrupt training before it has converged properly, yielding
a suboptimal solution (see Figure 11-8).
```

```
352 | Chapter 11: Training Deep Neural Networks
```

Figure 11-8. Learning curves for various learning rates η

As we discussed in Chapter 10, one approach is to start with a large learning rate, and
divide it by 3 until the training algorithm stops diverging. You will not be too far
from the optimal learning rate, which will learn quickly and converge to good solu‐
tion.

However, you can do better than a constant learning rate: if you start with a high
learning rate and then reduce it once it stops making fast progress, you can reach a
good solution faster than with the optimal constant learning rate. There are many dif‐
ferent strategies to reduce the learning rate during training. These strategies are called
learning schedules (we briefly introduced this concept in Chapter 4), the most com‐
mon of which are:

Power scheduling
Set the learning rate to a function of the iteration number t: η(t) = η 0 / (1 + t/k)c.
The initial learning rate η 0 , the power c (typically set to 1) and the steps s are
hyperparameters. The learning rate drops at each step, and after s steps it is down
to η 0 / 2. After s more steps, it is down to η 0 / 3. Then down to η 0 / 4, then η 0 / 5,
and so on. As you can see, this schedule first drops quickly, then more and more
slowly. Of course, this requires tuning η 0 , s (and possibly c).

Exponential scheduling
Set the learning rate to: η(t) = η 0 0.1t/s. The learning rate will gradually drop by a
factor of 10 every s steps. While power scheduling reduces the learning rate more
and more slowly, exponential scheduling keeps slashing it by a factor of 10 every
s steps.

Piecewise constant scheduling
Use a constant learning rate for a number of epochs (e.g., η 0 = 0.1 for 5 epochs),
then a smaller learning rate for another number of epochs (e.g., η 1 = 0.001 for 50
epochs), and so on. Although this solution can work very well, it requires fid‐

```
Faster Optimizers | 353
```

22 “An Empirical Study of Learning Rates in Deep Neural Networks for Speech Recognition,” A. Senior et al.
(2013).

```
dling around to figure out the right sequence of learning rates, and how long to
use each of them.
Performance scheduling
Measure the validation error every N steps (just like for early stopping) and
reduce the learning rate by a factor of λ when the error stops dropping.
A 2013 paper^22 by Andrew Senior et al. compared the performance of some of the
most popular learning schedules when training deep neural networks for speech rec‐
ognition using Momentum optimization. The authors concluded that, in this setting,
both performance scheduling and exponential scheduling performed well. They
favored exponential scheduling because it was easy to tune and it converged slightly
faster to the optimal solution (they also mentioned that it was easier to implement
than performance scheduling, but in Keras both options are easy).
Implementing power scheduling in Keras is the easiest option: just set the decay
hyperparameter when creating an optimizer. The decay is the inverse of s (the num‐
ber of steps it takes to divide the learning rate by one more unit), and Keras assumes
that c is equal to 1. For example:
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
Exponential scheduling and piecewise scheduling are quite simple too. You first need
to define a function that takes the current epoch and returns the learning rate. For
example, let’s implement exponential scheduling:
def exponential_decay_fn(epoch):
return 0.01 * 0.1**(epoch / 20)
If you do not want to hard-code η 0 and s, you can create a function that returns a
configured function:
def exponential_decay(lr0, s):
def exponential_decay_fn(epoch):
return lr0 * 0.1**(epoch / s)
return exponential_decay_fn
```

```
exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
Next, just create a LearningRateScheduler callback, giving it the schedule function,
and pass this callback to the fit() method:
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, [...], callbacks=[lr_scheduler])
```

```
354 | Chapter 11: Training Deep Neural Networks
```

The LearningRateScheduler will update the optimizer’s learning_rate attribute at
the beginning of each epoch. Updating the learning rate just once per epoch is usually
enough, but if you want it to be updated more often, for example at every step, you
need to write your own callback (see the notebook for an example). This can make
sense if there are many steps per epoch.

The schedule function can optionally take the current learning rate as a second argu‐
ment. For example, the following schedule function just multiplies the previous
learning rate by 0.1&1/20, which results in the same exponential decay (except the decay
now starts at the beginning of epoch 0 instead of 1). This implementation relies on
the optimizer’s initial learning rate (contrary to the previous implementation), so
make sure to set it appropriately.

```
def exponential_decay_fn(epoch, lr):
return lr * 0.1**(1 / 20)
```

When you save a model, the optimizer and its learning rate get saved along with it.
This means that with this new schedule function, you could just load a trained model
and continue training where it left off, no problem. However, things are not so simple
if your schedule function uses the epoch argument: indeed, the epoch does not get
saved, and it gets reset to 0 every time you call the fit() method. This could lead to a
very large learning rate when you continue training a model where it left off, which
would likely damage your model’s weights. One solution is to manually set the fit()
method’s initial_epoch argument so the epoch starts at the right value.

For piecewise constant scheduling, you can use a schedule function like the following
one (as earlier, you can define a more general function if you want, see the notebook
for an example), then create a LearningRateScheduler callback with this function
and pass it to the fit() method, just like we did for exponential scheduling:

```
def piecewise_constant_fn(epoch):
if epoch < 5:
return 0.01
elif epoch < 15:
return 0.005
else :
return 0.001
```

For performance scheduling, simply use the ReduceLROnPlateau callback. For exam‐
ple, if you pass the following callback to the fit() method, it will multiply the learn‐
ing rate by 0.5 whenever the best validation loss does not improve for 5 consecutive
epochs (other options are available, please check the documentation for more
details):

```
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
```

Lastly, tf.keras offers an alternative way to implement learning rate scheduling: just
define the learning rate using one of the schedules available in keras.optimiz

```
Faster Optimizers | 355
```

ers.schedules, then pass this learning rate to any optimizer. This approach updates
the learning rate at each step rather than at each epoch. For example, here is how to
implement the same exponential schedule as earlier:

```
s = 20 * len(X_train) // 32 # number of steps in 20 epochs (batch size = 32)
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)
```

This is nice and simple, plus when you save the model, the learning rate and its
schedule (including its state) get saved as well. However, this approach is not part of
the Keras API, it is specific to tf.keras.

To sum up, exponential decay or performance scheduling can considerably speed up
convergence, so give them a try!

### Avoiding Overfitting Through Regularization

```
With four parameters I can fit an elephant and with five I can make him wiggle his
trunk.
—John von Neumann, cited by Enrico Fermi in Nature 427
```

With thousands of parameters you can fit the whole zoo. Deep neural networks typi‐
cally have tens of thousands of parameters, sometimes even millions. With so many
parameters, the network has an incredible amount of freedom and can fit a huge vari‐
ety of complex datasets. But this great flexibility also means that it is prone to overfit‐
ting the training set. We need regularization.

We already implemented one of the best regularization techniques in Chapter 10:
early stopping. Moreover, even though Batch Normalization was designed to solve
the vanishing/exploding gradients problems, is also acts like a pretty good regularizer.
In this section we will present other popular regularization techniques for neural net‐
works: ℓ 1 and ℓ 2 regularization, dropout and max-norm regularization.

#### ℓ 1 and ℓ 2 Regularization

Just like you did in Chapter 4 for simple linear models, you can use ℓ 1 and ℓ 2 regulari‐
zation to constrain a neural network’s connection weights (but typically not its bia‐
ses). Here is how to apply ℓ 2 regularization to a Keras layer’s connection weights,
using a regularization factor of 0.01:

```
layer = keras.layers.Dense(100, activation="elu",
kernel_initializer="he_normal",
kernel_regularizer=keras.regularizers.l2(0.01))
```

The l2() function returns a regularizer that will be called to compute the regulariza‐
tion loss, at each step during training. This regularization loss is then added to the
final loss. As you might expect, you can just use keras.regularizers.l1() if you

**356 | Chapter 11: Training Deep Neural Networks**

23 “Improving neural networks by preventing co-adaptation of feature detectors,” G. Hinton et al. (2012).
24 “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” N. Srivastava et al. (2014).

```
want ℓ 1 regularization, and if you want both ℓ 1 and ℓ 2 regularization, use keras.regu
larizers.l1_l2() (specifying both regularization factors).
Since you will typically want to apply the same regularizer to all layers in your net‐
work, as well as the same activation function and the same initialization strategy in all
hidden layers, you may find yourself repeating the same arguments over and over.
This makes it ugly and error-prone. To avoid this, you can try refactoring your code
to use loops. Another option is to use Python’s functools.partial() function: it lets
you create a thin wrapper for any callable, with some default argument values. For
example:
from functools import partial
```

```
RegularizedDense = partial(keras.layers.Dense,
activation="elu",
kernel_initializer="he_normal",
kernel_regularizer=keras.regularizers.l2(0.01))
```

```
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
RegularizedDense(300),
RegularizedDense(100),
RegularizedDense(10, activation="softmax",
kernel_initializer="glorot_uniform")
])
```

#### Dropout

```
Dropout is one of the most popular regularization techniques for deep neural net‐
works. It was proposed^23 by Geoffrey Hinton in 2012 and further detailed in a paper^24
by Nitish Srivastava et al., and it has proven to be highly successful: even the state-of-
the-art neural networks got a 1–2% accuracy boost simply by adding dropout. This
may not sound like a lot, but when a model already has 95% accuracy, getting a 2%
accuracy boost means dropping the error rate by almost 40% (going from 5% error to
roughly 3%).
It is a fairly simple algorithm: at every training step, every neuron (including the
input neurons, but always excluding the output neurons) has a probability p of being
temporarily “dropped out,” meaning it will be entirely ignored during this training
step, but it may be active during the next step (see Figure 11-9). The hyperparameter
p is called the dropout rate, and it is typically set to 50%. After training, neurons don’t
get dropped anymore. And that’s all (except for a technical detail we will discuss
momentarily).
```

```
Avoiding Overfitting Through Regularization | 357
```

Figure 11-9. Dropout regularization

It is quite surprising at first that this rather brutal technique works at all. Would a
company perform better if its employees were told to toss a coin every morning to
decide whether or not to go to work? Well, who knows; perhaps it would! The com‐
pany would obviously be forced to adapt its organization; it could not rely on any sin‐
gle person to fill in the coffee machine or perform any other critical tasks, so this
expertise would have to be spread across several people. Employees would have to
learn to cooperate with many of their coworkers, not just a handful of them. The
company would become much more resilient. If one person quit, it wouldn’t make
much of a difference. It’s unclear whether this idea would actually work for compa‐
nies, but it certainly does for neural networks. Neurons trained with dropout cannot
co-adapt with their neighboring neurons; they have to be as useful as possible on
their own. They also cannot rely excessively on just a few input neurons; they must
pay attention to each of their input neurons. They end up being less sensitive to slight
changes in the inputs. In the end you get a more robust network that generalizes bet‐
ter.

Another way to understand the power of dropout is to realize that a unique neural
network is generated at each training step. Since each neuron can be either present or
absent, there is a total of 2N possible networks (where N is the total number of drop‐
pable neurons). This is such a huge number that it is virtually impossible for the same
neural network to be sampled twice. Once you have run a 10,000 training steps, you
have essentially trained 10,000 different neural networks (each with just one training
instance). These neural networks are obviously not independent since they share
many of their weights, but they are nevertheless all different. The resulting neural
network can be seen as an averaging ensemble of all these smaller neural networks.

There is one small but important technical detail. Suppose p = 50%, in which case
during testing a neuron will be connected to twice as many input neurons as it was
(on average) during training. To compensate for this fact, we need to multiply each

**358 | Chapter 11: Training Deep Neural Networks**

25 This is specific to tf.keras, so you may prefer to use keras.backend.set_learning_phase(1) before calling
the fit() method (and set it back to 0 right after).

```
neuron’s input connection weights by 0.5 after training. If we don’t, each neuron will
get a total input signal roughly twice as large as what the network was trained on, and
it is unlikely to perform well. More generally, we need to multiply each input connec‐
tion weight by the keep probability (1 – p) after training. Alternatively, we can divide
each neuron’s output by the keep probability during training (these alternatives are
not perfectly equivalent, but they work equally well).
To implement dropout using Keras, you can use the keras.layers.Dropout layer.
During training, it randomly drops some inputs (setting them to 0) and divides the
remaining inputs by the keep probability. After training, it does nothing at all, it just
passes the inputs to the next layer. For example, the following code applies dropout
regularization before every Dense layer, using a dropout rate of 0.2:
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dropout(rate=0.2),
keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
keras.layers.Dropout(rate=0.2),
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
keras.layers.Dropout(rate=0.2),
keras.layers.Dense(10, activation="softmax")
])
```

```
Since dropout is only active during training, the training loss is
penalized compared to the validation loss, so comparing the two
can be misleading. In particular, a model may be overfitting the
training set and yet have similar training and validation losses. So
make sure to evaluate the training loss without dropout (e.g., after
training). Alternatively, you can call the fit() method inside a
with keras.backend.learning_phase_scope(1) block: this will
force dropout to be active during both training and validation.^25
```

```
If you observe that the model is overfitting, you can increase the dropout rate. Con‐
versely, you should try decreasing the dropout rate if the model underfits the training
set. It can also help to increase the dropout rate for large layers, and reduce it for
small ones. Moreover, many state-of-the-art architectures only use dropout after the
last hidden layer, so you may want to try this if full dropout is too strong.
Dropout does tend to significantly slow down convergence, but it usually results in a
much better model when tuned properly. So, it is generally well worth the extra time
and effort.
```

```
Avoiding Overfitting Through Regularization | 359
```

26 “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” Y. Gal and Z.
Ghahramani (2016).
27 Specifically, they show that training a dropout network is mathematically equivalent to approximate Bayesian
inference in a specific type of probabilistic model called a deep Gaussian Process.

```
If you want to regularize a self-normalizing network based on the
SELU activation function (as discussed earlier), you should use
AlphaDropout: this is a variant of dropout that preserves the mean
and standard deviation of its inputs (it was introduced in the same
paper as SELU, as regular dropout would break self-normalization).
```

#### Monte-Carlo (MC) Dropout

```
In 2016, a paper^26 by Yarin Gal and Zoubin Ghahramani added more good reasons to
use dropout:
```

- First, the paper establishes a profound connection between dropout networks
  (i.e., neural networks containing a dropout layer before every weight layer) and
  approximate Bayesian inference^27 , giving dropout a solid mathematical justifica‐
  tion.
- Second, they introduce a powerful technique called MC Dropout, which can
  boost the performance of any trained dropout model, without having to retrain it
  or even modify it at all!
- Moreover, MC Dropout also provides a much better measure of the model’s
  uncertainty.
- Finally, it is also amazingly simple to implement. If this all sounds like a “one
  weird trick” advertisement, then take a look at the following code. It is the full
  implementation of MC Dropout, boosting the dropout model we trained earlier,
  without retraining it:

```
with keras.backend.learning_phase_scope(1): # force training mode = dropout on
y_probas = np.stack([model.predict(X_test_scaled)
for sample in range(100)])
y_proba = y_probas.mean(axis=0)
We first force training mode on, using a learning_phase_scope(1) context. This
turns dropout on within the with block. Then we make 100 predictions over the test
set, and we stack them. Since dropout is on, all predictions will be different. Recall
that predict() returns a matrix with one row per instance, and one column per class.
Since there are 10,000 instances in the test set, and 10 classes, this is a matrix of shape
[10000, 10]. We stack 100 such matrices, so y_probas is an array of shape [100, 10000,
10]. Once we average over the first dimension (axis=0), we get y_proba, an array of
shape [10000, 10], like we would get with a single prediction. That’s all! Averaging
```

```
360 | Chapter 11: Training Deep Neural Networks
```

over multiple predictions with dropout on gives us a Monte Carlo estimate that is
generally more reliable than the result of a single prediction with dropout off. For
example, let’s look at the model’s prediction for the first instance in the test set, with
dropout off:

```
>>> np.round(model.predict(X_test_scaled[:1]), 2)
array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.01, 0. , 0.99]],
dtype=float32)
```

The model seems almost certain that this image belongs to class 9 (ankle boot).
Should you trust it? Is there really so little room for doubt? Compare this with the
predictions made when dropout is activated:

```
>>> np.round(y_probas[:, :1], 2)
array([[[0. , 0. , 0. , 0. , 0. , 0.14, 0. , 0.17, 0. , 0.68]],
[[0. , 0. , 0. , 0. , 0. , 0.16, 0. , 0.2 , 0. , 0.64]],
[[0. , 0. , 0. , 0. , 0. , 0.02, 0. , 0.01, 0. , 0.97]],
[...]
```

This tells a very different story: apparently, when we activate dropout, the model is
not sure anymore. It still seems to prefer class 9, but sometimes it hesitates with
classes 5 (sandal) and 7 (sneaker), which makes sense given they’re all footwear. Once
we average over the first dimension, we get the following MC dropout predictions:

```
>>> np.round(y_proba[:1], 2)
array([[0. , 0. , 0. , 0. , 0. , 0.22, 0. , 0.16, 0. , 0.62]],
dtype=float32)
```

The model still thinks this image belongs to class 9, but only with a 62% confidence,
which seems much more reasonable than 99%. Plus it’s useful to know exactly which
other classes it thinks are likely. And you can also take a look at the standard devia‐
tion of the probability estimates:

```
>>> y_std = y_probas.std(axis=0)
>>> np.round(y_std[:1], 2)
array([[0. , 0. , 0. , 0. , 0. , 0.28, 0. , 0.21, 0.02, 0.32]],
dtype=float32)
```

Apparently there’s quite a lot of variance in the probability estimates: if you were
building a risk-sensitive system (e.g., a medical or financial system), you should prob‐
ably treat such an uncertain prediction with extreme caution. You definitely would
not treat it like a 99% confident prediction. Moreover, the model’s accuracy got a
small boost from 86.8 to 86.9:

```
>>> accuracy = np.sum(y_pred == y_test) / len(y_test)
>>> accuracy
0.8694
```

```
Avoiding Overfitting Through Regularization | 361
```

```
The number of Monte Carlo samples you use (100 in this example)
is a hyperparameter you can tweak. The higher it is, the more accu‐
rate the predictions and their uncertainty estimates will be. How‐
ever, it you double it, inference time will also be doubled.
Moreover, above a certain number of samples, you will notice little
improvement. So your job is to find the right tradeoff between
latency and accuracy, depending on your application.
```

If your model contains other layers that behave in a special way during training (such
as Batch Normalization layers), then you should not force training mode like we just
did. Instead, you should replace the Dropout layers with the following MCDropout
class:

```
class MCDropout (keras.layers.Dropout):
def call(self, inputs):
return super().call(inputs, training=True)
```

We just sublass the Dropout layer and override the call() method to force its train
ing argument to True (see Chapter 12). Similarly, you could define an MCAlphaDrop
out class by subclassing AlphaDropout instead. If you are creating a model from
scratch, it’s just a matter of using MCDropout rather than Dropout. But if you have a
model that was already trained using Dropout, you need to create a new model, iden‐
tical to the existing model except replacing the Dropout layers with MCDropout, then
copy the existing model’s weights to your new model.

In short, MC Dropout is a fantastic technique that boosts dropout models and pro‐
vides better uncertainty estimates. And of course, since it is just regular dropout dur‐
ing training, it also acts like a regularizer.

#### Max-Norm Regularization

Another regularization technique that is quite popular for neural networks is called
max-norm regularization: for each neuron, it constrains the weights **w** of the incom‐
ing connections such that ∥ _w_ ∥ 2 ≤ _r_, where r is the max-norm hyperparameter
and ∥ · ∥ 2 is the ℓ 2 norm.

Max-norm regularization does not add a regularization loss term to the overall loss
function. Instead, it is typically implemented by computing ∥ **w** ∥ 2 after each training

step and clipping **w** if needed ( **w w** ∥ **w** r∥
2

###### ).

Reducing r increases the amount of regularization and helps reduce overfitting. Max-
norm regularization can also help alleviate the vanishing/exploding gradients prob‐
lems (if you are not using Batch Normalization).

**362 | Chapter 11: Training Deep Neural Networks**

To implement max-norm regularization in Keras, just set every hidden layer’s ker
nel_constraint argument to a max_norm() constraint, with the appropriate max
value, for example:

```
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal",
kernel_constraint=keras.constraints.max_norm(1.))
```

After each training iteration, the model’s fit() method will call the object returned
by max_norm(), passing it the layer’s weights and getting clipped weights in return,
which then replace the layer’s weights. As we will see in Chapter 12, you can define
your own custom constraint function if you ever need to, and use it as the ker
nel_constraint. You can also constrain the bias terms by setting the bias_con
straint argument.

The max_norm() function has an axis argument that defaults to 0. A Dense layer usu‐
ally has weights of shape [number of inputs, number of neurons], so using axis=0
means that the max norm constraint will apply independently to each neuron’s weight
vector. If you want to use max-norm with convolutional layers (see Chapter 14),
make sure to set the max_norm() constraint’s axis argument appropriately (usually
axis=[0, 1, 2]).

### Summary and Practical Guidelines

In this chapter, we have covered a wide range of techniques and you may be wonder‐
ing which ones you should use. The configuration in Table 11-2 will work fine in
most cases, without requiring much hyperparameter tuning.

Table 11-2. Default DNN configuration

```
Hyperparameter Default value
Kernel initializer: LeCun initialization
Activation function: SELU
Normalization: None (self-normalization)
Regularization: Early stopping
Optimizer: Nadam
Learning rate schedule: Performance scheduling
```

Don’t forget to standardize the input features! Of course, you should also try to reuse
parts of a pretrained neural network if you can find one that solves a similar problem,
or use unsupervised pretraining if you have a lot of unlabeled data, or pretraining on
an auxiliary task if you have a lot of labeled data for a similar task.

The default configuration in Table 11-2 may need to be tweaked:

```
Summary and Practical Guidelines | 363
```

- If your model self-normalizes:
  —If it overfits the training set, then you should add alpha dropout (and always
  use early stopping as well). Do not use other regularization methods, or else
  they would break self-normalization.
- If your model cannot self-normalize (e.g., it is a recurrent net or it contains skip
  connections):
  —You can try using ELU (or another activation function) instead of SELU, it
  may perform better. Make sure to change the initialization method accord‐
  ingly (e.g., He init for ELU or ReLU).
  —If it is a deep network, you should use Batch Normalization after every hidden
  layer. If it overfits the training set, you can also try using max-norm or ℓ 2 reg‐
  ularization.
- If you need a sparse model, you can use ℓ 1 regularization (and optionally zero out
  the tiny weights after training). If you need an even sparser model, you can try
  using FTRL instead of Nadam optimization, along with ℓ 1 regularization. In any
  case, this will break self-normalization, so you will need to switch to BN if your
  model is deep.
- If you need a low-latency model (one that performs lightning-fast predictions),
  you may need to use less layers, avoid Batch Normalization, and possibly replace
  the SELU activation function with the leaky ReLU. Having a sparse model will
  also help. You may also want to reduce the float precision from 32-bits to 16-bit
  (or even 8-bits) (see ???).
- If you are building a risk-sensitive application, or inference latency is not very
  important in your application, you can use MC Dropout to boost performance
  and get more reliable probability estimates, along with uncertainty estimates.

With these guidelines, you are now ready to train very deep nets! I hope you are now
convinced that you can go a very long way using just Keras. However, there may
come a time when you need to have even more control, for example to write a custom
loss function or to tweak the training algorithm. For such cases, you will need to use
TensorFlow’s lower-level API, as we will see in the next chapter.

### Exercises

1. Is it okay to initialize all the weights to the same value as long as that value is
   selected randomly using He initialization?
2. Is it okay to initialize the bias terms to 0?
3. Name three advantages of the SELU activation function over ReLU.

**364 | Chapter 11: Training Deep Neural Networks**

4. In which cases would you want to use each of the following activation functions:
   SELU, leaky ReLU (and its variants), ReLU, tanh, logistic, and softmax?
5. What may happen if you set the momentum hyperparameter too close to 1 (e.g.,
   0.99999) when using an SGD optimizer?
6. Name three ways you can produce a sparse model.
7. Does dropout slow down training? Does it slow down inference (i.e., making
   predictions on new instances)? What are about MC dropout?
8. Deep Learning.
   a.Build a DNN with five hidden layers of 100 neurons each, He initialization,
   and the ELU activation function.
   b. Using Adam optimization and early stopping, try training it on MNIST but
   only on digits 0 to 4, as we will use transfer learning for digits 5 to 9 in the
   next exercise. You will need a softmax output layer with five neurons, and as
   always make sure to save checkpoints at regular intervals and save the final
   model so you can reuse it later.
   c.Tune the hyperparameters using cross-validation and see what precision you
   can achieve.
   d.Now try adding Batch Normalization and compare the learning curves: is it
   converging faster than before? Does it produce a better model?
   e.Is the model overfitting the training set? Try adding dropout to every layer
   and try again. Does it help?
9. Transfer learning.
   a.Create a new DNN that reuses all the pretrained hidden layers of the previous
   model, freezes them, and replaces the softmax output layer with a new one.
   b. Train this new DNN on digits 5 to 9, using only 100 images per digit, and time
   how long it takes. Despite this small number of examples, can you achieve
   high precision?
   c.Try caching the frozen layers, and train the model again: how much faster is it
   now?
   d.Try again reusing just four hidden layers instead of five. Can you achieve a
   higher precision?
   e.Now unfreeze the top two hidden layers and continue training: can you get
   the model to perform even better?
10. Pretraining on an auxiliary task.

```
a.In this exercise you will build a DNN that compares two MNIST digit images
and predicts whether they represent the same digit or not. Then you will reuse
the lower layers of this network to train an MNIST classifier using very little
```

```
Exercises | 365
```

```
training data. Start by building two DNNs (let’s call them DNN A and B), both
similar to the one you built earlier but without the output layer: each DNN
should have five hidden layers of 100 neurons each, He initialization, and ELU
activation. Next, add one more hidden layer with 10 units on top of both
DNNs. To do this, you should use a keras.layers.Concatenate layer to con‐
catenate the outputs of both DNNs for each instance, then feed the result to
the hidden layer. Finally, add an output layer with a single neuron using the
logistic activation function.
b. Split the MNIST training set in two sets: split #1 should containing 55,000
images, and split #2 should contain contain 5,000 images. Create a function
that generates a training batch where each instance is a pair of MNIST images
picked from split #1. Half of the training instances should be pairs of images
that belong to the same class, while the other half should be images from dif‐
ferent classes. For each pair, the training label should be 0 if the images are
from the same class, or 1 if they are from different classes.
c.Train the DNN on this training set. For each image pair, you can simultane‐
ously feed the first image to DNN A and the second image to DNN B. The
whole network will gradually learn to tell whether two images belong to the
same class or not.
d.Now create a new DNN by reusing and freezing the hidden layers of DNN A
and adding a softmax output layer on top with 10 neurons. Train this network
on split #2 and see if you can achieve high performance despite having only
500 images per class.
```

Solutions to these exercises are available in ???.

**366 | Chapter 11: Training Deep Neural Networks**

**CHAPTER 12**

**Custom Models and Training with**

**TensorFlow**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 12 in the final
release of the book.
```

So far we have used only TensorFlow’s high level API, tf.keras, but it already got us
pretty far: we built various neural network architectures, including regression and
classification nets, wide & deep nets and self-normalizing nets, using all sorts of tech‐
niques, such as Batch Normalization, dropout, learning rate schedules, and more. In
fact, 95% of the use cases you will encounter will not require anything else than
tf.keras (and tf.data, see Chapter 13). But now it’s time to dive deeper into TensorFlow
and take a look at its lower-level Python API. This will be useful when you need extra
control, to write custom loss functions, custom metrics, layers, models, initializers,
regularizers, weight constraints and more. You may even need to fully control the
training loop itself, for example to apply special transformations or constraints to the
gradients (beyond just clipping them), or to use multiple optimizers for different
parts of the network. We will cover all these cases in this chapter, then we will also
look at how you can boost your custom models and training algorithms using Ten‐
sorFlow’s automatic graph generation feature. But first, let’s take a quick tour of Ten‐
sorFlow.

```
367
```

```
1 TensorFlow also includes another Deep Learning API called the Estimators API, but it is now recommended
to use tf.keras instead.
```

```
TensorFlow 2.0 was released in March 2019, making TensorFlow
much easier to use. The first edition of this book used TF 1, while
this edition uses TF 2.
```

### A Quick Tour of TensorFlow

As you know, TensorFlow is a powerful library for numerical computation, particu‐
larly well suited and fine-tuned for large-scale Machine Learning (but you could use
it for anything else that requires heavy computations). It was developed by the Google
Brain team and it powers many of Google’s large-scale services, such as Google Cloud
Speech, Google Photos, and Google Search. It was open sourced in November 2015,
and it is now the most popular deep learning library (in terms of citations in papers,
adoption in companies, stars on github, etc.): countless projects use TensorFlow for
all sorts of Machine Learning tasks, such as image classification, natural language
processing (NLP), recommender systems, time series forecasting, and much more.

So what does TensorFlow actually offer? Here’s a summary:

- Its core is very similar to NumPy, but with GPU support.
- It also supports distributed computing (across multiple devices and servers).
- It includes a kind of just-in-time (JIT) compiler that allows it to optimize compu‐
  tations for speed and memory usage: it works by extracting the computation
  graph from a Python function, then optimizing it (e.g., by pruning unused nodes)
  and finally running it efficiently (e.g., by automatically running independent
  operations in parallel).
- Computation graphs can be exported to a portable format, so you can train a
  TensorFlow model in one environment (e.g., using Python on Linux), and run it
  in another (e.g., using Java on an Android device).
- It implements autodiff (see Chapter 10 and ???), and provides some excellent
  optimizers, such as RMSProp, Nadam and FTRL (see Chapter 11), so you can
  easily minimize all sorts of loss functions.
- TensorFlow offers many more features, built on top of these core features: the
  most important is of course tf.keras^1 , but it also has data loading & preprocessing
  ops (tf.data, tf.io, etc.), image processing ops (tf.image), signal processing ops
  (tf.signal), and more (see Figure 12-1 for an overview of TensorFlow’s Python
  API).

**368 | Chapter 12: Custom Models and Training with TensorFlow**

```
2 If you ever need to (but you probably won’t), you can write your own operations using the C++ API.
3 If you are a researcher, you may be eligible to use these TPUs for free, see https://tensorflow.org/tfrc/ for more
details.
```

Figure 12-1. TensorFlow’s Python API

```
We will cover many of the packages and functions of the Tensor‐
Flow API, but it’s impossible to cover them all so you should really
take some time to browse through the API: you will find that it is
quite rich and well documented.
```

At the lowest level, each TensorFlow operation is implemented using highly efficient
C++ code^2. Many operations (or ops for short) have multiple implementations, called
kernels: each kernel is dedicated to a specific device type, such as CPUs, GPUs, or
even TPUs (Tensor Processing Units). As you may know, GPUs can dramatically speed
up computations by splitting computations into many smaller chunks and running
them in parallel across many GPU threads. TPUs are even faster. You can purchase
your own GPU devices (for now, TensorFlow only supports Nvidia cards with CUDA
Compute Capability 3.5+), but TPUs are only available on Google Cloud Machine
Learning Engine (see ???).^3

TensorFlow’s architecture is shown in Figure 12-2: most of the time your code will
use the high level APIs (especially tf.keras and tf.data), but when you need more flexi‐
bility you will use the lower level Python API, handling tensors directly. Note that
APIs for other languages are also available. In any case, TensorFlow’s execution

```
A Quick Tour of TensorFlow | 369
```

engine will take care of running the operations efficiently, even across multiple devi‐
ces and machines if you tell it to.

Figure 12-2. TensorFlow’s architecture

TensorFlow runs not only on Windows, Linux, and MacOS, but also on mobile devi‐
ces (using TensorFlow Lite), including both iOS and Android (see ???). If you do not
want to use the Python API, there are also C++, Java, Go and Swift APIs. There is
even a Javascript implementation called TensorFlow.js that makes it possible to run
your models directly in your browser.

There’s more to TensorFlow than just the library. TensorFlow is at the center of an
extensive ecosystem of libraries. First, there’s TensorBoard for visualization (see
Chapter 10). Next, there’s TensorFlow Extended (TFX), which is a set of libraries built
by Google to productionize TensorFlow projects: it includes tools for data validation,
preprocessing, model analysis and serving (with TF Serving, see ???). Google also
launched TensorFlow Hub, a way to easily download and reuse pretrained neural net‐
works. You can also get many neural network architectures, some of them pretrained,
in TensorFlow’s model garden. Check out the TensorFlow Resources, or https://
github.com/jtoy/awesome-tensorflow for more TensorFlow-based projects. You will
find hundreds of TensorFlow projects on GitHub, so it is often easy to find existing
code for whatever you are trying to do.

```
More and more ML papers are released along with their implemen‐
tation, and sometimes even with pretrained models. Check out
https://paperswithcode.com/ to easily find them.
```

**370 | Chapter 12: Custom Models and Training with TensorFlow**

Last but not least, TensorFlow has a dedicated team of passionate and helpful devel‐
opers, and a large community contributing to improving it. To ask technical ques‐
tions, you should use [http://stackoverflow.com/](http://stackoverflow.com/) and tag your question with tensorflow
and python. You can file bugs and feature requests through GitHub. For general dis‐
cussions, join the Google group.

Okay, it’s time to start coding!

### Using TensorFlow like NumPy

TensorFlow’s API revolves around tensors, hence the name Tensor-Flow. A tensor is
usually a multidimensional array (exactly like a NumPy ndarray), but it can also hold
a scalar (a simple value, such as 42). These tensors will be important when we create
custom cost functions, custom metrics, custom layers and more, so let’s see how to
create and manipulate them.

#### Tensors and Operations

You can easily create a tensor, using tf.constant(). For example, here is a tensor
representing a matrix with two rows and three columns of floats:

```
>>> tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
<tf.Tensor: id=0, shape=(2, 3), dtype=float32, numpy=
array([[1., 2., 3.],
[4., 5., 6.]], dtype=float32)>
>>> tf.constant(42) # scalar
<tf.Tensor: id=1, shape=(), dtype=int32, numpy=42>
```

Just like an ndarray, a tf.Tensor has a shape and a data type (dtype):

```
>>> t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
>>> t.shape
TensorShape([2, 3])
>>> t.dtype
tf.float32
```

Indexing works much like in NumPy:

```
>>> t[:, 1:]
<tf.Tensor: id=5, shape=(2, 2), dtype=float32, numpy=
array([[2., 3.],
[5., 6.]], dtype=float32)>
>>> t[..., 1, tf.newaxis]
<tf.Tensor: id=15, shape=(2, 1), dtype=float32, numpy=
array([[2.],
[5.]], dtype=float32)>
```

Most importantly, all sorts of tensor operations are available:

```
>>> t + 10
<tf.Tensor: id=18, shape=(2, 3), dtype=float32, numpy=
```

```
Using TensorFlow like NumPy | 371
```

```
array([[11., 12., 13.],
[14., 15., 16.]], dtype=float32)>
>>> tf.square(t)
<tf.Tensor: id=20, shape=(2, 3), dtype=float32, numpy=
array([[ 1., 4., 9.],
[16., 25., 36.]], dtype=float32)>
>>> t @ tf.transpose(t)
<tf.Tensor: id=24, shape=(2, 2), dtype=float32, numpy=
array([[14., 32.],
[32., 77.]], dtype=float32)>
```

Note that writing t + 10 is equivalent to calling tf.add(t, 10) (indeed, Python calls
the magic method t.**add**(10), which just calls tf.add(t, 10)). Other operators
(like -, \*, etc.) are also supported. The @ operator was added in Python 3.5, for matrix
multiplication: it is equivalent to calling the tf.matmul() function.

You will find all the basic math operations you need (e.g., tf.add(), tf.multiply(),
tf.square(), tf.exp(), tf.sqrt()...), and more generally most operations that you
can find in NumPy (e.g., tf.reshape(), tf.squeeze(), tf.tile()), but sometimes
with a different name (e.g., tf.reduce_mean(), tf.reduce_sum(), tf.reduce_max(),
tf.math.log() are the equivalent of np.mean(), np.sum(), np.max() and np.log()).
When the name differs, there is often a good reason for it: for example, in Tensor‐
Flow you must write tf.transpose(t), you cannot just write t.T like in NumPy. The
reason is that it does not do exactly the same thing: in TensorFlow, a new tensor is
created with its own copy of the transposed data, while in NumPy, t.T is just a trans‐
posed view on the same data. Similarly, the tf.reduce_sum() operation is named this
way because its GPU kernel (i.e., GPU implementation) uses a reduce algorithm that
does not guarantee the order in which the elements are added: because 32-bit floats
have limited precision, this means that the result may change ever so slightly every
time you call this operation. The same is true of tf.reduce_mean() (but of course
tf.reduce_max() is deterministic).

**372 | Chapter 12: Custom Models and Training with TensorFlow**

```
4 A notable exception is tf.math.log() which is commonly used but there is no tf.log() alias (as it might be
confused with logging).
```

```
Many functions and classes have aliases. For example, tf.add()
and tf.math.add() are the same function. This allows TensorFlow
to have concise names for the most common operations^4 , while
preserving well organized packages.
```

```
Keras’ Low-Level API
The Keras API actually has its own low-level API, located in keras.backend. It
includes functions like square(), exp(), sqrt() and so on. In tf.keras, these func‐
tions generally just call the corresponding TensorFlow operations. If you want to
write code that will be portable to other Keras implementations, you should use these
Keras functions. However, they only cover a subset of all functions available in Ten‐
sorFlow, so in this book we will use the TensorFlow operations directly. Here is as
simple example using keras.backend, which is commonly named K for short:
>>> from tensorflow import keras
>>> K = keras.backend
>>> K.square(K.transpose(t)) + 10
<tf.Tensor: id=39, shape=(3, 2), dtype=float32, numpy=
array([[11., 26.],
[14., 35.],
[19., 46.]], dtype=float32)>
```

#### Tensors and NumPy

Tensors play nice with NumPy: you can create a tensor from a NumPy array, and vice
versa, and you can even apply TensorFlow operations to NumPy arrays and NumPy
operations to tensors:

```
>>> a = np.array([2., 4., 5.])
>>> tf.constant(a)
<tf.Tensor: id=111, shape=(3,), dtype=float64, numpy=array([2., 4., 5.])>
>>> t.numpy() # or np.array(t)
array([[1., 2., 3.],
[4., 5., 6.]], dtype=float32)
>>> tf.square(a)
<tf.Tensor: id=116, shape=(3,), dtype=float64, numpy=array([4., 16., 25.])>
>>> np.square(t)
array([[ 1., 4., 9.],
[16., 25., 36.]], dtype=float32)
```

```
Using TensorFlow like NumPy | 373
```

```
Notice that NumPy uses 64-bit precision by default, while Tensor‐
Flow uses 32-bit. This is because 32-bit precision is generally more
than enough for neural networks, plus it runs faster and uses less
RAM. So when you create a tensor from a NumPy array, make sure
to set dtype=tf.float32.
```

#### Type Conversions

Type conversions can significantly hurt performance, and they can easily go unno‐
ticed when they are done automatically. To avoid this, TensorFlow does not perform
any type conversions automatically: it just raises an exception if you try to execute an
operation on tensors with incompatible types. For example, you cannot add a float
tensor and an integer tensor, and you cannot even add a 32-bit float and a 64-bit float:

```
>>> tf.constant(2.) + tf.constant(40)
Traceback[...]InvalidArgumentError[...]expected to be a float[...]
>>> tf.constant(2.) + tf.constant(40., dtype=tf.float64)
Traceback[...]InvalidArgumentError[...]expected to be a double[...]
```

This may be a bit annoying at first, but remember that it’s for a good cause! And of
course you can use tf.cast() when you really need to convert types:

```
>>> t2 = tf.constant(40., dtype=tf.float64)
>>> tf.constant(2.0) + tf.cast(t2, tf.float32)
<tf.Tensor: id=136, shape=(), dtype=float32, numpy=42.0>
```

#### Variables

So far, we have used constant tensors: as their name suggests, you cannot modify
them. However, the weights in a neural network need to be tweaked by backpropaga‐
tion, and other parameters may also need to change over time (e.g., a momentum
optimizer keeps track of past gradients). What we need is a tf.Variable:

```
>>> v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
>>> v
<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[1., 2., 3.],
[4., 5., 6.]], dtype=float32)>
```

A tf.Variable acts much like a constant tensor: you can perform the same opera‐
tions with it, it plays nicely with NumPy as well, and it is just as picky with types. But
it can also be modified in place using the assign() method (or assign_add() or
assign_sub() which increment or decrement the variable by the given value). You
can also modify individual cells (or slices), using the cell’s (or slice’s) assign()
method (direct item assignment will not work), or using the scatter_update() or
scatter_nd_update() methods:

```
v.assign(2 * v) # => [[2., 4., 6.], [8., 10., 12.]]
v[0, 1].assign(42) # => [[2., 42., 6.], [8., 10., 12.]]
```

**374 | Chapter 12: Custom Models and Training with TensorFlow**

```
v[:, 2].assign([0., 1.]) # => [[2., 42., 0.], [8., 10., 1.]]
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])
# => [[100., 42., 0.], [8., 10., 200.]]
```

```
In practice you will rarely have to create variables manually, since
Keras provides an add_weight() method that will take care of it for
you, as we will see. Moreover, model parameters will generally be
updated directly by the optimizers, so you will rarely need to
update variables manually.
```

#### Other Data Structures

TensorFlow supports several other data structures, including the following (please see
the notebook or ??? for more details):

- Sparse tensors (tf.SparseTensor) efficiently represent tensors containing mostly
  0s. The tf.sparse package contains operations for sparse tensors.
- Tensor arrays (tf.TensorArray) are lists of tensors. They have a fixed size by
  default, but can optionally be made dynamic. All tensors they contain must have
  the same shape and data type.
- Ragged tensors (tf.RaggedTensor) represent static lists of lists of tensors, where
  every tensor has the same shape and data type. The tf.ragged package contains
  operations for ragged tensors.
- String tensors are regular tensors of type tf.string. These actually represent byte
  strings, not Unicode strings, so if you create a string tensor using a Unicode
  string (e.g., a regular Python 3 string like "café"`), then it will get encoded to
  UTF-8 automatically (e.g., b"caf\xc3\xa9"). Alternatively, you can represent
  Unicode strings using tensors of type tf.int32, where each item represents a
  Unicode codepoint (e.g., [99, 97, 102, 233]). The tf.strings package (with
  an s) contains ops for byte strings and Unicode strings (and to convert one into
  the other).
- Sets are just represented as regular tensors (or sparse tensors) containing one or
  more sets, and you can manipulate them using operations from the tf.sets
  package.
- Queues, including First In, First Out (FIFO) queues (FIFOQueue), queues that can
  prioritize some items (PriorityQueue), queues that shuffle their items (Random
  ShuffleQueue), and queues that can batch items of different shapes by padding
  (PaddingFIFOQueue). These classes are all in the tf.queue package.

With tensors, operations, variables and various data structures at your disposal, you
are now ready to customize your models and training algorithms!

```
Using TensorFlow like NumPy | 375
```

### Customizing Models and Training Algorithms

Let’s start by creating a custom loss function, which is a simple and common use case.

#### Custom Loss Functions

Suppose you want to train a regression model, but your training set is a bit noisy. Of
course, you start by trying to clean up your dataset by removing or fixing the outliers,
but it turns out to be insufficient, the dataset is still noisy. Which loss function should
you use? The mean squared error might penalize large errors too much, so your
model will end up being imprecise. The mean absolute error would not penalize out‐
liers as much, but training might take a while to converge and the trained model
might not be very precise. This is probably a good time to use the Huber loss (intro‐
duced in Chapter 10) instead of the good old MSE. The Huber loss is not currently
part of the official Keras API, but it is available in tf.keras (just use an instance of the
keras.losses.Huber class). But let’s pretend it’s not there: implementing it is easy as
pie! Just create a function that takes the labels and predictions as arguments, and use
TensorFlow operations to compute every instance’s loss:

```
def huber_fn(y_true, y_pred):
error = y_true - y_pred
is_small_error = tf.abs(error) < 1
squared_loss = tf.square(error) / 2
linear_loss = tf.abs(error) - 0.5
return tf.where(is_small_error, squared_loss, linear_loss)
```

```
For better performance, you should use a vectorized implementa‐
tion, as in this example. Moreover, if you want to benefit from Ten‐
sorFlow’s graph features, you should use only TensorFlow
operations.
```

It is also preferable to return a tensor containing one loss per instance, rather than
returning the mean loss. This way, Keras can apply class weights or sample weights
when requested (see Chapter 10).

Next, you can just use this loss when you compile the Keras model, then train your
model:

```
model.compile(loss=huber_fn, optimizer="nadam")
model.fit(X_train, y_train, [...])
```

And that’s it! For each batch during training, Keras will call the huber_fn() function
to compute the loss, and use it to perform a Gradient Descent step. Moreover, it will
keep track of the total loss since the beginning of the epoch, and it will display the
mean loss.

**376 | Chapter 12: Custom Models and Training with TensorFlow**

But what happens to this custom loss when we save the model?

#### Saving and Loading Models That Contain Custom Components

Saving a model containing a custom loss function actually works fine, as Keras just
saves the name of the function. However, whenever you load it, you need to provide a
dictionary that maps the function name to the actual function. More generally, when
you load a model containing custom objects, you need to map the names to the
objects:

```
model = keras.models.load_model("my_model_with_a_custom_loss.h5",
custom_objects={"huber_fn": huber_fn})
```

With the current implementation, any error between -1 and 1 is considered “small”.
But what if we want a different threshold? One solution is to create a function that
creates a configured loss function:

```
def create_huber(threshold=1.0):
def huber_fn(y_true, y_pred):
error = y_true - y_pred
is_small_error = tf.abs(error) < threshold
squared_loss = tf.square(error) / 2
linear_loss = threshold * tf.abs(error) - threshold**2 / 2
return tf.where(is_small_error, squared_loss, linear_loss)
return huber_fn
model.compile(loss=create_huber(2.0), optimizer="nadam")
```

Unfortunately, when you save the model, the threshold will not be saved. This means
that you will have to specify the threshold value when loading the model (note that
the name to use is "huber_fn", which is the name of the function we gave Keras, not
the name of the function that created it):

```
model = keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5",
custom_objects={"huber_fn": create_huber(2.0)})
```

You can solve this by creating a subclass of the keras.losses.Loss class, and imple‐
ment its get_config() method:

```
class HuberLoss (keras.losses.Loss):
def __init__ (self, threshold=1.0, **kwargs):
self.threshold = threshold
super(). __init__ (**kwargs)
def call(self, y_true, y_pred):
error = y_true - y_pred
is_small_error = tf.abs(error) < self.threshold
squared_loss = tf.square(error) / 2
linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
return tf.where(is_small_error, squared_loss, linear_loss)
def get_config(self):
base_config = super().get_config()
return {**base_config, "threshold": self.threshold}
```

```
Customizing Models and Training Algorithms | 377
```

```
5 It would not be a good idea to use a weighted mean: if we did, then two instances with the same weight but in
different batches would have a different impact on training, depending on the total weight of each batch.
```

```
The Keras API only specifies how to use subclassing to define lay‐
ers, models, callbacks, and regularizers. If you build other compo‐
nents (such as losses, metrics, initializers or constraints) using
subclassing, they may not be portable to other Keras implementa‐
tions.
```

Let’s walk through this code:

- The constructor accepts \*\*kwargs and passes them to the parent constructor,
  which handles standard hyperparameters: the name of the loss and the reduction
  algorithm to use to aggregate the individual instance losses. By default, it is
  "sum_over_batch_size", which means that the loss will be the sum of the
  instance losses, possibly weighted by the sample weights, if any, and then divide
  the result by the batch size (not by the sum of weights, so this is not the weighted
  mean).^5. Other possible values are "sum" and None.
- The call() method takes the labels and predictions, computes all the instance
  losses, and returns them.
- The get_config() method returns a dictionary mapping each hyperparameter
  name to its value. It first calls the parent class’s get_config() method, then adds
  the new hyperparameters to this dictionary (note that the convenient {\*\*x} syn‐
  tax was added in Python 3.5).

You can then use any instance of this class when you compile the model:

```
model.compile(loss=HuberLoss(2.), optimizer="nadam")
```

When you save the model, the threshold will be saved along with it, and when you
load the model you just need to map the class name to the class itself:

```
model = keras.models.load_model("my_model_with_a_custom_loss_class.h5",
custom_objects={"HuberLoss": HuberLoss})
```

When you save a model, Keras calls the loss instance’s get_config() method and
saves the config as JSON in the HDF5 file. When you load the model, it calls the
from_config() class method on the HuberLoss class: this method is implemented by
the base class (Loss) and just creates an instance of the class, passing \*\*config to the
constructor.

That’s it for losses! It was not too hard, was it? Well it’s just as simple for custom acti‐
vation functions, initializers, regularizers, and constraints. Let’s look at these now.

**378 | Chapter 12: Custom Models and Training with TensorFlow**

#### Custom Activation Functions, Initializers, Regularizers, and Constraints

Most Keras functionalities, such as losses, regularizers, constraints, initializers, met‐
rics, activation functions, layers and even full models can be customized in very much
the same way. Most of the time, you will just need to write a simple function, with the
appropriate inputs and outputs. For example, here are examples of a custom activa‐
tion function (equivalent to keras.activations.softplus or tf.nn.softplus), a
custom Glorot initializer (equivalent to keras.initializers.glorot_normal), a cus‐
tom ℓ 1 regularizer (equivalent to keras.regularizers.l1(0.01)) and a custom con‐
straint that ensures weights are all positive (equivalent to
keras.constraints.nonneg() or tf.nn.relu):

```
def my_softplus(z): # return value is just tf.nn.softplus(z)
return tf.math.log(tf.exp(z) + 1.0)
```

```
def my_glorot_initializer(shape, dtype=tf.float32):
stddev = tf.sqrt(2. / (shape[0] + shape[1]))
return tf.random.normal(shape, stddev=stddev, dtype=dtype)
```

```
def my_l1_regularizer(weights):
return tf.reduce_sum(tf.abs(0.01 * weights))
```

```
def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
return tf.where(weights < 0., tf.zeros_like(weights), weights)
```

As you can see, the arguments depend on the type of custom function. These custom
functions can then be used normally, for example:

```
layer = keras.layers.Dense(30, activation=my_softplus,
kernel_initializer=my_glorot_initializer,
kernel_regularizer=my_l1_regularizer,
kernel_constraint=my_positive_weights)
```

The activation function will be applied to the output of this Dense layer, and its result
will be passed on to the next layer. The layer’s weights will be initialized using the
value returned by the initializer. At each training step the weights will be passed to the
regularization function to compute the regularization loss, which will be added to the
main loss to get the final loss used for training. Finally, the constraint function will be
called after each training step, and the layer’s weights will be replaced by the con‐
strained weights.

If a function has some hyperparameters that need to be saved along with the model,
then you will want to subclass the appropriate class, such as keras.regulariz
ers.Regularizer, keras.constraints.Constraint, keras.initializers.Initial
izer or keras.layers.Layer (for any layer, including activation functions). For
example, much like we did for the custom loss, here is a simple class for ℓ 1 regulariza‐

```
Customizing Models and Training Algorithms | 379
```

```
6 However, the Huber loss is seldom used as a metric (the MAE or MSE are preferred).
```

tion, that saves its factor hyperparameter (this time we do not need to call the parent
constructor or the get_config() method, as they are not defined by the parent class):

```
class MyL1Regularizer (keras.regularizers.Regularizer):
def __init__ (self, factor):
self.factor = factor
def __call__ (self, weights):
return tf.reduce_sum(tf.abs(self.factor * weights))
def get_config(self):
return {"factor": self.factor}
```

Note that you must implement the call() method for losses, layers (including activa‐
tion functions) and models, or the **call**() method for regularizers, initializers
and constraints. For metrics, things are a bit different, as we will see now.

#### Custom Metrics

Losses and metrics are conceptually not the same thing: losses are used by Gradient
Descent to train a model, so they must be differentiable (at least where they are evalu‐
ated) and their gradients should not be 0 everywhere. Plus, it’s okay if they are not
easily interpretable by humans (e.g. cross-entropy). In contrast, metrics are used to
evaluate a model, they must be more easily interpretable, and they can be non-
differentiable or have 0 gradients everywhere (e.g., accuracy).

That said, in most cases, defining a custom metric function is exactly the same as
defining a custom loss function. In fact, we could even use the Huber loss function we
created earlier as a metric^6 , it would work just fine (and persistence would also work
the same way, in this case only saving the name of the function, "huber_fn"):

```
model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
```

For each batch during training, Keras will compute this metric and keep track of its
mean since the beginning of the epoch. Most of the time, this is exactly what you
want. But not always! Consider a binary classifier’s precision, for example. As we saw
in Chapter 3, precision is the number of true positives divided by the number of posi‐
tive predictions (including both true positives and false positives). Suppose the model
made 5 positive predictions in the first batch, 4 of which were correct: that’s 80% pre‐
cision. Then suppose the model made 3 positive predictions in the second batch, but
they were all incorrect: that’s 0% precision for the second batch. If you just compute
the mean of these two precisions, you get 40%. But wait a second, this is not the mod‐
el’s precision over these two batches! Indeed, there were a total of 4 true positives (4 + 0) out of 8 positive predictions (5 + 3), so the overall precision is 50%, not 40%. What
we need is an object that can keep track of the number of true positives and the num‐

**380 | Chapter 12: Custom Models and Training with TensorFlow**

ber of false positives, and compute their ratio when requested. This is precisely what
the keras.metrics.Precision class does:

```
>>> precision = keras.metrics.Precision()
>>> precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
<tf.Tensor: id=581729, shape=(), dtype=float32, numpy=0.8>
>>> precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
<tf.Tensor: id=581780, shape=(), dtype=float32, numpy=0.5>
```

In this example, we created a Precision object, then we used it like a function, pass‐
ing it the labels and predictions for the first batch, then for the second batch (note
that we could also have passed sample weights). We used the same number of true
and false positives as in the example we just discussed. After the first batch, it returns
the precision of 80%, then after the second batch it returns 50% (which is the overall
precision so far, not the second batch’s precision). This is called a streaming metric (or
stateful metric), as it is gradually updated, batch after batch.

At any point, we can call the result() method to get the current value of the metric.
We can also look at its variables (tracking the number of true and false positives)
using the variables attribute, and reset these variables using the reset_states()
method:

```
>>> p.result()
<tf.Tensor: id=581794, shape=(), dtype=float32, numpy=0.5>
>>> p.variables
[<tf.Variable 'true_positives:0' [...] numpy=array([4.], dtype=float32)>,
<tf.Variable 'false_positives:0' [...] numpy=array([4.], dtype=float32)>]
>>> p.reset_states() # both variables get reset to 0.0
```

If you need to create such a streaming metric, you can just create a subclass of the
keras.metrics.Metric class. Here is a simple example that keeps track of the total
Huber loss and the number of instances seen so far. When asked for the result, it
returns the ratio, which is simply the mean Huber loss:

```
class HuberMetric (keras.metrics.Metric):
def __init__ (self, threshold=1.0, **kwargs):
super(). __init__ (**kwargs) # handles base args (e.g., dtype)
self.threshold = threshold
self.huber_fn = create_huber(threshold)
self.total = self.add_weight("total", initializer="zeros")
self.count = self.add_weight("count", initializer="zeros")
def update_state(self, y_true, y_pred, sample_weight=None):
metric = self.huber_fn(y_true, y_pred)
self.total.assign_add(tf.reduce_sum(metric))
self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
def result(self):
return self.total / self.count
def get_config(self):
base_config = super().get_config()
return {**base_config, "threshold": self.threshold}
```

```
Customizing Models and Training Algorithms | 381
```

```
7 This class is for illustration purposes only. A simpler and better implementation would just subclass the
keras.metrics.Mean class, see the notebook for an example.
```

Let’s walk through this code:^7 :

- The constructor uses the add_weight() method to create the variables needed to
  keep track of the metric’s state over multiple batches, in this case the sum of all
  Huber losses (total) and the number of instances seen so far (count). You could
  just create variables manually if you preferred. Keras tracks any tf.Variable that
  is set as an attribute (and more generally, any “trackable” object, such as layers or
  models).
- The update_state() method is called when you use an instance of this class as a
  function (as we did with the Precision object). It updates the variables given the
  labels and predictions for one batch (and sample weights, but in this case we just
  ignore them).
- The result() method computes and returns the final result, in this case just the
  mean Huber metric over all instances. When you use the metric as a function, the
  update_state() method gets called first, then the result() method is called,
  and its output is returned.
- We also implement the get_config() method to ensure the threshold gets
  saved along with the model.
- The default implementation of the reset_states() method just resets all vari‐
  ables to 0.0 (but you can override it if needed).

```
Keras will take care of variable persistence seamlessly, no action is
required.
```

When you define a metric using a simple function, Keras automatically calls it for
each batch, and it keeps track of the mean during each epoch, just like we did man‐
ually. So the only benefit of our HuberMetric class is that the threshold will be saved.
But of course, some metrics, like precision, cannot simply be averaged over batches:
in thoses cases, there’s no other option than to implement a streaming metric.

Now that we have built a streaming metric, building a custom layer will seem like a
walk in the park!

**382 | Chapter 12: Custom Models and Training with TensorFlow**

#### Custom Layers

You may occasionally want to build an architecture that contains an exotic layer for
which TensorFlow does not provide a default implementation. In this case, you will
need to create a custom layer. Or sometimes you may simply want to build a very
repetitive architecture, containing identical blocks of layers repeated many times, and
it would be convenient to treat each block of layers as a single layer. For example, if
the model is a sequence of layers A, B, C, A, B, C, A, B, C, then you might want to
define a custom layer D containing layers A, B, C, and your model would then simply
be D, D, D. Let’s see how to build custom layers.

First, some layers have no weights, such as keras.layers.Flatten or keras.lay
ers.ReLU. If you want to create a custom layer without any weights, the simplest
option is to write a function and wrap it in a keras.layers.Lambda layer. For exam‐
ple, the following layer will apply the exponential function to its inputs:

```
exponential_layer = keras.layers.Lambda( lambda x: tf.exp(x))
```

This custom layer can then be used like any other layer, using the sequential API, the
functional API, or the subclassing API. You can also use it as an activation function
(or you could just use activation=tf.exp, or activation=keras.activations.expo
nential, or simply activation="exponential"). The exponential layer is sometimes
used in the output layer of a regression model when the values to predict have very
different scales (e.g., 0.001, 10., 1000.).

As you probably guessed by now, to build a custom stateful layer (i.e., a layer with
weights), you need to create a subclass of the keras.layers.Layer class. For exam‐
ple, the following class implements a simplified version of the Dense layer:

```
class MyDense (keras.layers.Layer):
def __init__ (self, units, activation=None, **kwargs):
super(). __init__ (**kwargs)
self.units = units
self.activation = keras.activations.get(activation)
```

```
def build(self, batch_input_shape):
self.kernel = self.add_weight(
name="kernel", shape=[batch_input_shape[-1], self.units],
initializer="glorot_normal")
self.bias = self.add_weight(
name="bias", shape=[self.units], initializer="zeros")
super().build(batch_input_shape) # must be at the end
```

```
def call(self, X):
return self.activation(X @ self.kernel + self.bias)
```

```
def compute_output_shape(self, batch_input_shape):
return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
```

```
Customizing Models and Training Algorithms | 383
```

```
8 This function is specific to tf.keras. You could use keras.activations.Activation instead.
9 The Keras API calls this argument input_shape, but since it also includes the batch dimension, I prefer to call
it batch_input_shape. Same for compute_output_shape().
```

```
def get_config(self):
base_config = super().get_config()
return {**base_config, "units": self.units,
"activation": keras.activations.serialize(self.activation)}
```

Let’s walk through this code:

- The constructor takes all the hyperparameters as arguments (in this example just
  units and activation), and importantly it also takes a \*\*kwargs argument. It
  calls the parent constructor, passing it the kwargs: this takes care of standard
  arguments such as input_shape, trainable, name, and so on. Then it saves the
  hyperparameters as attributes, converting the activation argument to the
  appropriate activation function using the keras.activations.get() function (it
  accepts functions, standard strings like "relu" or "selu", or simply None)^8.
- The build() method’s role is to create the layer’s variables, by calling the
  add_weight() method for each weight. The build() method is called the first
  time the layer is used. At that point, Keras will know the shape of this layer’s
  inputs, and it will pass it to the build() method^9 , which is often necessary to cre‐
  ate some of the weights. For example, we need to know the number of neurons in
  the previous layer in order to create the connection weights matrix (i.e., the "ker
  nel"): this corresponds to the size of the last dimension of the inputs. At the end
  of the build() method (and only at the end), you must call the parent’s build()
  method: this tells Keras that the layer is built (it just sets self.built = True).
- The call() method actually performs the desired operations. In this case, we
  compute the matrix multiplication of the inputs X and the layer’s kernel, we add
  the bias vector, we apply the activation function to the result, and this gives us the
  output of the layer.
- The compute_output_shape() method simply returns the shape of this layer’s
  outputs. In this case, it is the same shape as the inputs, except the last dimension
  is replaced with the number of neurons in the layer. Note that in tf.keras, shapes
  are instances of the tf.TensorShape class, which you can convert to Python lists
  using as_list().
- The get_config() method is just like earlier. Note that we save the activation
  function’s full configuration by calling keras.activations.serialize().

You can now use a MyDense layer just like any other layer!

**384 | Chapter 12: Custom Models and Training with TensorFlow**

```
You can generally omit the compute_output_shape() method, as
tf.keras automatically infers the output shape, except when the
layer is dynamic (as we will see shortly). In other Keras implemen‐
tations, this method is either required or by default it assumes the
output shape is the same as the input shape.
```

To create a layer with multiple inputs (e.g., Concatenate), the argument to the call()
method should be a tuple containing all the inputs, and similarly the argument to the
compute_output_shape() method should be a tuple containing each input’s batch
shape. To create a layer with multiple outputs, the call() method should return the
list of outputs, and the compute_output_shape() should return the list of batch out‐
put shapes (one per output). For example, the following toy layer takes two inputs
and returns three outputs:

```
class MyMultiLayer (keras.layers.Layer):
def call(self, X):
X1, X2 = X
return [X1 + X2, X1 * X2, X1 / X2]
```

```
def compute_output_shape(self, batch_input_shape):
b1, b2 = batch_input_shape
return [b1, b1, b1] # should probably handle broadcasting rules
```

This layer may now be used like any other layer, but of course only using the func‐
tional and subclassing APIs, not the sequential API (which only accepts layers with
one input and one output).

If your layer needs to have a different behavior during training and during testing
(e.g., if it uses Dropout or BatchNormalization layers), then you must add a train
ing argument to the call() method and use this argument to decide what to do. For
example, let’s create a layer that adds Gaussian noise during training (for regulariza‐
tion), but does nothing during testing (Keras actually has a layer that does the same
thing: keras.layers.GaussianNoise):

```
class MyGaussianNoise (keras.layers.Layer):
def __init__ (self, stddev, **kwargs):
super(). __init__ (**kwargs)
self.stddev = stddev
```

```
def call(self, X, training=None):
if training:
noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
return X + noise
else :
return X
```

```
def compute_output_shape(self, batch_input_shape):
return batch_input_shape
```

```
Customizing Models and Training Algorithms | 385
```

10 The name “subclassing API” usually refers only to the creation of custom models by subclassing, although
many other things can be created by subclassing, as we saw in this chapter.

```
With that, you can now build any custom layer you need! Now let’s create custom
models.
```

#### Custom Models

```
We already looked at custom model classes in Chapter 10 when we discussed the sub‐
classing API.^10 It is actually quite straightforward, just subclass the keras.mod
els.Model class, create layers and variables in the constructor, and implement the
call() method to do whatever you want the model to do. For example, suppose you
want to build the model represented in Figure 12-3:
```

```
Figure 12-3. Custom Model Example
```

```
The inputs go through a first dense layer, then through a residual block composed of
two dense layers and an addition operation (as we will see in Chapter 14, a residual
block adds its inputs to its outputs), then through this same residual block 3 more
times, then through a second residual block, and the final result goes through a dense
output layer. Note that this model does not make much sense, it’s just an example to
illustrate the fact that you can easily build any kind of model you want, even contain‐
```

```
386 | Chapter 12: Custom Models and Training with TensorFlow
```

ing loops and skip connections. To implement this model, it is best to first create a
ResidualBlock layer, since we are going to create a couple identical blocks (and we
might want to reuse it in another model):

```
class ResidualBlock (keras.layers.Layer):
def __init__ (self, n_layers, n_neurons, **kwargs):
super(). __init__ (**kwargs)
self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
kernel_initializer="he_normal")
for _ in range(n_layers)]
```

```
def call(self, inputs):
Z = inputs
for layer in self.hidden:
Z = layer(Z)
return inputs + Z
```

This layer is a bit special since it contains other layers. This is handled transparently
by Keras: it automatically detects that the hidden attribute contains trackable objects
(layers in this case), so their variables are automatically added to this layer’s list of
variables. The rest of this class is self-explanatory. Next, let’s use the subclassing API
to define the model itself:

```
class ResidualRegressor (keras.models.Model):
def __init__ (self, output_dim, **kwargs):
super(). __init__ (**kwargs)
self.hidden1 = keras.layers.Dense(30, activation="elu",
kernel_initializer="he_normal")
self.block1 = ResidualBlock(2, 30)
self.block2 = ResidualBlock(2, 30)
self.out = keras.layers.Dense(output_dim)
```

```
def call(self, inputs):
Z = self.hidden1(inputs)
for _ in range(1 + 3):
Z = self.block1(Z)
Z = self.block2(Z)
return self.out(Z)
```

We create the layers in the constructor, and use them in the call() method. This
model can then be used like any other model (compile it, fit it, evaluate it and use it to
make predictions). If you also want to be able to save the model using the save()
method, and load it using the keras.models.load_model() function, you must
implement the get_config() method (as we did earlier) in both the ResidualBlock
class and the ResidualRegressor class. Alternatively, you can just save and load the
weights using the save_weights() and load_weights() methods.

The Model class is actually a subclass of the Layer class, so models can be defined and
used exactly like layers. But a model also has some extra functionalities, including of
course its compile(), fit(), evaluate() and predict() methods (and a few var‐

```
Customizing Models and Training Algorithms | 387
```

iants, such as train_on_batch() or fit_generator()), plus the get_layers()
method (which can return any of the model’s layers by name or by index), and the
save() method (and support for keras.models.load_model() and keras.mod
els.clone_model()). So if models provide more functionalities than layers, why not
just define every layer as a model? Well, technically you could, but it is probably
cleaner to distinguish the internal components of your model (layers or reusable
blocks of layers) from the model itself. The former should subclass the Layer class,
while the latter should subclass the Model class.

With that, you can quite naturally and concisely build almost any model that you find
in a paper, either using the sequential API, the functional API, the subclassing API, or
even a mix of these. “Almost” any model? Yes, there are still a couple things that we
need to look at: first, how to define losses or metrics based on model internals, and
second how to build a custom training loop.

#### Losses and Metrics Based on Model Internals

The custom losses and metrics we defined earlier were all based on the labels and the
predictions (and optionally sample weights). However, you will occasionally want to
define losses based on other parts of your model, such as the weights or activations of
its hidden layers. This may be useful for regularization purposes, or to monitor some
internal aspect of your model.

To define a custom loss based on model internals, just compute it based on any part
of the model you want, then pass the result to the add_loss() method. For example,
the following custom model represents a standard MLP regressor with 5 hidden lay‐
ers, except it also implements a reconstruction loss (see ???): we add an extra Dense
layer on top of the last hidden layer, and its role is to try to reconstruct the inputs of
the model. Since the reconstruction must have the same shape as the model’s inputs,
we need to create this Dense layer in the build() method to have access to the shape
of the inputs. In the call() method, we compute both the regular output of the MLP,
plus the output of the reconstruction layer. We then compute the mean squared dif‐
ference between the reconstructions and the inputs, and we add this value (times
0.05) to the model’s list of losses by calling add_loss(). During training, Keras will
add this loss to the main loss (which is why we scaled down the reconstruction loss,
to ensure the main loss dominates). As a result, the model will be forced to preserve
as much information as possible through the hidden layers, even information that is
not directly useful for the regression task itself. In practice, this loss sometimes
improves generalization; it is a regularization loss:

```
class ReconstructingRegressor (keras.models.Model):
def __init__ (self, output_dim, **kwargs):
super(). __init__ (**kwargs)
self.hidden = [keras.layers.Dense(30, activation="selu",
kernel_initializer="lecun_normal")
```

**388 | Chapter 12: Custom Models and Training with TensorFlow**

```
for _ in range(5)]
self.out = keras.layers.Dense(output_dim)
```

```
def build(self, batch_input_shape):
n_inputs = batch_input_shape[-1]
self.reconstruct = keras.layers.Dense(n_inputs)
super().build(batch_input_shape)
```

```
def call(self, inputs):
Z = inputs
for layer in self.hidden:
Z = layer(Z)
reconstruction = self.reconstruct(Z)
recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
self.add_loss(0.05 * recon_loss)
return self.out(Z)
```

Similarly, you can add a custom metric based on model internals by computing it in
any way you want, as long at the result is the output of a metric object. For example,
you can create a keras.metrics.Mean() object in the constructor, then call it in the
call() method, passing it the recon_loss, and finally add it to the model by calling
the model’s add_metric() method. This way, when you train the model, Keras will
display both the mean loss over each epoch (the loss is the sum of the main loss plus
0.05 times the reconstruction loss) and the mean reconstruction error over each
epoch. Both will go down during training:

```
Epoch 1/5
11610/11610 [=============] [...] loss: 4.3092 - reconstruction_error: 1.7360
Epoch 2/5
11610/11610 [=============] [...] loss: 1.1232 - reconstruction_error: 0.8964
[...]
```

In over 99% of the cases, everything we have discussed so far will be sufficient to
implement whatever model you want to build, even with complex architectures, los‐
ses, metrics, and so on. However, in some rare cases you may need to customize the
training loop itself. However, before we get there, we need to look at how to compute
gradients automatically in TensorFlow.

#### Computing Gradients Using Autodiff

To understand how to use autodiff (see Chapter 10 and ???) to compute gradients
automatically, let’s consider a simple toy function:

```
def f(w1, w2):
return 3 * w1 ** 2 + 2 * w1 * w2
```

If you know calculus, you can analytically find that the partial derivative of this func‐
tion with regards to w1 is 6 _ w1 + 2 _ w2. You can also find that its partial derivative
with regards to w2 is 2 \* w1. For example, at the point (w1, w2) = (5, 3), these par‐

```
Customizing Models and Training Algorithms | 389
```

tial derivatives are equal to 36 and 10, respectively, so the gradient vector at this point
is (36, 10). But if this were a neural network, the function would be much more com‐
plex, typically with tens of thousands of parameters, and finding the partial deriva‐
tives analytically by hand would be an almost impossible task. One solution could be
to compute an approximation of each partial derivative by measuring how much the
function’s output changes when you tweak the corresponding parameter:

```
>>> w1, w2 = 5, 3
>>> eps = 1e-6
>>> (f(w1 + eps, w2) - f(w1, w2)) / eps
36.000003007075065
>>> (f(w1, w2 + eps) - f(w1, w2)) / eps
10.000000003174137
```

Looks about right! This works rather well and it is trivial to implement, but it is just
an approximation, and importantly you need to call f() at least once per parameter
(not twice, since we could compute f(w1, w2) just once). This makes this approach
intractable for large neural networks. So instead we should use autodiff (see Chap‐
ter 10 and ???). TensorFlow makes this pretty simple:

```
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
z = f(w1, w2)
```

```
gradients = tape.gradient(z, [w1, w2])
```

We first define two variables w1 and w2, then we create a tf.GradientTape context
that will automatically record every operation that involves a variable, and finally we
ask this tape to compute the gradients of the result z with regards to both variables
[w1, w2]. Let’s take a look at the gradients that TensorFlow computed:

```
>>> gradients
[<tf.Tensor: id=828234, shape=(), dtype=float32, numpy=36.0>,
<tf.Tensor: id=828229, shape=(), dtype=float32, numpy=10.0>]
```

Perfect! Not only is the result accurate (the precision is only limited by the floating
point errors), but the gradient() method only goes through the recorded computa‐
tions once (in reverse order), no matter how many variables there are, so it is incredi‐
bly efficient. It’s like magic!

```
Only put the strict minimum inside the tf.GradientTape() block,
to save memory. Alternatively, you can pause recording by creating
a with tape.stop_recording() block inside the tf.Gradient
Tape() block.
```

The tape is automatically erased immediately after you call its gradient() method, so
you will get an exception if you try to call gradient() twice:

**390 | Chapter 12: Custom Models and Training with TensorFlow**

```
with tf.GradientTape() as tape:
z = f(w1, w2)
```

```
dz_dw1 = tape.gradient(z, w1) # => tensor 36.0
dz_dw2 = tape.gradient(z, w2) # RuntimeError!
```

If you need to call gradient() more than once, you must make the tape persistent,
and delete it when you are done with it to free resources:

```
with tf.GradientTape(persistent=True) as tape:
z = f(w1, w2)
```

```
dz_dw1 = tape.gradient(z, w1) # => tensor 36.0
dz_dw2 = tape.gradient(z, w2) # => tensor 10.0, works fine now!
del tape
```

By default, the tape will only track operations involving variables, so if you try to
compute the gradient of z with regards to anything else than a variable, the result will
be None:

```
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
z = f(c1, c2)
```

```
gradients = tape.gradient(z, [c1, c2]) # returns [None, None]
```

However, you can force the tape to watch any tensors you like, to record every opera‐
tion that involves them. You can then compute gradients with regards to these ten‐
sors, as if they were variables:

```
with tf.GradientTape() as tape:
tape.watch(c1)
tape.watch(c2)
z = f(c1, c2)
```

```
gradients = tape.gradient(z, [c1, c2]) # returns [tensor 36., tensor 10.]
```

This can be useful in some cases, for example if you want to implement a regulariza‐
tion loss that penalizes activations that vary a lot when the inputs vary little: the loss
will be based on the gradient of the activations with regards to the inputs. Since the
inputs are not variables, you would need to tell the tape to watch them.

If you compute the gradient of a list of tensors (e.g., [z1, z2, z3]) with regards to
some variables (e.g., [w1, w2]), TensorFlow actually efficiently computes the sum of
the gradients of these tensors (i.e., gradient(z1, [w1, w2]), plus gradient(z2,
[w1, w2]), plus gradient(z3, [w1, w2])). Due to the way reverse-mode autodiff
works, it is not possible to compute the individual gradients (z1, z2 and z3) without
actually calling gradient() multiple times (once for z1, once for z2 and once for z3),
which requires making the tape persistent (and deleting it afterwards).

```
Customizing Models and Training Algorithms | 391
```

Moreover, it is actually possible to compute second order partial derivatives (the Hes‐
sians, i.e., the partial derivatives of the partial derivatives)! To do this, we need to
record the operations that are performed when computing the first-order partial
derivatives (the Jacobians): this requires a second tape. Here is how it works:

```
with tf.GradientTape(persistent=True) as hessian_tape:
with tf.GradientTape() as jacobian_tape:
z = f(w1, w2)
jacobians = jacobian_tape.gradient(z, [w1, w2])
hessians = [hessian_tape.gradient(jacobian, [w1, w2])
for jacobian in jacobians]
del hessian_tape
```

The inner tape is used to compute the Jacobians, as we did earlier. The outer tape is
used to compute the partial derivatives of each Jacobian. Since we need to call gradi
ent() once for each Jacobian (or else we would get the sum of the partial derivatives
over all the Jabobians, as explained earlier), we need the outer tape to be persistent, so
we delete it at the end. The Jacobians are obviously the same as earlier (36 and 5), but
now we also have the Hessians:

```
>>> hessians # dz_dw1_dw1, dz_dw1_dw2, dz_dw2_dw1, dz_dw2_dw2
[[<tf.Tensor: id=830578, shape=(), dtype=float32, numpy=6.0>,
<tf.Tensor: id=830595, shape=(), dtype=float32, numpy=2.0>],
[<tf.Tensor: id=830600, shape=(), dtype=float32, numpy=2.0>, None]]
```

Let’s verify these Hessians. The first two are the partial derivatives of 6 _ w1 + 2 _ w2
(which is, as we saw earlier, the partial derivative of f with regards to w1), with
regards to w1 and w2. The result is correct: 6 for w1 and 2 for w2. The next two are the
partial derivatives of 2 _ w1 (the partial derivative of f with regards to w2), with
regards to w1 and w2, which are 2 for w1 and 0 for w2. Note that TensorFlow returns
None instead of 0 since w2 does not appear at all in 2 _ w1. TensorFlow also returns
None when you use an operation whose gradients are not defined (e.g., tf.argmax()).

In some rare cases you may want to stop gradients from backpropagating through
some part of your neural network. To do this, you must use the tf.stop_gradient()
function: it just returns its inputs during the forward pass (like tf.identity()), but
it does not let gradients through during backpropagation (it acts like a constant). For
example:

```
def f(w1, w2):
return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)
```

```
with tf.GradientTape() as tape:
z = f(w1, w2) # same result as without stop_gradient()
```

```
gradients = tape.gradient(z, [w1, w2]) # => returns [tensor 30., None]
```

**392 | Chapter 12: Custom Models and Training with TensorFlow**

Finally, you may occasionally run into some numerical issues when computing gradi‐
ents. For example, if you compute the gradients of the my_softplus() function for
large inputs, the result will be NaN:

```
>>> x = tf.Variable([100.])
>>> with tf.GradientTape() as tape:
... z = my_softplus(x)
```

```
>>> tape.gradient(z, [x])
<tf.Tensor: [...] numpy=array([nan], dtype=float32)>
```

This is because computing the gradients of this function using autodiff leads to some
numerical difficulties: due to floating point precision errors, autodiff ends up com‐
puting infinity divided by infinity (which returns NaN). Fortunately, we can analyti‐
cally find that the derivative of the softplus function is just 1 / (1 + 1 / exp(x)), which
is numerically stable. Next, we can tell TensorFlow to use this stable function when
computing the gradients of the my_softplus() function, by decorating it with
@tf.custom_gradient, and making it return both its normal output and the function
that computes the derivatives (note that it will receive as input the gradients that were
backpropagated so far, down to the softplus function, and according to the chain rule
we should multiply them with this function’s gradients):

```
@tf.custom_gradient
def my_better_softplus(z):
exp = tf.exp(z)
def my_softplus_gradients(grad):
return grad / (1 + 1 / exp)
return tf.math.log(exp + 1), my_softplus_gradients
```

Now when we compute the gradients of the my_better_softplus() function, we get
the proper result, even for large input values (however, the main output still explodes
because of the exponential: one workaround is to use tf.where() to just return the
inputs when they are large).

Congratulations! You can now compute the gradients of any function (provided it is
differentiable at the point where you compute it), you can even compute Hessians,
block backpropagation when needed and even write your own gradient functions!
This is probably more flexibility than you will ever need, even if you build your own
custom training loops, as we will see now.

#### Custom Training Loops

In some rare cases, the fit() method may not be flexible enough for what you need
to do. For example, the Wide and Deep paper we discussed in Chapter 10 actually
uses two different optimizers: one for the wide path and the other for the deep path.
Since the fit() method only uses one optimizer (the one that we specify when

```
Customizing Models and Training Algorithms | 393
```

compiling the model), implementing this paper requires writing your own custom
loop.

You may also like to write your own custom training loops simply to feel more confi‐
dent that it does precisely what you intent it to do (perhaps you are unsure about
some details of the fit() method). It can sometimes feel safer to make everything
explicit. However, remember that writing a custom training loop will make your code
longer, more error prone and harder to maintain.

```
Unless you really need the extra flexibility, you should prefer using
the fit() method rather than implementing your own training
loop, especially if you work in a team.
```

First, let’s build a simple model. No need to compile it, since we will handle the train‐
ing loop manually:

```
l2_reg = keras.regularizers.l2(0.05)
model = keras.models.Sequential([
keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
kernel_regularizer=l2_reg),
keras.layers.Dense(1, kernel_regularizer=l2_reg)
])
```

Next, let’s create a tiny function that will randomly sample a batch of instances from
the training set (in Chapter 13 we will discuss the Data API, which offers a much bet‐
ter alternative):

```
def random_batch(X, y, batch_size=32):
idx = np.random.randint(len(X), size=batch_size)
return X[idx], y[idx]
```

Let’s also define a function that will display the training status, including the number
of steps, the total number of steps, the mean loss since the start of the epoch (i.e., we
will use the Mean metric to compute it), and other metrics:

```
def print_status_bar(iteration, total, loss, metrics=None):
metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
for m in [loss] + (metrics or [])])
end = "" if iteration < total else " \n "
print (" \r {}/{} - ".format(iteration, total) + metrics,
end=end)
```

This code is self-explanatory, unless you are unfamiliar with Python string format‐
ting: {:.4f} will format a float with 4 digits after the decimal point. Moreover, using
\r (carriage return) along with end="" ensures that the status bar always gets printed
on the same line. In the notebook, the print_status_bar() function also includes a
progress bar, but you could use the handy tqdm library instead.

**394 | Chapter 12: Custom Models and Training with TensorFlow**

With that, let’s get down to business! First, we need to define some hyperparameters,
choose the optimizer, the loss function and the metrics (just the MAE in this exam‐
ple):

```
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]
```

And now we are ready to build the custom loop!

```
for epoch in range(1, n_epochs + 1):
print ("Epoch {}/{}".format(epoch, n_epochs))
for step in range(1, n_steps + 1):
X_batch, y_batch = random_batch(X_train_scaled, y_train)
with tf.GradientTape() as tape:
y_pred = model(X_batch, training=True)
main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
loss = tf.add_n([main_loss] + model.losses)
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
mean_loss(loss)
for metric in metrics:
metric(y_batch, y_pred)
print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
for metric in [mean_loss] + metrics:
metric.reset_states()
```

There’s a lot going on in this code, so let’s walk through it:

- We create two nested loops: one for the epochs, the other for the batches within
  an epoch.
- Then we sample a random batch from the training set.
- Inside the tf.GradientTape() block, we make a prediction for one batch (using
  the model as a function), and we compute the loss: it is equal to the main loss
  plus the other losses (in this model, there is one regularization loss per layer).
  Since the mean_squared_error() function returns one loss per instance, we
  compute the mean over the batch using tf.reduce_mean() (if you wanted to
  apply different weights to each instance, this is where you would do it). The regu‐
  larization losses are already reduced to a single scalar each, so we just need to
  sum them (using tf.add_n(), which sums multiple tensors of the same shape
  and data type).

```
Customizing Models and Training Algorithms | 395
```

11 The truth is we did not process every single instance in the training set because we sampled instances ran‐
domly, so some were processed more than once while others were not processed at all. In practice that’s fine.
Moreover, if the training set size is not a multiple of the batch size, we will miss a few instances.
12 Alternatively, check out K.learning_phase(), K.set_learning_phase() and K.learning_phase_scope().
13 With the exception of optimizers, as very few people ever customize these: see the notebook for an example.

- Next, we ask the tape to compute the gradient of the loss with regards to each
  trainable variable (not all variables!), and we apply them to the optimizer to per‐
  form a Gradient Descent step.
- Next we update the mean loss and the metrics (over the current epoch), and we
  display the status bar.
- At the end of each epoch, we display the status bar again to make it look com‐
  plete^11 and to print a line feed, and we reset the states of the mean loss and the
  metrics.

```
If you set the optimizer’s clipnorm or clipvalue hyperparameters, it will take care of
this for you. If you want to apply any other transformation to the gradients, simply do
so before calling the apply_gradients() method.
If you add weight constraints to your model (e.g., by setting kernel_constraint or
bias_constraint when creating a layer), you should update the training loop to
apply these constraints just after apply_gradients():
for variable in model.variables:
if variable.constraint is not None:
variable.assign(variable.constraint(variable))
Most importantly, this training loop does not handle layers that behave differently
during training and testing (e.g., BatchNormalization or Dropout). To handle these,
you need to call the model with training=True and make sure it propagates this to
every layer that needs it.^12
As you can see, there are quite a lot of things you need to get right, it is easy to make a
mistake. But on the bright side, you get full control, so it’s your call.
Now that you know how to customize any part of your models^13 and training algo‐
rithms, let’s see how you can use TensorFlow’s automatic graph generation feature: it
can speed up your custom code considerably, and it will also make it portable to any
platform supported by TensorFlow (see ???).
```

### TensorFlow Functions and Graphs

```
In TensorFlow 1, graphs were unavoidable (as were the complexities that came with
them): they were a central part of TensorFlow’s API. In TensorFlow 2, they are still
```

```
396 | Chapter 12: Custom Models and Training with TensorFlow
```

there, but not as central, and much (much!) simpler to use. To demonstrate this, let’s
start with a trivial function that just computes the cube of its input:

```
def cube(x):
return x ** 3
```

We can obviously call this function with a Python value, such as an int or a float, or
we can call it with a tensor:

```
>>> cube(2)
8
>>> cube(tf.constant(2.0))
<tf.Tensor: id=18634148, shape=(), dtype=float32, numpy=8.0>
```

Now, let’s use tf.function() to convert this Python function to a TensorFlow Func‐
tion:

```
>>> tf_cube = tf.function(cube)
>>> tf_cube
<tensorflow.python.eager.def_function.Function at 0x1546fc080>
```

This TF Function can then be used exactly like the original Python function, and it
will return the same result (but as tensors):

```
>>> tf_cube(2)
<tf.Tensor: id=18634201, shape=(), dtype=int32, numpy=8>
>>> tf_cube(tf.constant(2.0))
<tf.Tensor: id=18634211, shape=(), dtype=float32, numpy=8.0>
```

Under the hood, tf.function() analyzed the computations performed by the cube()
function and generated an equivalent computation graph! As you can see, it was
rather painless (we will see how this works shortly). Alternatively, we could have used
tf.function as a decorator; this is actually more common:

```
@tf.function
def tf_cube(x):
return x ** 3
```

The original Python function is still available via the TF Function’s python_function
attribute, in case you ever need it:

```
>>> tf_cube.python_function(2)
8
```

TensorFlow optimizes the computation graph, pruning unused nodes, simplifying
expressions (e.g., 1 + 2 would get replaced with 3), and more. Once the optimized
graph is ready, the TF Function efficiently executes the operations in the graph, in the
appropriate order (and in parallel when it can). As a result, a TF Function will usually
run much faster than the original Python function, especially if it performs complex

```
TensorFlow Functions and Graphs | 397
```

14 However, in this trivial example, the computation graph is so small that there is nothing at all to optimize, so
tf_cube() actually runs much slower than cube().

```
computations.^14 Most of the time you will not really need to know more than that:
when you want to boost a Python function, just transform it into a TF Function.
That’s all!
Moreover, when you write a custom loss function, a custom metric, a custom layer or
any other custom function, and you use it in a Keras model (as we did throughout
this chapter), Keras automatically converts your function into a TF Function, no need
to use tf.function(). So most of the time, all this magic is 100% transparent.
```

```
You can tell Keras not to convert your Python functions to TF
Functions by setting dynamic=True when creating a custom layer
or a custom model. Alternatively, you can set run_eagerly=True
when calling the model’s compile() method.
```

```
TF Function generates a new graph for every unique set of input shapes and data
types, and it caches it for subsequent calls. For example, if you call tf_cube(tf.con
stant(10)), a graph will be generated for int32 tensors of shape []. Then if you call
tf_cube(tf.constant(20)), the same graph will be reused. But if you then call
tf_cube(tf.constant([10, 20])), a new graph will be generated for int32 tensors
of shape [2]. This is how TF Functions handle polymorphism (i.e., varying argument
types and shapes). However, this is only true for tensor arguments: if you pass numer‐
ical Python values to a TF Function, a new graph will be generated for every distinct
value: for example, calling tf_cube(10) and tf_cube(20) will generate two graphs.
```

```
If you call a TF Function many times with different numerical
Python values, then many graphs will be generated, slowing down
your program and using up a lot of RAM. Python values should be
reserved for arguments that will have few unique values, such as
hyperparameters like the number of neurons per layer. This allows
TensorFlow to better optimize each variant of your model.
```

#### Autograph and Tracing

```
So how does TensorFlow generate graphs? Well, first it starts by analyzing the Python
function’s source code to capture all the control flow statements, such as for loops
and while loops, if statements, as well as break, continue and return statements.
This first step is called autograph. The reason TensorFlow has to analyze the source
code is that Python does not provide any other way to capture control flow state‐
ments: it offers magic methods like __add__() or __mul__() to capture operators like
```

```
398 | Chapter 12: Custom Models and Training with TensorFlow
```

- and \*, but there are no **while**() or **if**() magic methods. After analyzing
  the function’s code, autograph outputs an upgraded version of that function in which
  all the control flow statements are replaced by the appropriate TensorFlow opera‐
  tions, such as tf.while_loop() for loops and tf.cond() for if statements. For
  example, in Figure 12-4, autograph analyzes the source code of the sum_squares()
  Python function, and it generates the tf\_\_sum_squares() function. In this function,
  the for loop is replaced by the definition of the loop_body() function (containing
  the body of the original for loop), followed by a call to the for_stmt() function. This
  call will build the appropriate tf.while_loop() operation in the computation graph.

Figure 12-4. How TensorFlow generates graphs using autograph and tracing

Next, TensorFlow calls this “upgraded” function, but instead of passing the actual
argument, it passes a symbolic tensor, meaning a tensor without any actual value, only
a name, a data type, and a shape. For example, if you call sum_squares(tf.con
stant(10)), then the tf**sum_squares() function will actually be called with a sym‐
bolic tensor of type int32 and shape []. The function will run in graph mode, meaning
that each TensorFlow operation will just add a node in the graph to represent itself
and its output tensor(s) (as opposed to the regular mode, called eager execution, or
eager mode). In graph mode, TF operations do not perform any actual computations.
This should feel familiar if you know TensorFlow 1, as graph mode was the default
mode. In Figure 12-4, you can see the tf**sum_squares() function being called with
a symbolic tensor as argument (in this case, an int32 tensor of shape []), and the final
graph generated during tracing. The ellipses represent operations, and the arrows
represent tensors (both the generated function and the graph are simplified).

```
TensorFlow Functions and Graphs | 399
```

```
To view the generated function’s source code, you can call tf.auto
graph.to_code(sum_squares.python_function). The code is not
meant to be pretty, but it can sometimes help for debugging.
```

#### TF Function Rules

Most of the time, converting a Python function that performs TensorFlow operations
into a TF Function is trivial: just decorate it with @tf.function or let Keras take care
of it for you. However, there are a few rules to respect:

- If you call any external library, including NumPy or even the standard library,
  this call will run only during tracing, it will not be part of the graph. Indeed, a
  TensorFlow graph can only include TensorFlow constructs (tensors, operations,
  variables, datasets, and so on). So make sure you use tf.reduce_sum() instead of
  np.sum(), and tf.sort() instead of the built-in sorted() function, and so on
  (unless you really want the code to run only during tracing).
  —For example, if you define a TF function f(x) that just returns np.ran
  dom.rand(), a random number will only be generated when the function is
  traced, so f(tf.constant(2.)) and f(tf.constant(3.)) will return the
  same random number, but f(tf.constant([2., 3.])) will return a different
  one. If you replace np.random.rand() with tf.random.uniform([]), then a
  new random number will be generated upon every call, since the operation
  will be part of the graph.
  —If your non-TensorFlow code has side-effects (such as logging something or
  updating a Python counter), then you should not expect that side-effect to
  occur every time you call the TF Function, as it will only occur when the func‐
  tion is traced.
  —You can wrap arbitrary Python code in a tf.py_function() operation, but
  this will hinder performance, as TensorFlow will not be able to do any graph
  optimization on this code, and it will also reduce portability, as the graph will
  only run on platforms where Python is available (and the right libraries
  installed).
- You can call other Python functions or TF Functions, but they should follow the
  same rules, as TensorFlow will also capture their operations in the computation
  graph. Note that these other functions do not need to be decorated with
  @tf.function.
- If the function creates a TensorFlow variable (or any other stateful TensorFlow
  object, such as a dataset or a queue), it must do so upon the very first call, and
  only then, or else you will get an exception. It is usually preferable to create vari‐
  ables outside of the TF Function (e.g., in the build() method of a custom layer).

**400 | Chapter 12: Custom Models and Training with TensorFlow**

- The source code of your Python function should be available to TensorFlow. If
  the source code is unavailable (for example, if you define your function in the
  Python shell, which does not give access to the source code, or if you deploy only
  the compiled Python files \*.pyc to production), then the graph generation pro‐
  cess will fail or have limited functionality.
- TensorFlow will only capture for loops that iterate over a tensor or a Dataset. So
  make sure you use for i in tf.range(10) rather than for i in range(10), or
  else the loop will not be captured in the graph. Instead, it will run during tracing.
  This may be what you want, if the for loop is meant to build the graph, for exam‐
  ple to create each layer in a neural network.
- And as always, for performance reasons, you should prefer a vectorized imple‐
  mentation whenever you can, rather than using loops.

It’s time to sum up! In this chapter we started with a brief overview of TensorFlow,
then we looked at TensorFlow’s low-level API, including tensors, operations, variables
and special data structures. We then used these tools to customize almost every com‐
ponent in tf.keras. Finally, we looked at how TF Functions can boost performance,
how graphs are generated using autograph and tracing, and what rules to follow when
you write TF Functions (if you would like to open the black box a bit further, for
example to explore the generated graphs, you will find further technical details
in ???).

In the next chapter, we will look at how to efficiently load and preprocess data with
TensorFlow.

```
TensorFlow Functions and Graphs | 401
```

**CHAPTER 13**

**Loading and Preprocessing Data with**

**TensorFlow**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 13 in the final
release of the book.
```

So far we have used only datasets that fit in memory, but Deep Learning systems are
often trained on very large datasets that will not fit in RAM. Ingesting a large dataset
and preprocessing it efficiently can be tricky to implement with other Deep Learning
libraries, but TensorFlow makes it easy thanks to the Data API: you just create a data‐
set object, tell it where to get the data, then transform it in any way you want, and
TensorFlow takes care of all the implementation details, such as multithreading,
queuing, batching, prefetching, and so on.

Off the shelf, the Data API can read from text files (such as CSV files), binary files
with fixed-size records, and binary files that use TensorFlow’s TFRecord format,
which supports records of varying sizes. TFRecord is a flexible and efficient binary
format based on Protocol Buffers (an open source binary format). The Data API also
has support for reading from SQL databases. Moreover, many Open Source exten‐
sions are available to read from all sorts of data sources, such as Google’s BigQuery
service.

However, reading huge datasets efficiently is not the only difficulty: the data also
needs to be preprocessed. Indeed, it is not always composed strictly of convenient
numerical fields: sometimes there will be text features, categorical features, and so on.
To handle this, TensorFlow provides the Features API: it lets you easily convert these
features to numerical features that can be consumed by your neural network. For

```
403
```

example, categorical features with a large number of categories (such as cities, or
words) can be encoded using embeddings (as we will see, an embedding is a trainable
dense vector that represents a category).

```
Both the Data API and the Features API work seamlessly with
tf.keras.
```

In this chapter, we will cover the Data API, the TFRecord format and the Features
API in detail. We will also take a quick look at a few related projects from Tensor‐
Flow’s ecosystem:

- TF Transform (tf.Transform) makes it possible to write a single preprocessing
  function that can be run both in batch mode on your full training set, before
  training (to speed it up), and then exported to a TF Function and incorporated
  into your trained model, so that once it is deployed in production, it can take
  care of preprocessing new instances on the fly.
- TF Datasets (TFDS) provides a convenient function to download many common
  datasets of all kinds, including large ones like ImageNet, and it provides conve‐
  nient dataset objects to manipulate them using the Data API.

So let’s get started!

### The Data API

The whole Data API revolves around the concept of a dataset: as you might suspect,
this represents a sequence of data items. Usually you will use datasets that gradually
read data from disk, but for simplicity let’s just create a dataset entirely in RAM using
tf.data.Dataset.from_tensor_slices():

```
>>> X = tf.range(10) # any data tensor
>>> dataset = tf.data.Dataset.from_tensor_slices(X)
>>> dataset
<TensorSliceDataset shapes: (), types: tf.int32>
```

The from_tensor_slices() function takes a tensor and creates a tf.data.Dataset
whose elements are all the slices of X (along the first dimension), so this dataset con‐
tains 10 items: tensors 0, 1, 2, ..., 9. In this case we would have obtained the same
dataset if we had used tf.data.Dataset.range(10).

You can simply iterate over a dataset’s items like this:

```
>>> for item in dataset:
... print (item)
```

**404 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
[...]
tf.Tensor(9, shape=(), dtype=int32)
```

#### Chaining Transformations

Once you have a dataset, you can apply all sorts of transformations to it by calling its
transformation methods. Each method returns a new dataset, so you can chain trans‐
formations like this (this chain is illustrated in Figure 13-1):

```
>>> dataset = dataset.repeat(3).batch(7)
>>> for item in dataset:
... print (item)
...
tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
tf.Tensor([8 9], shape=(2,), dtype=int32)
```

Figure 13-1. Chaining Dataset Transformations

In this example, we first call the repeat() method on the original dataset, and it
returns a new dataset that will repeat the items of the original dataset 3 times. Of
course, this will not copy the whole data in memory 3 times! In fact, if you call this
method with no arguments, the new dataset will repeat the source dataset forever.
Then we call the batch() method on this new dataset, and again this creates a new
dataset. This one will group the items of the previous dataset in batches of 7 items.
Finally, we iterate over the items of this final dataset. As you can see, the batch()
method had to output a final batch of size 2 instead of 7, but you can call it with
drop_remainder=True if you want it to drop this final batch so that all batches have
the exact same size.

```
The Data API | 405
```

```
The dataset methods do not modify datasets, they create new ones,
so make sure to keep a reference to these new datasets (e.g., data
set = ...), or else nothing will happen.
```

You can also apply any transformation you want to the items by calling the map()
method. For example, this creates a new dataset with all items doubled:

```
>>> dataset = dataset.map( lambda x: x * 2) # Items: [0,2,4,6,8,10,12]
```

This function is the one you will call to apply any preprocessing you want to your
data. Sometimes, this will include computations that can be quite intensive, such as
reshaping or rotating an image, so you will usually want to spawn multiple threads to
speed things up: it’s as simple as setting the num_parallel_calls argument.

While the map() applies a transformation to each item, the apply() method applies a
transformation to the dataset as a whole. For example, the following code “unbatches”
the dataset, by applying the unbatch() function to the dataset (this function is cur‐
rently experimental, but it will most likely move to the core API in a future release).
Each item in the new dataset will be a single integer tensor instead of a batch of 7
integers:

```
>>> dataset = dataset.apply(tf.data.experimental.unbatch()) # Items: 0,2,4,...
```

It is also possible to simply filter the dataset using the filter() method:

```
>>> dataset = dataset.filter( lambda x: x < 10) # Items: 0 2 4 6 8 0 2 4 6...
```

You will often want to look at just a few items from a dataset. You can use the take()
method for that:

```
>>> for item in dataset.take(3):
... print (item)
```

```
tf.Tensor(0, shape=(), dtype=int64)
tf.Tensor(2, shape=(), dtype=int64)
tf.Tensor(4, shape=(), dtype=int64)
```

#### Shuffling the Data

As you know, Gradient Descent works best when the instances in the training set are
independent and identically distributed (see Chapter 4). A simple way to ensure this
is to shuffle the instances. For this, you can just use the shuffle() method. It will
create a new dataset that will start by filling up a buffer with the first items of the
source dataset, then whenever it is asked for an item, it will pull one out randomly
from the buffer, and replace it with a fresh one from the source dataset, until it has
iterated entirely through the source dataset. At this point it continues to pull out
items randomly from the buffer until it is empty. You must specify the buffer size, and

**406 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
1 Imagine a sorted deck of cards on your left: suppose you just take the top 3 cards and shuffle them, then pick
one randomly and put it to your right, keeping the other 2 in your hands. Take another card on your left,
shuffle the 3 cards in your hands and pick one of them randomly, and put it on your right. When you are
done going through all the cards like this, you will have a deck of cards on your right: do you think it will be
perfectly shuffled?
```

it is important to make it large enough or else shuffling will not be very efficient.^1
However, obviously do not exceed the amount of RAM you have, and even if you
have plenty of it, there’s no need to go well beyond the dataset’s size. You can provide
a random seed if you want the same random order every time you run your program.

```
>>> dataset = tf.data.Dataset.range(10).repeat(3) # 0 to 9, three times
>>> dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
>>> for item in dataset:
... print (item)
...
tf.Tensor([0 2 3 6 7 9 4], shape=(7,), dtype=int64)
tf.Tensor([5 0 1 1 8 6 5], shape=(7,), dtype=int64)
tf.Tensor([4 8 7 1 2 3 0], shape=(7,), dtype=int64)
tf.Tensor([5 4 2 7 8 9 9], shape=(7,), dtype=int64)
tf.Tensor([3 6], shape=(2,), dtype=int64)
```

```
If you call repeat() on a shuffled dataset, by default it will generate
a new order at every iteration. This is generally a good idea, but if
you prefer to reuse the same order at each iteration (e.g., for tests
or debugging), you can set reshuffle_each_iteration=False.
```

For a large dataset that does not fit in memory, this simple shuffling-buffer approach
may not be sufficient, since the buffer will be small compared to the dataset. One sol‐
ution is to shuffle the source data itself (for example, on Linux you can shuffle text
files using the shuf command). This will definitely improve shuffling a lot! However,
even if the source data is shuffled, you will usually want to shuffle it some more, or
else the same order will be repeated at each epoch, and the model may end up being
biased (e.g., due to some spurious patterns present by chance in the source data’s
order). To shuffle the instances some more, a common approach is to split the source
data into multiple files, then read them in a random order during training. However,
instances located in the same file will still end up close to each other. To avoid this
you can pick multiple files randomly, and read them simultaneously, interleaving
their lines. Then on top of that you can add a shuffling buffer using the shuffle()
method. If all this sounds like a lot of work, don’t worry: the Data API actually makes
all this possible in just a few lines of code. Let’s see how to do this.

```
The Data API | 407
```

**Interleaving Lines From Multiple Files**

First, let’s suppose that you loaded the California housing dataset, you shuffled it
(unless it was already shuffled), you split it into a training set, a validation set and a
test set, then you split each set into many CSV files that each look like this (each row
contains 8 input features plus the target median house value):

```
MedInc,HouseAge,AveRooms,AveBedrms,Popul,AveOccup,Lat,Long,MedianHouseValue
3.5214,15.0,3.0499,1.1065,1447.0,1.6059,37.63,-122.43,1.442
5.3275,5.0,6.4900,0.9910,3464.0,3.4433,33.69,-117.39,1.687
3.1,29.0,7.5423,1.5915,1328.0,2.2508,38.44,-122.98,1.621
[...]
```

Let’s also suppose train_filepaths contains the list of file paths (and you also have
valid_filepaths and test_filepaths):

```
>>> train_filepaths
['datasets/housing/my_train_00.csv', 'datasets/housing/my_train_01.csv',...]
```

Now let’s create a dataset containing only these file paths:

```
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
```

By default, the list_files() function returns a dataset that shuffles the file paths. In
general this is a good thing, but you can set shuffle=False if you do not want that,
for some reason.

Next, we can call the interleave() method to read from 5 files at a time and inter‐
leave their lines (skipping the first line of each file, which is the header row, using the
skip() method):

```
n_readers = 5
dataset = filepath_dataset.interleave(
lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
cycle_length=n_readers)
```

The interleave() method will create a dataset that will pull 5 file paths from the
filepath_dataset, and for each one it will call the function we gave it (a lambda in
this example) to create a new dataset, in this case a TextLineDataset. It will then
cycle through these 5 datasets, reading one line at a time from each until all datasets
are out of items. Then it will get the next 5 file paths from the filepath_dataset, and
interleave them the same way, and so on until it runs out of file paths.

```
For interleaving to work best, it is preferable to have files of identi‐
cal length, or else the end of the longest files will not be interleaved.
```

**408 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

By default, interleave() does not use parallelism, it just reads one line at a time
from each file, sequentially. However, if you want it to actually read files in parallel,
you can set the num_parallel_calls argument to the number of threads you want.
You can even set it to tf.data.experimental.AUTOTUNE to make TensorFlow choose
the right number of threads dynamically based on the available CPU (however, this is
an experimental feature for now). Let’s look at what the dataset contains now:

```
>>> for line in dataset.take(5):
... print (line.numpy())
```

```
b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782'
b'4.1812,52.0,5.7013,0.9965,692.0,2.4027,33.73,-118.31,3.215'
b'3.6875,44.0,4.5244,0.9930,457.0,3.1958,34.04,-118.15,1.625'
b'3.3456,37.0,4.5140,0.9084,458.0,3.2253,36.67,-121.7,2.526'
b'3.5214,15.0,3.0499,1.1065,1447.0,1.6059,37.63,-122.43,1.442'
```

These are the first rows (ignoring the header row) of 5 CSV files, chosen randomly.
Looks good! But as you can see, these are just byte strings, we need to parse them,
and also scale the data.

#### Preprocessing the Data

Let’s implement a small function that will perform this preprocessing:

```
X_mean, X_std = [...] # mean and scale of each feature in the training set
n_inputs = 8
```

```
def preprocess(line):
defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
fields = tf.io.decode_csv(line, record_defaults=defs)
x = tf.stack(fields[:-1])
y = tf.stack(fields[-1:])
return (x - X_mean) / X_std, y
```

Let’s walk through this code:

- First, we assume that you have precomputed the mean and standard deviation of
  each feature in the training set. X_mean and X_std are just 1D tensors (or NumPy
  arrays) containing 8 floats, one per input feature.
- The preprocess() function takes one CSV line, and starts by parsing it. For this,
  it uses the tf.io.decode_csv() function, which takes two arguments: the first is
  the line to parse, and the second is an array containing the default value for each
  column in the CSV file. This tells TensorFlow not only the default value for each
  column, but also the number of columns and the type of each column. In this
  example, we tell it that all feature columns are floats and missing values should
  default to 0, but we provide an empty array of type tf.float32 as the default
  value for the last column (the target): this tells TensorFlow that this column con‐

```
The Data API | 409
```

```
tains floats, but that there is no default value, so it will raise an exception if it
encounters a missing value.
```

- The decode_csv() function returns a list of scalar tensors (one per column) but
  we need to return 1D tensor arrays. So we call tf.stack() on all tensors except
  for the last one (the target): this will stack these tensors into a 1D array. We then
  do the same for the target value (this makes it a 1D tensor array with a single
  value, rather than a scalar tensor).
- Finally, we scale the input features by subtracting the feature means and then
  dividing by the feature standard deviations, and we return a tuple containing the
  scaled features and the target.

Let’s test this preprocessing function:

```
>>> preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')
(<tf.Tensor: id=6227, shape=(8,), dtype=float32, numpy=
array([ 0.16579159, 1.216324 , -0.05204564, -0.39215982, -0.5277444 ,
-0.2633488 , 0.8543046 , -1.3072058 ], dtype=float32)>,
<tf.Tensor: [...], numpy=array([2.782], dtype=float32)>)
```

We can now apply this preprocessing function to the dataset.

#### Putting Everything Together

To make the code reusable, let’s put together everything we have discussed so far into
a small helper function: it will create and return a dataset that will efficiently load Cal‐
ifornia housing data from multiple CSV files, then shuffle it, preprocess it and batch it
(see Figure 13-2):

```
def csv_reader_dataset(filepaths, repeat=None, n_readers=5,
n_read_threads=None, shuffle_buffer_size=10000,
n_parse_threads=5, batch_size=32):
dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
dataset = dataset.interleave(
lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
cycle_length=n_readers, num_parallel_calls=n_read_threads)
dataset = dataset.shuffle(shuffle_buffer_size)
dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
dataset = dataset.batch(batch_size)
return dataset.prefetch(1)
```

**410 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
2 In general, just prefetching one batch is fine, but in some cases you may need to prefetch a few more. Alterna‐
tively, you can let TensorFlow decide automatically by passing tf.data.experimental.AUTOTUNE (this is an
experimental feature for now).
```

Figure 13-2. Loading and Preprocessing Data From Multiple CSV Files

Everything should make sense in this code, except the very last line (prefetch(1)),
which is actually quite important for performance.

#### Prefetching

By calling prefetch(1) at the end, we are creating a dataset that will do its best to
always be one batch ahead^2. In other words, while our training algorithm is working
on one batch, the dataset will already be working in parallel on getting the next batch
ready. This can improve performance dramatically, as is illustrated on Figure 13-3. If
we also ensure that loading and preprocessing are multithreaded (by setting num_par
allel_calls when calling interleave() and map()), we can exploit multiple cores
on the CPU and hopefully make preparing one batch of data shorter than running a
training step on the GPU: this way the GPU will be almost 100% utilized (except for
the data transfer time from the CPU to the GPU), and training will run much faster.

```
The Data API | 411
```

Figure 13-3. Speedup Training Thanks to Prefetching and Multithreading

```
If you plan to purchase a GPU card, its processing power and its
memory size are of course very important (in particular, a large
RAM is crucial for computer vision), but its memory bandwidth is
just as important as the processing power to get good performance:
this is the number of gigabytes of data it can get in or out of its
RAM per second.
```

With that, you can now build efficient input pipelines to load and preprocess data
from multiple text files. We have discussed the most common dataset methods, but
there are a few more you may want to look at: concatenate(), zip(), window(),
reduce(), cache(), shard(), flat_map() and padded_batch(). There are also a cou‐
ple more class methods: from_generator() and from_tensors(), which create a new
dataset from a Python generator or a list of tensors respectively. Please check the API
documentation for more details. Also note that there are experimental features avail‐
able in tf.data.experimental, many of which will most likely make it to the core
API in future releases (e.g., check out the CsvDataset class and the SqlDataset
classes).

**412 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
3 Support for datasets is specific to tf.keras, it will not work on other implementations of the Keras API.
4 The number of steps per epoch is optional if the dataset just goes through the data once, but if you do not
specify it, the progress bar will not be displayed during the first epoch.
5 Note that for now the dataset must be created within the TF Function. This may be fixed by the time you read
these lines (see TensorFlow issue #25414).
```

#### Using the Dataset With tf.keras

Now we can use the csv_reader_dataset() function to create a dataset for the train‐
ing set (ensuring it repeats the data forever), the validation set and the test set:

```
train_set = csv_reader_dataset(train_filepaths, repeat=None)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)
```

And now we can simply build and train a Keras model using these datasets.^3 All we
need to do is to call the fit() method with the datasets instead of X_train and
y_train, and specify the number of steps per epoch for each set:^4

```
model = keras.models.Sequential([...])
model.compile([...])
model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10,
validation_data=valid_set,
validation_steps=len(X_valid) // batch_size)
```

Similarly, we can pass a dataset to the evaluate() and predict() methods (and again
specify the number of steps per epoch):

```
model.evaluate(test_set, steps=len(X_test) // batch_size)
model.predict(new_set, steps=len(X_new) // batch_size)
```

Unlike the other sets, the new_set will usually not contain labels (if it does, Keras will
just ignore them). Note that in all these cases, you can still use NumPy arrays instead
of datasets if you want (but of course they need to have been loaded and preprocessed
first).

If you want to build your own custom training loop (as in Chapter 12), you can just
iterate over the training set, very naturally:

```
for X_batch, y_batch in train_set:
[...] # perform one gradient descent step
```

In fact, it is even possible to create a tf.function (see Chapter 12) that performs the
whole training loop!^5

```
@tf.function
def train(model, optimizer, loss_fn, n_epochs, [...]):
train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, [...])
for X_batch, y_batch in train_set:
with tf.GradientTape() as tape:
```

```
The Data API | 413
```

```
y_pred = model(X_batch)
main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
loss = tf.add_n([main_loss] + model.losses)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Congratulations, you now know how to build powerful input pipelines using the Data
API! However, so far we have used CSV files, which are common, simple and conve‐
nient, but they are not really efficient, and they do not support large or complex data
structures very well, such as images or audio. So let’s use TFRecords instead.

```
If you are happy with CSV files (or whatever other format you are
using), you do not have to use TFRecords. As the saying goes, if it
ain’t broke, don’t fix it! TFRecords are useful when the bottleneck
during training is loading and parsing the data.
```

### The TFRecord Format

The TFRecord format is TensorFlow’s preferred format for storing large amounts of
data and reading it efficiently. It is a very simple binary format that just contains a
sequence of binary records of varying sizes (each record just has a length, a CRC
checksum to check that the length was not corrupted, then the actual data, and finally
a CRC checksum for the data). You can easily create a TFRecord file using the
tf.io.TFRecordWriter class:

```
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
f.write(b"This is the first record")
f.write(b"And this is the second record")
```

And you can then use a tf.data.TFRecordDataset to read one or more TFRecord
files:

```
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
print (item)
```

This will output:

```
tf.Tensor(b'This is the first record', shape=(), dtype=string)
tf.Tensor(b'And this is the second record', shape=(), dtype=string)
```

```
By default, a TFRecordDataset will read files one by one, but you
can make it read multiple files in parallel and interleave their
records by setting num_parallel_reads. Alternatively, you could
obtain the same result by using list_files() and interleave()
as we did earlier to read multiple CSV files.
```

**414 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
6 Since protobuf objects are meant to be serialized and transmitted, they are called messages.
```

#### Compressed TFRecord Files

It can sometimes be useful to compress your TFRecord files, especially if they need to
be loaded via a network connection. You can create a compressed TFRecord file by
setting the options argument:

```
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
[...]
```

When reading a compressed TFRecord file, you need to specify the compression type:

```
dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],
compression_type="GZIP")
```

#### A Brief Introduction to Protocol Buffers

Even though each record can use any binary format you want, TFRecord files usually
contain serialized Protocol Buffers (also called protobufs). This is a portable, extensi‐
ble and efficient binary format developed at Google back in 2001 and Open Sourced
in 2008, and they are now widely used, in particular in gRPC, Google’s remote proce‐
dure call system. Protocol Buffers are defined using a simple language that looks like
this:

```
syntax = "proto3";
message Person {
string name = 1;
int32 id = 2;
repeated string email = 3;
}
```

This definition says we are using the protobuf format version 3, and it specifies that
each Person object^6 may (optionally) have a name of type string, an id of type int32,
and zero or more email fields, each of type string. The numbers 1, 2 and 3 are the
field identifiers: they will be used in each record’s binary representation. Once you
have a definition in a .proto file, you can compile it. This requires protoc, the proto‐
buf compiler, to generate access classes in Python (or some other language). Note that
the protobuf definitions we will use have already been compiled for you, and their
Python classes are part of TensorFlow, so you will not need to use protoc. All you
need to know is how to use protobuf access classes in Python. To illustrate the basics,
let’s look at a simple example that uses the access classes generated for the Person
protobuf (the code is explained in the comments):

```
>>> from person_pb2 import Person # import the generated access class
>>> person = Person(name="Al", id=123, email=["a@b.com"]) # create a Person
>>> print (person) # display the Person
```

```
The TFRecord Format | 415
```

```
7 This chapter contains the bare minimum you need to know about protobufs to use TFRecords. To learn more
about protobufs, please visit https://homl.info/protobuf.
```

```
name: "Al"
id: 123
email: "a@b.com"
>>> person.name # read a field
"Al"
>>> person.name = "Alice" # modify a field
>>> person.email[0] # repeated fields can be accessed like arrays
"a@b.com"
>>> person.email.append("c@d.com") # add an email address
>>> s = person.SerializeToString() # serialize the object to a byte string
>>> s
b'\n\x05Alice\x10{\x1a\x07a@b.com\x1a\x07c@d.com'
>>> person2 = Person() # create a new Person
>>> person2.ParseFromString(s) # parse the byte string (27 bytes long)
27
>>> person == person2 # now they are equal
True
```

In short, we import the Person class generated by protoc, we create an instance and
we play with it, visualizing it, reading and writing some fields, then we serialize it
using the SerializeToString() method. This is the binary data that is ready to be
saved or transmitted over the network. When reading or receiving this binary data,
we can parse it using the ParseFromString() method, and we get a copy of the object
that was serialized.^7

We could save the serialized Person object to a TFRecord file, then we could load and
parse it: everything would work fine. However, SerializeToString() and ParseFrom
String() are not TensorFlow operations (and neither are the other operations in this
code), so they cannot be included in a TensorFlow Function (except by wrapping
them in a tf.py_function() operation, which would make the code slower and less
portable, as we saw in Chapter 12). Fortunately, TensorFlow does include special pro‐
tobuf definitions for which it provides parsing operations.

#### TensorFlow Protobufs

The main protobuf typically used in a TFRecord file is the Example protobuf, which
represents one instance in a dataset. It contains a list of named features, where each
feature can either be a list of byte strings, a list of floats or a list of integers. Here is the
protobuf definition:

```
syntax = "proto3";
message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true ]; }
message Int64List { repeated int64 value = 1 [packed = true ]; }
```

**416 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
8 Why was Example even defined since it contains no more than a Features object? Well, TensorFlow may one
day decide to add more fields to it. As long as the new Example definition still contains the features field,
with the same id, it will be backward compatible. This extensibility is one of the great features of protobufs.
```

```
message Feature {
oneof kind {
BytesList bytes_list = 1;
FloatList float_list = 2;
Int64List int64_list = 3;
}
};
message Features { map<string, Feature> feature = 1; };
message Example { Features features = 1; };
```

The definitions of BytesList, FloatList and Int64List are straightforward enough
([packed = true] is used for repeated numerical fields, for a more efficient encod‐
ing). A Feature either contains a BytesList, a FloatList or an Int64List. A Fea
tures (with an s) contains a dictionary that maps a feature name to the
corresponding feature value. And finally, an Example just contains a Features object.^8
Here is how you could create a tf.train.Example representing the same person as
earlier, and write it to TFRecord file:

```
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example
```

```
person_example = Example(
features=Features(
feature={
"name": Feature(bytes_list=BytesList(value=[b"Alice"])),
"id": Feature(int64_list=Int64List(value=[123])),
"emails": Feature(bytes_list=BytesList(value=[b"a@b.com",
b"c@d.com"]))
}))
```

The code is a bit verbose and repetitive, but it’s rather straightforward (and you could
easily wrap it inside a small helper function). Now that we have an Example protobuf,
we can serialize it by calling its SerializeToString() method, then write the result‐
ing data to a TFRecord file:

```
with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
f.write(person_example.SerializeToString())
```

Normally you would write much more than just one example! Typically, you would
create a conversion script that reads from your current format (say, CSV files), creates
an Example protobuf for each instance, serializes them and saves them to several
TFRecord files, ideally shuffling them in the process. This requires a bit of work, so
once again make sure it is really necessary (perhaps your pipeline works fine with
CSV files).

```
The TFRecord Format | 417
```

Now that we have a nice TFRecord file containing a serialized Example, let’s try to
load it.

#### Loading and Parsing Examples

To load the serialized Example protobufs, we will use a tf.data.TFRecordDataset
once again, and we will parse each Example using tf.io.parse_single_example().
This is a TensorFlow operation so it can be included in a TF Function. It requires at
least two arguments: a string scalar tensor containing the serialized data, and a
description of each feature. The description is a dictionary that maps each feature
name to either a tf.io.FixedLenFeature descriptor indicating the feature’s shape,
type and default value, or a tf.io.VarLenFeature descriptor indicating only the type
(if the length may vary, such as for the "emails" feature). For example:

```
feature_description = {
"name": tf.io.FixedLenFeature([], tf.string, default_value=""),
"id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
"emails": tf.io.VarLenFeature(tf.string),
}
```

```
for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
parsed_example = tf.io.parse_single_example(serialized_example,
feature_description)
```

The fixed length features are parsed as regular tensors, but the variable length fea‐
tures are parsed as sparse tensors. You can convert a sparse tensor to a dense tensor
using tf.sparse.to_dense(), but in this case it is simpler to just access its values:

```
>>> tf.sparse.to_dense(parsed_example["emails"], default_value=b"")
<tf.Tensor: [...] dtype=string, numpy=array([b'a@b.com', b'c@d.com'], [...])>
>>> parsed_example["emails"].values
<tf.Tensor: [...] dtype=string, numpy=array([b'a@b.com', b'c@d.com'], [...])>
```

A BytesList can contain any binary data you want, including any serialized object.
For example, you can use tf.io.encode_jpeg() to encode an image using the JPEG
format, and put this binary data in a BytesList. Later, when your code reads the
TFRecord, it will start by parsing the Example, then you will need to call
tf.io.decode_jpeg() to parse the data and get the original image (or you can use
tf.io.decode_image(), which can decode any BMP, GIF, JPEG or PNG image). You
can also store any tensor you want in a BytesList by serializing the tensor using
tf.io.serialize_tensor(), then putting the resulting byte string in a BytesList
feature. Later, when you parse the TFRecord, you can parse this data using
tf.io.parse_tensor().

Instead of parsing examples one by one using tf.io.parse_single_example(), you
may want to parse them batch by batch using tf.io.parse_example():

**418 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(10)
for serialized_examples in dataset:
parsed_examples = tf.io.parse_example(serialized_examples,
feature_description)
```

As you can see, the Example proto will probably be sufficient for most use cases.
However, it may be a bit cumbersome to use when you are dealing with lists of lists.
For example, suppose you want to classify text documents. Each document may be
represented as a list of sentences, where each sentence is represented as a list of
words. And perhaps each document also has a list of comments, where each com‐
ment is also represented as a list of words. Moreover, there may be some contextual
data as well, such as the document’s author, title and publication date. TensorFlow’s
SequenceExample protobuf is designed for such use cases.

#### Handling Lists of Lists Using the SequenceExample Protobuf

Here is the definition of the SequenceExample protobuf:

```
message FeatureList { repeated Feature feature = 1; };
message FeatureLists { map<string, FeatureList> feature_list = 1; };
message SequenceExample {
Features context = 1;
FeatureLists feature_lists = 2;
};
```

A SequenceExample contains a Features object for the contextual data and a Fea
tureLists object which contains one or more named FeatureList objects (e.g., a
FeatureList named "content" and another named "comments"). Each FeatureList
just contains a list of Feature objects, each of which may be a list of byte strings, a list
of 64-bit integers or a list of floats (in this example, each Feature would represent a
sentence or a comment, perhaps in the form of a list of word identifiers). Building a
SequenceExample, serializing it and parsing it is very similar to building, serializing
and parsing an Example, but you must use tf.io.parse_single_sequence_exam
ple() to parse a single SequenceExample or tf.io.parse_sequence_example() to
parse a batch, and both functions return a tuple containing the context features (as a
dictionary) and the feature lists (also as a dictionary). If the feature lists contain
sequences of varying sizes (as in the example above), you may want to convert them
to ragged tensors using tf.RaggedTensor.from_sparse() (see the notebook for the
full code):

```
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
serialized_sequence_example, context_feature_descriptions,
sequence_feature_descriptions)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
```

Now that you know how to efficiently store, load and parse data, the next step is to
prepare it so that it can be fed to a neural network. This means converting all features

```
The TFRecord Format | 419
```

into numerical features (ideally not too sparse), scaling them, and more. In particular,
if your data contains categorical features or text features, they need to be converted to
numbers. For this, the Features API can help.

### The Features API

Preprocessing your data can be performed in many ways: it can be done ahead of
time when preparing your data files, using any tool you like. Or you can preprocess
your data on the fly when loading it with the Data API (e.g., using the dataset’s map()
method, as we saw earlier). Or you can include a preprocessing layer directly in your
model. Whichever solution you prefer, the Features API can help you: it is a set of
functions available in the tf.feature_column package, which let you define how
each feature (or group of features) in your data should be preprocessed (therefore you
can think of this API as the analog of Scikit-Learn’s ColumnTransformer class). We
will start by looking at the different types of columns available, and then we will look
at how to use them.

Let’s go back to the variant of the California housing dataset that we used in Chap‐
ter 2, since it includes a categorical feature and missing data. Here is a simple numeri‐
cal column named "housing_median_age":

```
housing_median_age = tf.feature_column.numeric_column("housing_median_age")
```

Numeric columns let you specify a normalization function using the normalizer_fn
argument. For example, let’s tweak the "housing_median_age" column to define how
it should be scaled. Note that this requires computing ahead of time the mean and
standard deviation of this feature in the training set:

```
age_mean, age_std = X_mean[1], X_std[1] # The median age is column in 1
housing_median_age = tf.feature_column.numeric_column(
"housing_median_age", normalizer_fn= lambda x: (x - age_mean) / age_std)
```

In some cases, it might improve performance to bucketize some numerical features,
effectively transforming a numerical feature into a categorical feature. For example,
let’s create a bucketized column based on the median_income column, with 5 buckets:
less than 1.5 ($15,000), then 1.5 to 3, 3 to 4.5, 4.5 to 6., and above 6. (notice that when
you specify 4 boundaries, there are actually 5 buckets):

```
median_income = tf.feature_column.numeric_column("median_income")
bucketized_income = tf.feature_column.bucketized_column(
median_income, boundaries=[1.5, 3., 4.5, 6.])
```

If the median_income feature is equal to, say, 3.2, then the bucketized_income feature
will automatically be equal to 2 (i.e., the index of the corresponding income bucket).
Choosing the right boundaries can be somewhat of an art, but one approach is to just
use percentiles of the data (e.g., the 10th percentile, the 20th percentile, and so on). If
a feature is multimodal, meaning it has separate peaks in its distribution, you may

**420 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

want to define a bucket for each mode, placing the boundaries in between the peaks.
Whether you use the percentiles or the modes, you need to analyze the distribution of
your data ahead of time, just like we had to measure the mean and standard deviation
ahead of time to normalize the housing_median_age column.

#### Categorical Features

For categorical features such as ocean_proximity, there are several options. If it is
already represented as a category ID (i.e., an integer from 0 to the max ID), then you
can use the categorical_column_with_identity() function (specifying the max
ID). If not, and you know the list of all possible categories, then you can use categori
cal_column_with_vocabulary_list():

```
ocean_prox_vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
ocean_proximity = tf.feature_column.categorical_column_with_vocabulary_list(
"ocean_proximity", ocean_prox_vocab)
```

If you prefer to have TensorFlow load the vocabulary from a file, you can call catego
rical_column_with_vocabulary_file() instead. As you might expect, these two
functions will simply map each category to its index in the vocabulary (e.g., NEAR
BAY will be mapped to 3), and unknown categories will be mapped to -1.

For categorical columns with a large vocabulary (e.g., for zipcodes, cities, words,
products, users, etc.), it may not be convenient to get the full list of possible cate‐
gories, or perhaps categories may be added or removed so frequently that using cate‐
gory indices would be too unreliable. In this case, you may prefer to use a
categorical_column_with_hash_bucket(). If we had a "city" feature in the dataset,
we could encode it like this:

```
city_hash = tf.feature_column.categorical_column_with_hash_bucket(
"city", hash_bucket_size=1000)
```

This feature will compute a hash for each category (i.e., for each city), modulo the
number of hash buckets (hash_bucket_size). You must set the number of buckets
high enough to avoid getting too many collisions (i.e., different categories ending up
in the same bucket), but the higher you set it, the more RAM will be used (by the
embedding table, as we will see shortly).

#### Crossed Categorical Features

If you suspect that two (or more) categorical features are more meaningful when used
jointly, then you can create a crossed column. For example, suppose people are partic‐
ularly fond of old houses inland and new houses near the ocean, then it might help to

```
The Features API | 421
```

```
9 Since the housing_median_age feature was normalized, the boundaries are for normalized ages.
```

create a bucketized column for the housing_median_age feature^9 , and cross it with
the ocean_proximity column. The crossed column will compute a hash of every age
& ocean proximity combination it comes across, modulo the hash_bucket_size, and
this will give it the cross category ID. You may then choose to use only this crossed
column in your model, or also include the individual columns.

```
bucketized_age = tf.feature_column.bucketized_column(
housing_median_age, boundaries=[-1., -0.5, 0., 0.5, 1.]) # age was scaled
age_and_ocean_proximity = tf.feature_column.crossed_column(
[bucketized_age, ocean_proximity], hash_bucket_size=100)
```

Another common use case for crossed columns is to cross latitude and longitude into
a single categorical feature: you start by bucketizing the latitude and longitude, for
example into 20 buckets each, then you cross these bucketized features into a loca
tion column. This will create a 20×20 grid over California, and each cell in the grid
will correspond to one category:

```
latitude = tf.feature_column.numeric_column("latitude")
longitude = tf.feature_column.numeric_column("longitude")
bucketized_latitude = tf.feature_column.bucketized_column(
latitude, boundaries=list(np.linspace(32., 42., 20 - 1)))
bucketized_longitude = tf.feature_column.bucketized_column(
longitude, boundaries=list(np.linspace(-125., -114., 20 - 1)))
location = tf.feature_column.crossed_column(
[bucketized_latitude, bucketized_longitude], hash_bucket_size=1000)
```

#### Encoding Categorical Features Using One-Hot Vectors

No matter which option you choose to build a categorical feature (categorical col‐
umns, bucketized columns or crossed columns), it must be encoded before you can
feed it to a neural network. There are two options to encode a categorical feature:
one-hot vectors or embeddings. For the first option, simply use the indicator_col
umn() function:

```
ocean_proximity_one_hot = tf.feature_column.indicator_column(ocean_proximity)
```

A one-hot vector encoding has the size of the vocabulary length, which is fine if there
are just a few possible categories, but if the vocabulary is large, you will end up with
too many inputs fed to your neural network: it will have too many weights to learn
and it will probably not perform very well. In particular, this will typically be the case
when you use hash buckets. In this case, you should probably encode them using
embeddings instead.

**422 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
As a rule of thumb (but your mileage may vary!), if the number of
categories is lower than 10, then one-hot encoding is generally the
way to go. If the number of categories is greater than 50 (which is
often the case when you use hash buckets), then embeddings are
usually preferable. In between 10 and 50 categories, you may want
to experiment with both options and see which one works best for
your use case. Also, embeddings typically require more training
data, unless you can reuse pretrained embeddings.
```

#### Encoding Categorical Features Using Embeddings

An embedding is a trainable dense vector that represents a category. By default,
embeddings are initialized randomly, so for example the "NEAR BAY" category could
be represented initially by a random vector such as [0.131, 0.890], while the "NEAR
OCEAN" category may be represented by another random vector such as [0.631,
0.791] (in this example, we are using 2D embeddings, but the number of dimensions
is a hyperparameter you can tweak). Since these embeddings are trainable, they will
gradually improve during training, and as they represent fairly similar categories,
Gradient Descent will certainly end up pushing them closer together, while it will
tend to move them away from the "INLAND" category’s embedding (see Figure 13-4).
Indeed, the better the representation, the easier it will be for the neural network to
make accurate predictions, so training tends to make embeddings useful representa‐
tions of the categories. This is called representation learning (we will see other types of
representation learning in ???).

```
The Features API | 423
```

10 “Distributed Representations of Words and Phrases and their Compositionality”, T. Mikolov et al. (2013).

```
Figure 13-4. Embeddings Will Gradually Improve During Training
```

```
Word Embeddings
Not only will embeddings generally be useful representations for the task at hand, but
quite often these same embeddings can be reused successfully for other tasks as well.
The most common example of this is word embeddings (i.e., embeddings of individual
words): when you are working on a natural language processing task, you are often
better off reusing pretrained word embeddings than training your own. The idea of
using vectors to represent words dates back to the 1960s, and many sophisticated
techniques have been used to generate useful vectors, including using neural net‐
works, but things really took off in 2013, when Tomáš Mikolov and other Google
researchers published a paper^10 describing how to learn word embeddings using deep
neural networks, much faster than previous attempts. This allowed them to learn
embeddings on a very large corpus of text: they trained a deep neural network to pre‐
dict the words near any given word. This allowed them to obtain astounding word
embeddings. For example, synonyms had very close embeddings, and semantically
related words such as France, Spain, Italy, and so on, ended up clustered together. But
it’s not just about proximity: word embeddings were also organized along meaningful
axes in the embedding space. Here is a famous example: if you compute King – Man
+ Woman (adding and subtracting the embedding vectors of these words), then the
result will be very close to the embedding of the word Queen (see Figure 13-5). In
other words, the word embeddings encode the concept of gender! Similarly, you can
compute Madrid – Spain + France, and of course the result is close to Paris, which
seems to show that the notion of capital city was also encoded in the embeddings.
```

```
424 | Chapter 13: Loading and Preprocessing Data with TensorFlow
```

```
Figure 13-5. Word Embeddings
```

Let’s go back to the Features API. Here is how you could encode the ocean_proxim
ity categories as 2D embeddings:

```
ocean_proximity_embed = tf.feature_column.embedding_column(ocean_proximity,
dimension=2)
```

Each of the five ocean_proximity categories will now be represented as a 2D vector.
These vectors are stored in an embedding matrix with one row per category, and one
column per embedding dimension, so in this example it is a 5×2 matrix. When an
embedding column is given a category index as input (say, 3, which corresponds to
the category "NEAR BAY"), it just performs a lookup in the embedding matrix and
returns the corresponding row (say, [0.331, 0.190]). Unfortunately, the embedding
matrix can be quite large, especially when you have a large vocabulary: if this is the
case, the model can only learn good representations for the categories for which it has
sufficient training data. To reduce the size of the embedding matrix, you can of
course try lowering the dimension hyperparameter, but if you reduce this parameter
too much, the representations may not be as good. Another option is to reduce the
vocabulary size (e.g., if you are dealing with text, you can try dropping the rare words
from the vocabulary, and replace them all with a token like "<unknown>" or "<UNK>").
If you are using hash buckets, you can also try reducing the hash_bucket_size (but
not too much, or else you will get collisions).

```
The Features API | 425
```

```
If there are no pretrained embeddings that you can reuse for the
task you are trying to tackle, and if you do not have enough train‐
ing data to learn them, then you can try to learn them on some
auxiliary task for which it is easier to obtain plenty of training data.
After that, you can reuse the trained embeddings for your main
task.
```

#### Using Feature Columns for Parsing

Let’s suppose you have created feature columns for each of your input features, as well
as for the target. What can you do with them? Well, for one you can pass them to the
make_parse_example_spec() function to generate feature descriptions (so you don’t
have to do it manually, as we did earlier):

```
columns = [bucketized_age, ....., median_house_value] # all features + target
feature_descriptions = tf.feature_column.make_parse_example_spec(columns)
```

```
You don’t always have to create a separate feature column for each
and every feature. For example, instead of having 2 numerical fea‐
ture columns, you could choose to have a single 2D column: just
set shape=[2] when calling numerical_column().
```

You can then create a function that parses serialized examples using these feature
descriptions, and separates the target column from the input features:

```
def parse_examples(serialized_examples):
examples = tf.io.parse_example(serialized_examples, feature_descriptions)
targets = examples.pop("median_house_value") # separate the targets
return examples, targets
```

Next, you can create a TFRecordDataset that will read batches of serialized examples
(assuming the TFRecord file contains serialized Example protobufs with the appropri‐
ate features):

```
batch_size = 32
dataset = tf.data.TFRecordDataset(["my_data_with_features.tfrecords"])
dataset = dataset.repeat().shuffle(10000).batch(batch_size).map(parse_examples)
```

#### Using Feature Columns in Your Models

Feature columns can also be used directly in your model, to convert all your input
features into a single dense vector which the neural network can then process. For
this, all you need to do is add a keras.layers.DenseFeatures layer as the first layer
in your model, passing it the list of feature columns (excluding the target column):

```
columns_without_target = columns[:-1]
model = keras.models.Sequential([
keras.layers.DenseFeatures(feature_columns=columns_without_target),
```

**426 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

```
keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
steps_per_epoch = len(X_train) // batch_size
history = model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=5)
```

The DenseFeatures layer will take care of converting every input feature to a dense
representation, and it will also apply any extra transformation we specified, such as
scaling the housing_median_age using the normalizer_fn function we provided. You
can take a closer look at what the DenseFeatures layer does by calling it directly:

```
>>> some_columns = [ocean_proximity_embed, bucketized_income]
>>> dense_features = keras.layers.DenseFeatures(some_columns)
>>> dense_features({
... "ocean_proximity": [["NEAR OCEAN"], ["INLAND"], ["INLAND"]],
... "median_income": [[3.], [7.2], [1.]]
... })
...
<tf.Tensor: id=559790, shape=(3, 7), dtype=float32, numpy=
array([[ 0. , 0. , 1. , 0. , 0. ,-0.36277947 , 0.30109018],
[ 0. , 0. , 0. , 0. , 1. , 0.22548223 , 0.33142096],
[ 1. , 0. , 0. , 0. , 0. , 0.22548223 , 0.33142096]], dtype=float32)>
```

In this example, we create a DenseFeatures layer with just two columns, and we call
it with some data, in the form of a dictionary of features. In this case, since the bucke
tized_income column relies on the median_income column, the dictionary must
include the "median_income" key, and similarly since the ocean_proximity_embed
column is based on the ocean_proximity column, the dictionary must include the
"ocean_proximity" key. Columns are handled in alphabetical order, so first we look
at the bucketized income column (its name is the same as the median_income column
name, plus "\_bucketized"). The incomes 3, 7.2 and 1 get mapped respectively to cat‐
egory 2 (for incomes between 1.5 and 3), category 0 (for incomes below 1.5), and cat‐
egory 4 (for incomes greater than 6). Then these category IDs get one-hot encoded:
category 2 gets encoded as [0., 0., 1., 0., 0.] and so on (note that bucketized
columns get one-hot encoded by default, no need to call indicator_column()). Now
on to the ocean_proximity_embed column. The "NEAR OCEAN" and "INLAND" cate‐
gories just get mapped to their respective embeddings (which were initialized ran‐
domly). The resulting tensor is the concatenation of the one-hot vectors and the
embeddings.

Now you can feed all kinds of features to a neural network, including numerical fea‐
tures, categorical features, and even text (by splitting the text into words, then using
word embedding)! However, performing all the preprocessing on the fly can slow
down training. Let’s see how this can be improved.

```
The Features API | 427
```

### TF Transform

If preprocessing is computationally expensive, then handling it before training rather
than on the fly may give you a significant speedup: the data will be preprocessed just
once per instance before training, rather than once per instance and per epoch during
training. Tools like Apache Beam let you run efficient data processing pipelines over
large amounts of data, even distributed across multiple servers, so why not use it to
preprocess all the training data? This works great and indeed can speed up training,
but there is one problem: once your model is trained, suppose you want to deploy it
to a mobile app: you will need to write some code in your app to take care of prepro‐
cessing the data before it is fed to the model. And suppose you also want to deploy
the model to TensorFlow.js so it runs in a web browser? Once again, you will need to
write some preprocessing code. This can become a maintenance nightmare: when‐
ever you want to change the preprocessing logic, you will need to update your Apache
Beam code, your mobile app code and your Javascript code. It is not only time con‐
suming, but also error prone: you may end up with subtle differences between the
preprocessing operations performed before training and the ones performed in your
app or in the browser. This training/serving skew will lead to bugs or degraded perfor‐
mance.

One improvement would be to take the trained model (trained on data that was pre‐
processed by your Apache Beam code), and before deploying it to your app or the
browser, add an extra input layer to take care of preprocessing on the fly (either by
writing a custom layer or by using a DenseFeatures layer). That’s definitely better,
since now you just have two versions of your preprocessing code: the Apache Beam
code and the preprocessing layer’s code.

But what if you could define your preprocessing operations just once? This is what
TF Transform was designed for. It is part of TensorFlow Extended (TFX), an end-to-
end platform for productionizing TensorFlow models. First, to use a TFX component,
such as TF Transform, you must install it, it does not come bundled with TensorFlow.
You define your preprocessing function just once (in Python), by using TF Transform
functions for scaling, bucketizing, crossing features, and more. You can also use any
TensorFlow operation you need. Here is what this preprocessing function might look
like if we just had two features:

```
import tensorflow_transform as tft
```

```
def preprocess(inputs): # inputs is a batch of input features
median_age = inputs["housing_median_age"]
ocean_proximity = inputs["ocean_proximity"]
standardized_age = tft.scale_to_z_score(median_age - tft.mean(median_age))
ocean_proximity_id = tft.compute_and_apply_vocabulary(ocean_proximity)
return {
"standardized_median_age": standardized_age,
```

**428 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

11 At the time of writing, TFDS requires you to download a few files manually for ImageNet (for legal reasons),
but this will hopefully get resolved soon.

```
"ocean_proximity_id": ocean_proximity_id
}
Next, TF Transform lets you apply this preprocess() function to the whole training
set using Apache Beam (it provides an AnalyzeAndTransformDataset class that you
can use for this purpose in your Apache Beam pipeline). In the process, it will also
compute all the necessary statistics over the whole training set: in this example, the
mean and standard deviation of the housing_median_age feature, and the vocabulary
for the ocean_proximity feature. The components that compute these statistics are
called analyzers.
Importantly, TF Transform will also generate an equivalent TensorFlow Function that
you can plug into the model you deploy. This TF Function contains all the necessary
statistics computed by Apache Beam (the mean, standard deviation, and vocabulary),
simply included as constants.
```

```
At the time of this writing, TF Transform only supports Tensor‐
Flow 1. Moreover, Apache Beam only has partial support for
Python 3. That said, both these limitations will likely be fixed by
the time your read this.
```

```
With the Data API, TFRecords, the Features API and TF Transform, you can build
highly scalable input pipelines for training, and also benefit from fast and portable
data preprocessing in production.
But what if you just wanted to use a standard dataset? Well in that case, things are
much simpler: just use TFDS!
```

### The TensorFlow Datasets (TFDS) Project

```
The TensorFlow Datasets project makes it trivial to download common datasets, from
small ones like MNIST or Fashion MNIST, to huge datasets like ImageNet^11 (you will
need quite a bit of disk space!). The list includes image datasets, text datasets (includ‐
ing translation datasets), audio and video datasets, and more. You can visit https://
homl.info/tfds to view the full list, along with a description of each dataset.
TFDS is not bundled with TensorFlow, so you need to install the tensorflow-
datasets library (e.g., using pip). Then all you need to do is call the tfds.load()
function, and it will download the data you want (unless it was already downloaded
earlier), and return the data as a dictionary of Datasets (typically one for training,
```

```
The TensorFlow Datasets (TFDS) Project | 429
```

and one for testing, but this depends on the dataset you choose). For example, let’s
download MNIST:

```
import tensorflow_datasets as tfds
```

```
dataset = tfds.load(name="mnist")
mnist_train, mnist_test = dataset["train"], dataset["test"]
```

You can then apply any transformation you want (typically repeating, batching and
prefetching), and you’re ready to train your model. Here is a simple example:

```
mnist_train = mnist_train.repeat(5).batch(32).prefetch(1)
for item in mnist_train:
images = item["image"]
labels = item["label"]
[...]
```

```
In general, load() returns a shuffled training set, so there’s no need
to shuffle it some more.
```

Note that each item in the dataset is a dictionary containing both the features and the
labels. But Keras expects each item to be a tuple containing 2 elements (again, the fea‐
tures and the labels). You could transform the dataset using the map() method, like
this:

```
mnist_train = mnist_train.repeat(5).batch(32)
mnist_train = mnist_train.map( lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefetch(1)
```

Or you can just ask the load() function to do this for you by setting as_super
vised=True (obviously this works only for labeled datasets). You can also specify the
batch size if you want. Then the dataset can be passed directly to your tf.keras model:

```
dataset = tfds.load(name="mnist", batch_size=32, as_supervised=True)
mnist_train = dataset["train"].repeat().prefetch(1)
model = keras.models.Sequential([...])
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
model.fit(mnist_train, steps_per_epoch=60000 // 32, epochs=5)
```

This was quite a technical chapter, and you may feel that it is a bit far from the
abstract beauty of neural networks, but the fact is deep learning often involves large
amounts of data, and knowing how to load, parse and preprocess it efficiently is a
crucial skill to have. In the next chapter, we will look at Convolutional Neural Net‐
works, which are among the most successful neural net architectures for image pro‐
cessing, and many other applications.

**430 | Chapter 13: Loading and Preprocessing Data with TensorFlow**

**CHAPTER 14**

**Deep Computer Vision Using Convolutional**

**Neural Networks**

```
With Early Release ebooks, you get books in their earliest form—
the author’s raw and unedited content as he or she writes—so you
can take advantage of these technologies long before the official
release of these titles. The following will be Chapter 14 in the final
release of the book.
```

Although IBM’s Deep Blue supercomputer beat the chess world champion Garry Kas‐
parov back in 1996, it wasn’t until fairly recently that computers were able to reliably
perform seemingly trivial tasks such as detecting a puppy in a picture or recognizing
spoken words. Why are these tasks so effortless to us humans? The answer lies in the
fact that perception largely takes place outside the realm of our consciousness, within
specialized visual, auditory, and other sensory modules in our brains. By the time
sensory information reaches our consciousness, it is already adorned with high-level
features; for example, when you look at a picture of a cute puppy, you cannot choose
not to see the puppy, or not to notice its cuteness. Nor can you explain how you rec‐
ognize a cute puppy; it’s just obvious to you. Thus, we cannot trust our subjective
experience: perception is not trivial at all, and to understand it we must look at how
the sensory modules work.

Convolutional neural networks (CNNs) emerged from the study of the brain’s visual
cortex, and they have been used in image recognition since the 1980s. In the last few
years, thanks to the increase in computational power, the amount of available training
data, and the tricks presented in Chapter 11 for training deep nets, CNNs have man‐
aged to achieve superhuman performance on some complex visual tasks. They power
image search services, self-driving cars, automatic video classification systems, and
more. Moreover, CNNs are not restricted to visual perception: they are also successful

```
431
```

```
1 “Single Unit Activity in Striate Cortex of Unrestrained Cats,” D. Hubel and T. Wiesel (1958).
2 “Receptive Fields of Single Neurones in the Cat’s Striate Cortex,” D. Hubel and T. Wiesel (1959).
3 “Receptive Fields and Functional Architecture of Monkey Striate Cortex,” D. Hubel and T. Wiesel (1968).
```

at many other tasks, such as voice recognition or natural language processing (NLP);
however, we will focus on visual applications for now.

In this chapter we will present where CNNs came from, what their building blocks
look like, and how to implement them using TensorFlow and Keras. Then we will dis‐
cuss some of the best CNN architectures, and discuss other visual tasks, including
object detection (classifying multiple objects in an image and placing bounding boxes
around them) and semantic segmentation (classifying each pixel according to the class
of the object it belongs to).

### The Architecture of the Visual Cortex

David H. Hubel and Torsten Wiesel performed a series of experiments on cats in
19581 and 19592 (and a few years later on monkeys^3 ), giving crucial insights on the
structure of the visual cortex (the authors received the Nobel Prize in Physiology or
Medicine in 1981 for their work). In particular, they showed that many neurons in
the visual cortex have a small local receptive field, meaning they react only to visual
stimuli located in a limited region of the visual field (see Figure 14-1, in which the
local receptive fields of five neurons are represented by dashed circles). The receptive
fields of different neurons may overlap, and together they tile the whole visual field.
Moreover, the authors showed that some neurons react only to images of horizontal
lines, while others react only to lines with different orientations (two neurons may
have the same receptive field but react to different line orientations). They also
noticed that some neurons have larger receptive fields, and they react to more com‐
plex patterns that are combinations of the lower-level patterns. These observations
led to the idea that the higher-level neurons are based on the outputs of neighboring
lower-level neurons (in Figure 14-1, notice that each neuron is connected only to a
few neurons from the previous layer). This powerful architecture is able to detect all
sorts of complex patterns in any area of the visual field.

**432 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

```
4 “Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected
by Shift in Position,” K. Fukushima (1980).
5 “Gradient-Based Learning Applied to Document Recognition,” Y. LeCun et al. (1998).
```

Figure 14-1. Local receptive fields in the visual cortex

These studies of the visual cortex inspired the neocognitron, introduced in 1980,^4
which gradually evolved into what we now call convolutional neural networks. An
important milestone was a 1998 paper^5 by Yann LeCun, Léon Bottou, Yoshua Bengio,
and Patrick Haffner, which introduced the famous LeNet-5 architecture, widely used
to recognize handwritten check numbers. This architecture has some building blocks
that you already know, such as fully connected layers and sigmoid activation func‐
tions, but it also introduces two new building blocks: convolutional layers and pooling
layers. Let’s look at them now.

```
Why not simply use a regular deep neural network with fully con‐
nected layers for image recognition tasks? Unfortunately, although
this works fine for small images (e.g., MNIST), it breaks down for
larger images because of the huge number of parameters it
requires. For example, a 100 × 100 image has 10,000 pixels, and if
the first layer has just 1,000 neurons (which already severely
restricts the amount of information transmitted to the next layer),
this means a total of 10 million connections. And that’s just the first
layer. CNNs solve this problem using partially connected layers and
weight sharing.
```

```
The Architecture of the Visual Cortex | 433
```

```
6 A convolution is a mathematical operation that slides one function over another and measures the integral of
their pointwise multiplication. It has deep connections with the Fourier transform and the Laplace transform,
and is heavily used in signal processing. Convolutional layers actually use cross-correlations, which are very
similar to convolutions (see https://homl.info/76 for more details).
```

### Convolutional Layer

The most important building block of a CNN is the convolutional layer:^6 neurons in
the first convolutional layer are not connected to every single pixel in the input image
(like they were in previous chapters), but only to pixels in their receptive fields (see
Figure 14-2). In turn, each neuron in the second convolutional layer is connected
only to neurons located within a small rectangle in the first layer. This architecture
allows the network to concentrate on small low-level features in the first hidden layer,
then assemble them into larger higher-level features in the next hidden layer, and so
on. This hierarchical structure is common in real-world images, which is one of the
reasons why CNNs work so well for image recognition.

Figure 14-2. CNN layers with rectangular local receptive fields

```
Until now, all multilayer neural networks we looked at had layers
composed of a long line of neurons, and we had to flatten input
images to 1D before feeding them to the neural network. Now each
layer is represented in 2D, which makes it easier to match neurons
with their corresponding inputs.
```

A neuron located in row i, column j of a given layer is connected to the outputs of the
neurons in the previous layer located in rows i to i + fh – 1, columns j to j + fw – 1,
where fh and fw are the height and width of the receptive field (see Figure 14-3). In
order for a layer to have the same height and width as the previous layer, it is com‐

**434 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

mon to add zeros around the inputs, as shown in the diagram. This is called zero pad‐
ding.

Figure 14-3. Connections between layers and zero padding

It is also possible to connect a large input layer to a much smaller layer by spacing out
the receptive fields, as shown in Figure 14-4. The shift from one receptive field to the
next is called the stride. In the diagram, a 5 × 7 input layer (plus zero padding) is con‐
nected to a 3 × 4 layer, using 3 × 3 receptive fields and a stride of 2 (in this example
the stride is the same in both directions, but it does not have to be so). A neuron loca‐
ted in row i, column j in the upper layer is connected to the outputs of the neurons in
the previous layer located in rows i × sh to i × sh + fh – 1, columns j × sw to j × sw + fw –
1, where sh and sw are the vertical and horizontal strides.

```
Convolutional Layer | 435
```

Figure 14-4. Reducing dimensionality using a stride of 2

#### Filters

A neuron’s weights can be represented as a small image the size of the receptive field.
For example, Figure 14-5 shows two possible sets of weights, called filters (or convolu‐
tion kernels). The first one is represented as a black square with a vertical white line in
the middle (it is a 7 × 7 matrix full of 0s except for the central column, which is full of
1s); neurons using these weights will ignore everything in their receptive field except
for the central vertical line (since all inputs will get multiplied by 0, except for the
ones located in the central vertical line). The second filter is a black square with a
horizontal white line in the middle. Once again, neurons using these weights will
ignore everything in their receptive field except for the central horizontal line.

Now if all neurons in a layer use the same vertical line filter (and the same bias term),
and you feed the network the input image shown in Figure 14-5 (bottom image), the
layer will output the top-left image. Notice that the vertical white lines get enhanced
while the rest gets blurred. Similarly, the upper-right image is what you get if all neu‐
rons use the same horizontal line filter; notice that the horizontal white lines get
enhanced while the rest is blurred out. Thus, a layer full of neurons using the same
filter outputs a feature map, which highlights the areas in an image that activate the
filter the most. Of course you do not have to define the filters manually: instead, dur‐
ing training the convolutional layer will automatically learn the most useful filters for
its task, and the layers above will learn to combine them into more complex patterns.

**436 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

Figure 14-5. Applying two different filters to get two feature maps

#### Stacking Multiple Feature Maps

Up to now, for simplicity, I have represented the output of each convolutional layer as
a thin 2D layer, but in reality a convolutional layer has multiple filters (you decide
how many), and it outputs one feature map per filter, so it is more accurately repre‐
sented in 3D (see Figure 14-6). To do so, it has one neuron per pixel in each feature
map, and all neurons within a given feature map share the same parameters (i.e., the
same weights and bias term). However, neurons in different feature maps use differ‐
ent parameters. A neuron’s receptive field is the same as described earlier, but it
extends across all the previous layers’ feature maps. In short, a convolutional layer
simultaneously applies multiple trainable filters to its inputs, making it capable of
detecting multiple features anywhere in its inputs.

```
The fact that all neurons in a feature map share the same parame‐
ters dramatically reduces the number of parameters in the model.
Moreover, once the CNN has learned to recognize a pattern in one
location, it can recognize it in any other location. In contrast, once
a regular DNN has learned to recognize a pattern in one location, it
can recognize it only in that particular location.
```

Moreover, input images are also composed of multiple sublayers: one per color chan‐
nel. There are typically three: red, green, and blue (RGB). Grayscale images have just

```
Convolutional Layer | 437
```

one channel, but some images may have much more—for example, satellite images
that capture extra light frequencies (such as infrared).

Figure 14-6. Convolution layers with multiple feature maps, and images with three color
channels

Specifically, a neuron located in row i, column j of the feature map k in a given convo‐
lutional layer l is connected to the outputs of the neurons in the previous layer l – 1,
located in rows i × sh to i × sh + fh – 1 and columns j × sw to j × sw + fw – 1, across all
feature maps (in layer l – 1 ). Note that all neurons located in the same row i and col‐
umn j but in different feature maps are connected to the outputs of the exact same
neurons in the previous layer.

Equation 14-1 summarizes the preceding explanations in one big mathematical equa‐
tion: it shows how to compute the output of a given neuron in a convolutional layer.

**438 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

It is a bit ugly due to all the different indices, but all it does is calculate the weighted
sum of all the inputs, plus the bias term.

```
Equation 14-1. Computing the output of a neuron in a convolutional layer
```

```
zi,j,k=bk+ ∑
u= 0
```

```
fh− 1
∑
v= 0
```

```
fw− 1
∑
k′= 0
```

```
fn′− 1
xi′,j′,k′.wu,v,k′,k with
```

```
i′=i×sh+u
j′=j×sw+v
```

- zi, j, k is the output of the neuron located in row i, column j in feature map k of the
  convolutional layer (layer l).
- As explained earlier, sh and sw are the vertical and horizontal strides, fh and fw are
  the height and width of the receptive field, and fn′ is the number of feature maps
  in the previous layer (layer l – 1).
- xi′, j′, k′ is the output of the neuron located in layer l – 1, row i′, column j′, feature
  map k′ (or channel k′ if the previous layer is the input layer).
- bk is the bias term for feature map k (in layer l). You can think of it as a knob that
  tweaks the overall brightness of the feature map k.
- wu, v, k′ ,k is the connection weight between any neuron in feature map k of the layer
  l and its input located at row u, column v (relative to the neuron’s receptive field),
  and feature map k′.

#### TensorFlow Implementation

In TensorFlow, each input image is typically represented as a 3D tensor of shape
[height, width, channels]. A mini-batch is represented as a 4D tensor of shape
[mini-batch size, height, width, channels]. The weights of a convolutional
layer are represented as a 4D tensor of shape [fh, fw, fn′, fn]. The bias terms of a convo‐
lutional layer are simply represented as a 1D tensor of shape [fn].

Let’s look at a simple example. The following code loads two sample images, using
Scikit-Learn’s load_sample_images() (which loads two color images, one of a Chi‐
nese temple, and the other of a flower). The pixel intensities (for each color channel)
is represented as a byte from 0 to 255, so we scale these features simply by dividing by
255, to get floats ranging from 0 to 1. Then we create two 7 × 7 filters (one with a
vertical white line in the middle, and the other with a horizontal white line in the
middle), and we apply them to both images using the tf.nn.conv2d() function,
which is part of TensorFlow’s low-level Deep Learning API. In this example, we use
zero padding (padding="SAME") and a stride of 2. Finally, we plot one of the resulting
feature maps (similar to the top-right image in Figure 14-5).

```
Convolutional Layer | 439
```

```
from sklearn.datasets import load_sample_image
```

```
# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape
```

```
# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1 # vertical line
filters[3, :, :, 1] = 1 # horizontal line
```

```
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
```

```
plt.imshow(outputs[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.show()
```

Most of this code is self-explanatory, but the tf.nn.conv2d() line deserves a bit of
explanation:

- images is the input mini-batch (a 4D tensor, as explained earlier).
- filters is the set of filters to apply (also a 4D tensor, as explained earlier).
- strides is equal to 1, but it could also be a 1D array with 4 elements, where the
  two central elements are the vertical and horizontal strides (sh and sw). The first
  and last elements must currently be equal to 1. They may one day be used to
  specify a batch stride (to skip some instances) and a channel stride (to skip some
  of the previous layer’s feature maps or channels).
- padding must be either "VALID" or "SAME":
  —If set to "VALID", the convolutional layer does not use zero padding, and may
  ignore some rows and columns at the bottom and right of the input image,
  depending on the stride, as shown in Figure 14-7 (for simplicity, only the hor‐
  izontal dimension is shown here, but of course the same logic applies to the
  vertical dimension).
  —If set to "SAME", the convolutional layer uses zero padding if necessary. In this
  case, the number of output neurons is equal to the number of input neurons
  divided by the stride, rounded up (in this example, 13 / 5 = 2.6, rounded up to
  3). Then zeros are added as evenly as possible around the inputs.

**440 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

Figure 14-7. Padding options—input width: 13, filter width: 6, stride: 5

In this example, we manually defined the filters, but in a real CNN you would nor‐
mally define filters as trainable variables, so the neural net can learn which filters
work best, as explained earlier. Instead of manually creating the variables, however,
you can simply use the keras.layers.Conv2D layer:

```
conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
padding="SAME", activation="relu")
```

This code creates a Conv2D layer with 32 filters, each 3 × 3, using a stride of 1 (both
horizontally and vertically), SAME padding, and applying the ReLU activation func‐
tion to its outputs. As you can see, convolutional layers have quite a few hyperpara‐
meters: you must choose the number of filters, their height and width, the strides, and
the padding type. As always, you can use cross-validation to find the right hyperpara‐
meter values, but this is very time-consuming. We will discuss common CNN archi‐
tectures later, to give you some idea of what hyperparameter values work best in
practice.

### Memory Requirements

Another problem with CNNs is that the convolutional layers require a huge amount
of RAM. This is especially true during training, because the reverse pass of backpro‐
pagation requires all the intermediate values computed during the forward pass.

For example, consider a convolutional layer with 5 × 5 filters, outputting 200 feature
maps of size 150 × 100, with stride 1 and SAME padding. If the input is a 150 × 100

```
Convolutional Layer | 441
```

```
7 A fully connected layer with 150 × 100 neurons, each connected to all 150 × 100 × 3 inputs, would have 150^2
× 100^2 × 3 = 675 million parameters!
8 In the international system of units (SI), 1 MB = 1,000 kB = 1,000 × 1,000 bytes = 1,000 × 1,000 × 8 bits.
```

RGB image (three channels), then the number of parameters is (5 × 5 × 3 + 1) × 200
= 15,200 (the +1 corresponds to the bias terms), which is fairly small compared to a
fully connected layer.^7 However, each of the 200 feature maps contains 150 × 100 neu‐
rons, and each of these neurons needs to compute a weighted sum of its 5 × 5 × 3 =
75 inputs: that’s a total of 225 million float multiplications. Not as bad as a fully con‐
nected layer, but still quite computationally intensive. Moreover, if the feature maps
are represented using 32-bit floats, then the convolutional layer’s output will occupy
200 × 150 × 100 × 32 = 96 million bits (12 MB) of RAM.^8 And that’s just for one
instance! If a training batch contains 100 instances, then this layer will use up 1.2 GB
of RAM!

During inference (i.e., when making a prediction for a new instance) the RAM occu‐
pied by one layer can be released as soon as the next layer has been computed, so you
only need as much RAM as required by two consecutive layers. But during training
everything computed during the forward pass needs to be preserved for the reverse
pass, so the amount of RAM needed is (at least) the total amount of RAM required by
all layers.

```
If training crashes because of an out-of-memory error, you can try
reducing the mini-batch size. Alternatively, you can try reducing
dimensionality using a stride, or removing a few layers. Or you can
try using 16-bit floats instead of 32-bit floats. Or you could distrib‐
ute the CNN across multiple devices.
```

Now let’s look at the second common building block of CNNs: the pooling layer.

## Pooling Layer

Once you understand how convolutional layers work, the pooling layers are quite
easy to grasp. Their goal is to subsample (i.e., shrink) the input image in order to
reduce the computational load, the memory usage, and the number of parameters
(thereby limiting the risk of overfitting).

Just like in convolutional layers, each neuron in a pooling layer is connected to the
outputs of a limited number of neurons in the previous layer, located within a small
rectangular receptive field. You must define its size, the stride, and the padding type,
just like before. However, a pooling neuron has no weights; all it does is aggregate the
inputs using an aggregation function such as the max or mean. Figure 14-8 shows a
max pooling layer, which is the most common type of pooling layer. In this example,

**442 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

```
9 Other kernels we discussed so far had weights, but pooling kernels do not: they are just stateless sliding win‐
dows.
```

we use a 2 × 2 _pooling kernel_^9 , with a stride of 2, and no padding. Only the max
input value in each receptive field makes it to the next layer, while the other inputs
are dropped. For example, in the lower left receptive field in Figure 14-8, the input
values are 1, 5, 3, 2, so only the max value, 5, is propagated to the next layer. Because
of the stride of 2, the output image has half the height and half the width of the input
image (rounded down since we use no padding).

Figure 14-8. Max pooling layer (2 × 2 pooling kernel, stride 2, no padding)

```
A pooling layer typically works on every input channel independ‐
ently, so the output depth is the same as the input depth.
```

Other than reducing computations, memory usage and the number of parameters, a
max pooling layer also introduces some level of invariance to small translations, as
shown in Figure 14-9. Here we assume that the bright pixels have a lower value than
dark pixels, and we consider 3 images (A, B, C) going through a max pooling layer
with a 2 × 2 kernel and stride 2. Images B and C are the same as image A, but shifted
by one and two pixels to the right. As you can see, the outputs of the max pooling
layer for images A and B are identical. This is what translation invariance means.
However, for image C, the output is different: it is shifted by one pixel to the right
(but there is still 75% invariance). By inserting a max pooling layer every few layers in
a CNN, it is possible to get some level of translation invariance at a larger scale.
Moreover, max pooling also offers a small amount of rotational invariance and a
slight scale invariance. Such invariance (even if it is limited) can be useful in cases
where the prediction should not depend on these details, such as in classification
tasks.

```
Pooling Layer | 443
```

Figure 14-9. Invariance to small translations

But max pooling has some downsides: firstly, it is obviously very destructive: even
with a tiny 2 × 2 kernel and a stride of 2, the output will be two times smaller in both
directions (so its area will be four times smaller), simply dropping 75% of the input
values. And in some applications, invariance is not desirable, for example for seman‐
tic segmentation: this is the task of classifying each pixel in an image depending on the
object that pixel belongs to: obviously, if the input image is translated by 1 pixel to the
right, the output should also be translated by 1 pixel to the right. The goal in this case
is equivariance, not invariance: a small change to the inputs should lead to a corre‐
sponding small change in the output.

### TensorFlow Implementation

Implementing a max pooling layer in TensorFlow is quite easy. The following code
creates a max pooling layer using a 2 × 2 kernel. The strides default to the kernel size,
so this layer will use a stride of 2 (both horizontally and vertically). By default, it uses
VALID padding (i.e., no padding at all):

```
max_pool = keras.layers.MaxPool2D(pool_size=2)
```

To create an average pooling layer, just use AvgPool2D instead of MaxPool2D. As you
might expect, it works exactly like a max pooling layer, except it computes the mean
rather than the max. Average pooling layers used to be very popular, but people

**444 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

mostly use max pooling layers now, as they generally perform better. This may seem
surprising, since computing the mean generally loses less information than comput‐
ing the max. But on the other hand, max pooling preserves only the strongest feature,
getting rid of all the meaningless ones, so the next layers get a cleaner signal to work
with. Moreover, max pooling offers stronger translation invariance than average
pooling.

Note that max pooling and average pooling can be performed along the depth dimen‐
sion rather than the spatial dimensions, although this is not as common. This can
allow the CNN to learn to be invariant to various features. For example, it could learn
multiple filters, each detecting a different rotation of the same pattern, such as hand-
written digits (see Figure 14-10), and the depth-wise max pooling layer would ensure
that the output is the same regardless of the rotation. The CNN could similarly learn
to be invariant to anything else: thickness, brightness, skew, color, and so on.

Figure 14-10. Depth-wise max pooling can help the CNN learn any invariance

```
Pooling Layer | 445
```

Keras does not include a depth-wise max pooling layer, but TensorFlow’s low-level
Deep Learning API does: just use the tf.nn.max_pool() function, and specify the
kernel size and strides as 4-tuples. The first three values of each should be 1: this indi‐
cates that the kernel size and stride along the batch, height and width dimensions
shoud be 1. The last value should be whatever kernel size and stride you want along
the depth dimension, for example 3 (this must be a divisor of the input depth; for
example, it will not work if the previous layer outputs 20 feature maps, since 20 is not
a multiple of 3):

```
output = tf.nn.max_pool(images,
ksize=(1, 1, 1, 3),
strides=(1, 1, 1, 3),
padding="VALID")
```

If you want to include this as a layer in your Keras models, you can simply wrap it in
a Lambda layer (or create a custom Keras layer):

```
depth_pool = keras.layers.Lambda(
lambda X: tf.nn.max_pool(X, ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3),
padding="VALID"))
```

One last type of pooling layer that you will often see in modern architectures is the
global average pooling layer. It works very differently: all it does is compute the mean
of each entire feature map (it’s like an average pooling layer using a pooling kernel
with the same spatial dimensions as the inputs). This means that it just outputs a sin‐
gle number per feature map and per instance. Although this is of course extremely
destructive (most of the information in the feature map is lost), it can be useful as the
output layer, as we will see later in this chapter. To create such a layer, simply use the
keras.layers.GlobalAvgPool2D class:

```
global_avg_pool = keras.layers.GlobalAvgPool2D()
```

It is actually equivalent to this simple Lamba layer, which computes the mean over the
spatial dimensions (height and width):

```
global_avg_pool = keras.layers.Lambda( lambda X: tf.reduce_mean(X, axis=[1, 2]))
```

Now you know all the building blocks to create a convolutional neural network. Let’s
see how to assemble them.

## CNN Architectures

Typical CNN architectures stack a few convolutional layers (each one generally fol‐
lowed by a ReLU layer), then a pooling layer, then another few convolutional layers
(+ReLU), then another pooling layer, and so on. The image gets smaller and smaller
as it progresses through the network, but it also typically gets deeper and deeper (i.e.,
with more feature maps) thanks to the convolutional layers (see Figure 14-11). At the
top of the stack, a regular feedforward neural network is added, composed of a few

**446 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

fully connected layers (+ReLUs), and the final layer outputs the prediction (e.g., a
softmax layer that outputs estimated class probabilities).

Figure 14-11. Typical CNN architecture

```
A common mistake is to use convolution kernels that are too large.
For example, instead of using a convolutional layer with a 5 × 5
kernel, it is generally preferable to stack two layers with 3 × 3 ker‐
nels: it will use less parameters and require less computations, and
it will usually perform better. One exception to this recommenda‐
tion is for the first convolutional layer: it can typically have a large
kernel (e.g., 5 × 5), usually with stride of 2 or more: this will reduce
the spatial dimension of the image without losing too much infor‐
mation, and since the input image only has 3 channels in general, it
will not be too costly.
```

Here is how you can implement a simple CNN to tackle the fashion MNIST dataset
(introduced in Chapter 10):

```
from functools import partial
```

```
DefaultConv2D = partial(keras.layers.Conv2D,
kernel_size=3, activation='relu', padding="SAME")
```

```
model = keras.models.Sequential([
DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
keras.layers.MaxPooling2D(pool_size=2),
DefaultConv2D(filters=128),
DefaultConv2D(filters=128),
keras.layers.MaxPooling2D(pool_size=2),
DefaultConv2D(filters=256),
DefaultConv2D(filters=256),
keras.layers.MaxPooling2D(pool_size=2),
keras.layers.Flatten(),
keras.layers.Dense(units=128, activation='relu'),
keras.layers.Dropout(0.5),
keras.layers.Dense(units=64, activation='relu'),
keras.layers.Dropout(0.5),
keras.layers.Dense(units=10, activation='softmax'),
])
```

```
CNN Architectures | 447
```

- In this code, we start by using the partial() function to define a thin wrapper
  around the Conv2D class, called DefaultConv2D: it simply avoids having to repeat
  the same hyperparameter values over and over again.
- The first layer uses a large kernel size, but no stride because the input images are
  not very large. It also sets input_shape=[28, 28, 1], which means the images
  are 28 × 28 pixels, with a single color channel (i.e., grayscale).
- Next, we have a max pooling layer, which divides each spatial dimension by a fac‐
  tor of two (since pool_size=2).
- Then we repeat the same structure twice: two convolutional layers followed by a
  max pooling layer. For larger images, we could repeat this structure several times
  (the number of repetitions is a hyperparameter you can tune).
- Note that the number of filters grows as we climb up the CNN towards the out‐
  put layer (it is initially 64, then 128, then 256): it makes sense for it to grow, since
  the number of low level features is often fairly low (e.g., small circles, horizontal
  lines, etc.), but there are many different ways to combine them into higher level
  features. It is a common practice to double the number of filters after each pool‐
  ing layer: since a pooling layer divides each spatial dimension by a factor of 2, we
  can afford doubling the number of feature maps in the next layer, without fear of
  exploding the number of parameters, memory usage, or computational load.
- Next is the fully connected network, composed of 2 hidden dense layers and a
  dense output layer. Note that we must flatten its inputs, since a dense network
  expects a 1D array of features for each instance. We also add two dropout layers,
  with a dropout rate of 50% each, to reduce overfitting.

This CNN reaches over 92% accuracy on the test set. It’s not the state of the art, but it
is pretty good, and clearly much better than what we achieved with dense networks in
Chapter 10.

Over the years, variants of this fundamental architecture have been developed, lead‐
ing to amazing advances in the field. A good measure of this progress is the error rate
in competitions such as the ILSVRC ImageNet challenge. In this competition the
top-5 error rate for image classification fell from over 26% to less than 2.3% in just six
years. The top-five error rate is the number of test images for which the system’s top 5
predictions did not include the correct answer. The images are large (256 pixels high)
and there are 1,000 classes, some of which are really subtle (try distinguishing 120
dog breeds). Looking at the evolution of the winning entries is a good way to under‐
stand how CNNs work.

We will first look at the classical LeNet-5 architecture (1998), then three of the win‐
ners of the ILSVRC challenge: AlexNet (2012), GoogLeNet (2014), and ResNet
(2015).

**448 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

10 “Gradient-Based Learning Applied to Document Recognition”, Y. LeCun, L. Bottou, Y. Bengio and P. Haffner
(1998).

### LeNet-5

```
The LeNet-5 architecture^10 is perhaps the most widely known CNN architecture. As
mentioned earlier, it was created by Yann LeCun in 1998 and widely used for hand‐
written digit recognition (MNIST). It is composed of the layers shown in Table 14-1.
```

```
Table 14-1. LeNet-5 architecture
Layer Type Maps Size Kernel size Stride Activation
Out Fully Connected – 10 – – RBF
F6 Fully Connected – 84 – – tanh
C5 Convolution 120 1 × 1 5 × 5 1 tanh
S4 Avg Pooling 16 5 × 5 2 × 2 2 tanh
C3 Convolution 16 10 × 10 5 × 5 1 tanh
S2 Avg Pooling 6 14 × 14 2 × 2 2 tanh
C1 Convolution 6 28 × 28 5 × 5 1 tanh
In Input 1 32 × 32 – – –
```

```
There are a few extra details to be noted:
```

- MNIST images are 28 × 28 pixels, but they are zero-padded to 32 × 32 pixels and
  normalized before being fed to the network. The rest of the network does not use
  any padding, which is why the size keeps shrinking as the image progresses
  through the network.
- The average pooling layers are slightly more complex than usual: each neuron
  computes the mean of its inputs, then multiplies the result by a learnable coeffi‐
  cient (one per map) and adds a learnable bias term (again, one per map), then
  finally applies the activation function.
- Most neurons in C3 maps are connected to neurons in only three or four S2
  maps (instead of all six S2 maps). See table 1 (page 8) in the original paper^10 for
  details.
- The output layer is a bit special: instead of computing the matrix multiplication
  of the inputs and the weight vector, each neuron outputs the square of the Eucli‐
  dian distance between its input vector and its weight vector. Each output meas‐
  ures how much the image belongs to a particular digit class. The cross entropy
  cost function is now preferred, as it penalizes bad predictions much more, pro‐
  ducing larger gradients and converging faster.

```
CNN Architectures | 449
```

11 “ImageNet Classification with Deep Convolutional Neural Networks,” A. Krizhevsky et al. (2012).

```
Yann LeCun’s website (“LENET” section) features great demos of LeNet-5 classifying
digits.
```

### AlexNet

```
The AlexNet CNN architecture^11 won the 2012 ImageNet ILSVRC challenge by a
large margin: it achieved 17% top-5 error rate while the second best achieved only
26%! It was developed by Alex Krizhevsky (hence the name), Ilya Sutskever, and
Geoffrey Hinton. It is quite similar to LeNet-5, only much larger and deeper, and it
was the first to stack convolutional layers directly on top of each other, instead of
stacking a pooling layer on top of each convolutional layer. Table 14-2 presents this
architecture.
```

```
Table 14-2. AlexNet architecture
Layer Type Maps Size Kernel size Stride Padding Activation
Out Fully Connected – 1,000 – – – Softmax
F9 Fully Connected – 4,096 – – – ReLU
F8 Fully Connected – 4,096 – – – ReLU
C7 Convolution 256 13 × 13 3 × 3 1 SAME ReLU
C6 Convolution 384 13 × 13 3 × 3 1 SAME ReLU
C5 Convolution 384 13 × 13 3 × 3 1 SAME ReLU
S4 Max Pooling 256 13 × 13 3 × 3 2 VALID –
C3 Convolution 256 27 × 27 5 × 5 1 SAME ReLU
S2 Max Pooling 96 27 × 27 3 × 3 2 VALID –
C1 Convolution 96 55 × 55 11 × 11 4 VALID ReLU
In Input 3 (RGB) 227 × 227 – – – –
```

```
To reduce overfitting, the authors used two regularization techniques: first they
applied dropout (introduced in Chapter 11) with a 50% dropout rate during training
to the outputs of layers F8 and F9. Second, they performed data augmentation by ran‐
domly shifting the training images by various offsets, flipping them horizontally, and
changing the lighting conditions.
```

```
Data Augmentation
Data augmentation artificially increases the size of the training set by generating
many realistic variants of each training instance. This reduces overfitting, making this
a regularization technique. The generated instances should be as realistic as possible:
```

```
450 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

```
ideally, given an image from the augmented training set, a human should not be able
to tell whether it was augmented or not. Moreover, simply adding white noise will not
help; the modifications should be learnable (white noise is not).
For example, you can slightly shift, rotate, and resize every picture in the training set
by various amounts and add the resulting pictures to the training set (see
Figure 14-12). This forces the model to be more tolerant to variations in the position,
orientation, and size of the objects in the pictures. If you want the model to be more
tolerant to different lighting conditions, you can similarly generate many images with
various contrasts. In general, you can also flip the pictures horizontally (except for
text, and other non-symmetrical objects). By combining these transformations you
can greatly increase the size of your training set.
```

```
Figure 14-12. Generating new training instances from existing ones
```

AlexNet also uses a competitive normalization step immediately after the ReLU step
of layers C1 and C3, called local response normalization. The most strongly activated
neurons inhibit other neurons located at the same position in neighboring feature
maps (such competitive activation has been observed in biological neurons). This
encourages different feature maps to specialize, pushing them apart and forcing them

```
CNN Architectures | 451
```

12 “Going Deeper with Convolutions,” C. Szegedy et al. (2015).
13 In the 2010 movie Inception, the characters keep going deeper and deeper into multiple layers of dreams,
hence the name of these modules.

```
to explore a wider range of features, ultimately improving generalization. Equation
14-2 shows how to apply LRN.
```

```
Equation 14-2. Local response normalization
```

```
bi=aik+α ∑
j=jlow
```

```
jhigh
aj^2
```

```
−β
with
```

```
jhigh= min i+ 2 r,fn− 1
```

```
jlow= max 0,i− 2 r
```

- bi is the normalized output of the neuron located in feature map i, at some row u
  and column v (note that in this equation we consider only neurons located at this
  row and column, so u and v are not shown).
- ai is the activation of that neuron after the ReLU step, but before normalization.
- k, α, β, and r are hyperparameters. k is called the bias, and r is called the depth
  radius.
- fn is the number of feature maps.

```
For example, if r = 2 and a neuron has a strong activation, it will inhibit the activation
of the neurons located in the feature maps immediately above and below its own.
In AlexNet, the hyperparameters are set as follows: r = 2, α = 0.00002, β = 0.75, and k
= 1. This step can be implemented using the tf.nn.local_response_normaliza
tion() function (which you can wrap in a Lambda layer if you want to use it in a
Keras model).
A variant of AlexNet called ZF Net was developed by Matthew Zeiler and Rob Fergus
and won the 2013 ILSVRC challenge. It is essentially AlexNet with a few tweaked
hyperparameters (number of feature maps, kernel size, stride, etc.).
```

### GoogLeNet

```
The GoogLeNet architecture was developed by Christian Szegedy et al. from Google
Research,^12 and it won the ILSVRC 2014 challenge by pushing the top-5 error rate
below 7%. This great performance came in large part from the fact that the network
was much deeper than previous CNNs (see Figure 14-14). This was made possible by
sub-networks called inception modules,^13 which allow GoogLeNet to use parameters
```

```
452 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

much more efficiently than previous architectures: GoogLeNet actually has 10 times
fewer parameters than AlexNet (roughly 6 million instead of 60 million).

Figure 14-13 shows the architecture of an inception module. The notation “3 × 3 +
1(S)” means that the layer uses a 3 × 3 kernel, stride 1, and SAME padding. The input
signal is first copied and fed to four different layers. All convolutional layers use the
ReLU activation function. Note that the second set of convolutional layers uses differ‐
ent kernel sizes (1 × 1, 3 × 3, and 5 × 5), allowing them to capture patterns at different
scales. Also note that every single layer uses a stride of 1 and SAME padding (even
the max pooling layer), so their outputs all have the same height and width as their
inputs. This makes it possible to concatenate all the outputs along the depth dimen‐
sion in the final depth concat layer (i.e., stack the feature maps from all four top con‐
volutional layers). This concatenation layer can be implemented in TensorFlow using
the tf.concat() operation, with axis=3 (axis 3 is the depth).

Figure 14-13. Inception module

You may wonder why inception modules have convolutional layers with 1 × 1 ker‐
nels. Surely these layers cannot capture any features since they look at only one pixel
at a time? In fact, these layers serve three purposes:

- First, although they cannot capture spatial patterns, they can capture patterns
  along the depth dimension.
- Second, they are configured to output fewer feature maps than their inputs, so
  they serve as bottleneck layers, meaning they reduce dimensionality. This cuts the
  computational cost and the number of parameters, speeding up training and
  improving generalization.
- Lastly, each pair of convolutional layers ([1 × 1, 3 × 3] and [1 × 1, 5 × 5]) acts like
  a single, powerful convolutional layer, capable of capturing more complex pat‐
  terns. Indeed, instead of sweeping a simple linear classifier across the image (as a

```
CNN Architectures | 453
```

```
single convolutional layer does), this pair of convolutional layers sweeps a two-
layer neural network across the image.
```

In short, you can think of the whole inception module as a convolutional layer on
steroids, able to output feature maps that capture complex patterns at various scales.

```
The number of convolutional kernels for each convolutional layer
is a hyperparameter. Unfortunately, this means that you have six
more hyperparameters to tweak for every inception layer you add.
```

Now let’s look at the architecture of the GoogLeNet CNN (see Figure 14-14). The
number of feature maps output by each convolutional layer and each pooling layer is
shown before the kernel size. The architecture is so deep that it has to be represented
in three columns, but GoogLeNet is actually one tall stack, including nine inception
modules (the boxes with the spinning tops). The six numbers in the inception mod‐
ules represent the number of feature maps output by each convolutional layer in the
module (in the same order as in Figure 14-13). Note that all the convolutional layers
use the ReLU activation function.

**454 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

Figure 14-14. GoogLeNet architecture

Let’s go through this network:

- The first two layers divide the image’s height and width by 4 (so its area is divided
  by 16), to reduce the computational load. The first layer uses a large kernel size,
  so that much of the information is still preserved.
- Then the local response normalization layer ensures that the previous layers learn
  a wide variety of features (as discussed earlier).
- Two convolutional layers follow, where the first acts like a bottleneck layer. As
  explained earlier, you can think of this pair as a single smarter convolutional
  layer.
- Again, a local response normalization layer ensures that the previous layers cap‐
  ture a wide variety of patterns.

```
CNN Architectures | 455
```

14 “Very Deep Convolutional Networks for Large-Scale Image Recognition,” K. Simonyan and A. Zisserman
(2015).

- Next a max pooling layer reduces the image height and width by 2, again to speed
  up computations.
- Then comes the tall stack of nine inception modules, interleaved with a couple
  max pooling layers to reduce dimensionality and speed up the net.
- Next, the global average pooling layer simply outputs the mean of each feature
  map: this drops any remaining spatial information, which is fine since there was
  not much spatial information left at that point. Indeed, GoogLeNet input images
  are typically expected to be 224 × 224 pixels, so after 5 max pooling layers, each
  dividing the height and width by 2, the feature maps are down to 7 × 7. More‐
  over, it is a classification task, not localization, so it does not matter where the
  object is. Thanks to the dimensionality reduction brought by this layer, there is
  no need to have several fully connected layers at the top of the CNN (like in
  AlexNet), and this considerably reduces the number of parameters in the net‐
  work and limits the risk of overfitting.
- The last layers are self-explanatory: dropout for regularization, then a fully con‐
  nected layer with 1,000 units, since there are a 1,000 classes, and a softmax acti‐
  vation function to output estimated class probabilities.

```
This diagram is slightly simplified: the original GoogLeNet architecture also included
two auxiliary classifiers plugged on top of the third and sixth inception modules.
They were both composed of one average pooling layer, one convolutional layer, two
fully connected layers, and a softmax activation layer. During training, their loss
(scaled down by 70%) was added to the overall loss. The goal was to fight the vanish‐
ing gradients problem and regularize the network. However, it was later shown that
their effect was relatively minor.
Several variants of the GoogLeNet architecture were later proposed by Google
researchers, including Inception-v3 and Inception-v4, using slightly different incep‐
tion modules, and reaching even better performance.
```

### VGGNet

```
The runner up in the ILSVRC 2014 challenge was VGGNet^14 , developed by K. Simon‐
yan and A. Zisserman. It had a very simple and classical architecture, with 2 or 3 con‐
volutional layers, a pooling layer, then again 2 or 3 convolutional layers, a pooling
layer, and so on (with a total of just 16 convolutional layers), plus a final dense net‐
work with 2 hidden layers and the output layer. It used only 3 × 3 filters, but many
filters.
```

```
456 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

15 “Deep Residual Learning for Image Recognition,” K. He (2015).

### ResNet

```
The ILSVRC 2015 challenge was won using a Residual Network (or ResNet), devel‐
oped by Kaiming He et al.,^15 which delivered an astounding top-5 error rate under
3.6%, using an extremely deep CNN composed of 152 layers. It confirmed the general
trend: models are getting deeper and deeper, with fewer and fewer parameters. The
key to being able to train such a deep network is to use skip connections (also called
shortcut connections): the signal feeding into a layer is also added to the output of a
layer located a bit higher up the stack. Let’s see why this is useful.
When training a neural network, the goal is to make it model a target function h( x ).
If you add the input x to the output of the network (i.e., you add a skip connection),
then the network will be forced to model f( x ) = h( x ) – x rather than h( x ). This is
called residual learning (see Figure 14-15).
```

```
Figure 14-15. Residual learning
```

```
When you initialize a regular neural network, its weights are close to zero, so the net‐
work just outputs values close to zero. If you add a skip connection, the resulting net‐
work just outputs a copy of its inputs; in other words, it initially models the identity
function. If the target function is fairly close to the identity function (which is often
the case), this will speed up training considerably.
Moreover, if you add many skip connections, the network can start making progress
even if several layers have not started learning yet (see Figure 14-16). Thanks to skip
connections, the signal can easily make its way across the whole network. The deep
residual network can be seen as a stack of residual units, where each residual unit is a
small neural network with a skip connection.
```

```
CNN Architectures | 457
```

Figure 14-16. Regular deep neural network (left) and deep residual network (right)

Now let’s look at ResNet’s architecture (see Figure 14-17). It is actually surprisingly
simple. It starts and ends exactly like GoogLeNet (except without a dropout layer),
and in between is just a very deep stack of simple residual units. Each residual unit is
composed of two convolutional layers (and no pooling layer!), with Batch Normaliza‐
tion (BN) and ReLU activation, using 3 × 3 kernels and preserving spatial dimensions
(stride 1, SAME padding).

Figure 14-17. ResNet architecture

Note that the number of feature maps is doubled every few residual units, at the same
time as their height and width are halved (using a convolutional layer with stride 2).
When this happens the inputs cannot be added directly to the outputs of the residual
unit since they don’t have the same shape (for example, this problem affects the skip

**458 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

16 “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,” C. Szegedy et al.
(2016).
17 “Xception: Deep Learning with Depthwise Separable Convolutions,” François Chollet (2016)

```
connection represented by the dashed arrow in Figure 14-17). To solve this problem,
the inputs are passed through a 1 × 1 convolutional layer with stride 2 and the right
number of output feature maps (see Figure 14-18).
```

```
Figure 14-18. Skip connection when changing feature map size and depth
```

```
ResNet-34 is the ResNet with 34 layers (only counting the convolutional layers and
the fully connected layer) containing three residual units that output 64 feature maps,
4 RUs with 128 maps, 6 RUs with 256 maps, and 3 RUs with 512 maps. We will imple‐
ment this architecture later in this chapter.
ResNets deeper than that, such as ResNet-152, use slightly different residual units.
Instead of two 3 × 3 convolutional layers with (say) 256 feature maps, they use three
convolutional layers: first a 1 × 1 convolutional layer with just 64 feature maps (4
times less), which acts as a bottleneck layer (as discussed already), then a 3 × 3 layer
with 64 feature maps, and finally another 1 × 1 convolutional layer with 256 feature
maps (4 times 64) that restores the original depth. ResNet-152 contains three such
RUs that output 256 maps, then 8 RUs with 512 maps, a whopping 36 RUs with 1,024
maps, and finally 3 RUs with 2,048 maps.
```

```
Google’s Inception-v4^16 architecture merged the ideas of GoogLe‐
Net and ResNet and achieved close to 3% top-5 error rate on
ImageNet classification.
```

### Xception

```
Another variant of the GoogLeNet architecture is also worth noting: Xception^17
(which stands for Extreme Inception) was proposed in 2016 by François Chollet (the
```

```
CNN Architectures | 459
```

18 This name can sometimes be ambiguous, since spatially separable convolutions are often called “separable
convolutions” as well.

```
author of Keras), and it significantly outperformed Inception-v3 on a huge vision task
(350 million images and 17,000 classes). Just like Inception-v4, it also merges the
ideas of GoogLeNet and ResNet, but it replaces the inception modules with a special
type of layer called a depthwise separable convolution (or separable convolution for
short^18 ). These layers had been used before in some CNN architectures, but they were
not as central as in the Xception architecture. While a regular convolutional layer
uses filters that try to simultaneously capture spatial patterns (e.g., an oval) and cross-
channel patterns (e.g., mouth + nose + eyes = face), a separable convolutional layer
makes the strong assumption that spatial patterns and cross-channel patterns can be
modeled separately (see Figure 14-19). Thus, it is composed of two parts: the first part
applies a single spatial filter for each input feature map, then the second part looks
exclusively for cross-channel patterns—it is just a regular convolutional layer with 1 ×
1 filters.
```

```
Figure 14-19. Depthwise Separable Convolutional Layer
```

```
Since separable convolutional layers only have one spatial filter per input channel,
you should avoid using them after layers that have too few channels, such as the input
layer (granted, that’s what Figure 14-19 represents, but it is just for illustration pur‐
poses). For this reason, the Xception architecture starts with 2 regular convolutional
layers, but then the rest of the architecture uses only separable convolutions (34 in
```

```
460 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

19 “Crafting GBD-Net for Object Detection,” X. Zeng et al. (2016).
20 “Squeeze-and-Excitation Networks,” Jie Hu et al. (2017)

```
all), plus a few max pooling layers and the usual final layers (a global average pooling
layer, and a dense output layer).
You might wonder why Xception is considered a variant of GoogLeNet, since it con‐
tains no inception module at all? Well, as we discussed earlier, an Inception module
contains convolutional layers with 1 × 1 filters: these look exclusively for cross-
channel patterns. However, the convolution layers that sit on top of them are regular
convolutional layers that look both for spatial and cross-channel patterns. So you can
think of an Inception module as an intermediate between a regular convolutional
layer (which considers spatial patterns and cross-channel patterns jointly) and a sepa‐
rable convolutional layer (which considers them separately). In practice, it seems that
separable convolutions generally perform better.
```

```
Separable convolutions use less parameters, less memory and less
computations than regular convolutional layers, and in general
they even perform better, so you should consider using them by
default (except after layers with few channels).
```

```
The ILSVRC 2016 challenge was won by the CUImage team from the Chinese Uni‐
versity of Hong Kong. They used an ensemble of many different techniques, includ‐
ing a sophisticated object-detection system called GBD-Net^19 , to achieve a top-5 error
rate below 3%. Although this result is unquestionably impressive, the complexity of
the solution contrasted with the simplicity of ResNets. Moreover, one year later
another fairly simple architecture performed even better, as we will see now.
```

### SENet

```
The winning architecture in the ILSVRC 2017 challenge was the Squeeze-and-
Excitation Network (SENet)^20. This architecture extends existing architectures such as
inception networks or ResNets, and boosts their performance. This allowed SENet to
win the competition with an astonishing 2.25% top-5 error rate! The extended ver‐
sions of inception networks and ResNet are called SE-Inception and SE-ResNet respec‐
tively. The boost comes from the fact that a SENet adds a small neural network, called
a SE Block, to every unit in the original architecture (i.e., every inception module or
every residual unit), as shown in Figure 14-20.
```

```
CNN Architectures | 461
```

Figure 14-20. SE-Inception Module (left) and SE-ResNet Unit (right)

A SE Block analyzes the output of the unit it is attached to, focusing exclusively on
the depth dimension (it does not look for any spatial pattern), and it learns which fea‐
tures are usually most active together. It then uses this information to recalibrate the
feature maps, as shown in Figure 14-21. For example, a SE Block may learn that
mouths, noses and eyes usually appear together in pictures: if you see a mouth and a
nose, you should expect to see eyes as well. So if a SE Block sees a strong activation in
the mouth and nose feature maps, but only mild activation in the eye feature map, it
will boost the eye feature map (more accurately, it will reduce irrelevant feature
maps). If the eyes were somewhat confused with something else, this feature map
recalibration will help resolve the ambiguity.

**462 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

Figure 14-21. An SE Block Performs Feature Map Recalibration

A SE Block is composed of just 3 layers: a global average pooling layer, a hidden dense
layer using the ReLU activation function, and a dense output layer using the sigmoid
activation function (see Figure 14-22):

Figure 14-22. SE Block Architecture

```
CNN Architectures | 463
```

As earlier, the global average pooling layer computes the mean activation for each fea‐
ture map: for example, if its input contains 256 feature maps, it will output 256 num‐
bers representing the overall level of response for each filter. The next layer is where
the “squeeze” happens: this layer has much less than 256 neurons, typically 16 times
less than the number of feature maps (e.g., 16 neurons), so the 256 numbers get com‐
pressed into a small vector (e.g., 16 dimensional). This is a low-dimensional vector
representation (i.e., an embedding) of the distribution of feature responses. This bot‐
tleneck step forces the SE Block to learn a general representation of the feature com‐
binations (we will see this principle in action again when we discuss autoencoders
in ???). Finally, the output layer takes the embedding and outputs a recalibration vec‐
tor containing one number per feature map (e.g., 256), each between 0 and 1. The
feature maps are then multiplied by this recalibration vector, so irrelevant features
(with a low recalibration score) get scaled down while relevant features (with a recali‐
bration score close to 1) are left alone.

## Implementing a ResNet-34 CNN Using Keras

Most CNN architectures described so far are fairly straightforward to implement
(although generally you would load a pretrained network instead, as we will see). To
illustrate the process, let’s implement a ResNet-34 from scratch using Keras. First, let’s
create a ResidualUnit layer:

```
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
padding="SAME", use_bias=False)
```

```
class ResidualUnit (keras.layers.Layer):
def __init__ (self, filters, strides=1, activation="relu", **kwargs):
super(). __init__ (**kwargs)
self.activation = keras.activations.get(activation)
self.main_layers = [
DefaultConv2D(filters, strides=strides),
keras.layers.BatchNormalization(),
self.activation,
DefaultConv2D(filters),
keras.layers.BatchNormalization()]
self.skip_layers = []
if strides > 1:
self.skip_layers = [
DefaultConv2D(filters, kernel_size=1, strides=strides),
keras.layers.BatchNormalization()]
```

```
def call(self, inputs):
Z = inputs
for layer in self.main_layers:
Z = layer(Z)
skip_Z = inputs
for layer in self.skip_layers:
```

**464 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

```
skip_Z = layer(skip_Z)
return self.activation(Z + skip_Z)
```

As you can see, this code matches Figure 14-18 pretty closely. In the constructor, we
create all the layers we will need: the main layers are the ones on the right side of the
diagram, and the skip layers are the ones on the left (only needed if the stride is
greater than 1). Then in the call() method, we simply make the inputs go through
the main layers, and the skip layers (if any), then we add both outputs and we apply
the activation function.

Next, we can build the ResNet-34 simply using a Sequential model, since it is really
just a long sequence of layers (we can treat each residual unit as a single layer now
that we have the ResidualUnit class):

```
model = keras.models.Sequential()
model.add(DefaultConv2D(64, kernel_size=7, strides=2,
input_shape=[224, 224, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
strides = 1 if filters == prev_filters else 2
model.add(ResidualUnit(filters, strides=strides))
prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))
```

The only slightly tricky part in this code is the loop that adds the ResidualUnit layers
to the model: as explained earlier, the first 3 RUs have 64 filters, then the next 4 RUs
have 128 filters, and so on. We then set the strides to 1 when the number of filters is
the same as in the previous RU, or else we set it to 2. Then we add the ResidualUnit,
and finally we update prev_filters.

It is quite amazing that in less than 40 lines of code, we can build the model that won
the ILSVRC 2015 challenge! It demonstrates both the elegance of the ResNet model,
and the expressiveness of the Keras API. Implementing the other CNN architectures
is not much harder. However, Keras comes with several of these architectures built in,
so why not use them instead?

## Using Pretrained Models From Keras

In general, you won’t have to implement standard models like GoogLeNet or ResNet
manually, since pretrained networks are readily available with a single line of code, in
the keras.applications package. For example:

```
model = keras.applications.resnet50.ResNet50(weights="imagenet")
```

```
Using Pretrained Models From Keras | 465
```

21 In the ImageNet dataset, each image is associated to a word in the WordNet dataset: the class ID is just a
WordNet ID.

```
That’s all! This will create a ResNet-50 model and download weights pretrained on
the ImageNet dataset. To use it, you first need to ensure that the images have the right
size. A ResNet-50 model expects 224 × 224 images (other models may expect other
sizes, such as 299 × 299), so let’s use TensorFlow’s tf.image.resize() function to
resize the images we loaded earlier:
images_resized = tf.image.resize(images, [224, 224])
```

```
The tf.image.resize() will not preserve the aspect ratio. If this is
a problem, you can try cropping the images to the appropriate
aspect ratio before resizing. Both operations can be done in one
shot with tf.image.crop_and_resize().
```

```
The pretrained models assume that the images are preprocessed in a specific way. In
some cases they may expect the inputs to be scaled from 0 to 1, or -1 to 1, and so on.
Each model provides a preprocess_input() function that you can use to preprocess
your images. These functions assume that the pixel values range from 0 to 255, so we
must multiply them by 255 (since earlier we scaled them to the 0–1 range):
inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)
Now we can use the pretrained model to make predictions:
Y_proba = model.predict(inputs)
As usual, the output Y_proba is a matrix with one row per image and one column per
class (in this case, there are 1,000 classes). If you want to display the top K predic‐
tions, including the class name and the estimated probability of each predicted class,
you can use the decode_predictions() function. For each image, it returns an array
containing the top K predictions, where each prediction is represented as an array
containing the class identifier^21 , its name and the corresponding confidence score:
top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
print ("Image #{}".format(image_index))
for class_id, name, y_proba in top_K[image_index]:
print (" {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
print ()
The output looks like this:
Image #0
n03877845 - palace 42.87%
n02825657 - bell_cote 40.57%
n03781244 - monastery 14.56%
```

```
466 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

```
Image #1
n04522168 - vase 46.83%
n07930864 - cup 7.78%
n11939491 - daisy 4.87%
```

The correct classes (monastery and daisy) appear in the top 3 results for both images.
That’s pretty good considering that the model had to choose among 1,000 classes.

As you can see, it is very easy to create a pretty good image classifier using a pre‐
trained model. Other vision models are available in keras.applications, including
several ResNet variants, GoogLeNet variants like InceptionV3 and Xception,
VGGNet variants, MobileNet and MobileNetV2 (lightweight models for use in
mobile applications), and more.

But what if you want to use an image classifier for classes of images that are not part
of ImageNet? In that case, you may still benefit from the pretrained models to per‐
form transfer learning.

## Pretrained Models for Transfer Learning

If you want to build an image classifier, but you do not have enough training data,
then it is often a good idea to reuse the lower layers of a pretrained model, as we dis‐
cussed in Chapter 11. For example, let’s train a model to classify pictures of flowers,
reusing a pretrained Xception model. First, let’s load the dataset using TensorFlow
Datasets (see Chapter 13):

```
import tensorflow_datasets as tfds
```

```
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples # 3670
class_names = info.features["label"].names # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes # 5
```

Note that you can get information about the dataset by setting with_info=True. Here,
we get the dataset size and the names of the classes. Unfortunately, there is only a
"train" dataset, no test set or validation set, so we need to split the training set. The
TF Datasets project provides an API for this. For example, let’s take the first 10% of
the dataset for testing, the next 15% for validation, and the remaining 75% for train‐
ing:

```
test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])
```

```
test_set = tfds.load("tf_flowers", split=test_split, as_supervised=True)
valid_set = tfds.load("tf_flowers", split=valid_split, as_supervised=True)
train_set = tfds.load("tf_flowers", split=train_split, as_supervised=True)
```

```
Pretrained Models for Transfer Learning | 467
```

Next we must preprocess the images. The CNN expects 224 × 224 images, so we need
to resize them. We also need to run the image through Xception’s prepro
cess_input() function:

```
def preprocess(image, label):
resized_image = tf.image.resize(image, [224, 224])
final_image = keras.applications.xception.preprocess_input(resized_image)
return final_image, label
```

Let’s apply this preprocessing function to all 3 datasets, and let’s also shuffle & repeat
the training set, and add batching & prefetching to all datasets:

```
batch_size = 32
train_set = train_set.shuffle(1000).repeat()
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)
```

If you want to perform some data augmentation, you can just change the preprocess‐
ing function for the training set, adding some random transformations to the training
images. For example, use tf.image.random_crop() to randomly crop the images, use
tf.image.random_flip_left_right() to randomly flip the images horizontally, and
so on (see the notebook for an example).

Next let’s load an Xception model, pretrained on ImageNet. We exclude the top of the
network (by setting include_top=False): this excludes the global average pooling
layer and the dense output layer. We then add our own global average pooling layer,
based on the output of the base model, followed by a dense output layer with 1 unit
per class, using the softmax activation function. Finally, we create the Keras Model:

```
base_model = keras.applications.xception.Xception(weights="imagenet",
include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)
```

As explained in Chapter 11, it’s usually a good idea to freeze the weights of the pre‐
trained layers, at least at the beginning of training:

```
for layer in base_model.layers:
layer.trainable = False
```

```
Since our model uses the base model’s layers directly, rather than
the base_model object itself, setting base_model.trainable=False
would have no effect.
```

Finally, we can compile the model and start training:

**468 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

```
optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
metrics=["accuracy"])
history = model.fit(train_set,
steps_per_epoch=int(0.75 * dataset_size / batch_size),
validation_data=valid_set,
validation_steps=int(0.15 * dataset_size / batch_size),
epochs=5)
```

```
This will be very slow, unless you have a GPU. If you do not, then
you should run this chapter’s notebook in Colab, using a GPU run‐
time (it’s free!). See the instructions at https://github.com/ageron/
handson-ml2.
```

After training the model for a few epochs, its validation accuracy should reach about
75-80%, and stop making much progress. This means that the top layers are now
pretty well trained, so we are ready to unfreeze all layers (or you could try unfreezing
just the top ones), and continue training (don’t forget to compile the model when you
freeze or unfreeze layers). This time we use a much lower learning rate to avoid dam‐
aging the pretrained weights:

```
for layer in base_model.layers:
layer.trainable = True
```

```
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
model.compile(...)
history = model.fit(...)
```

It will take a while, but this model should reach around 95% accuracy on the test set.
With that, you can start training amazing image classifiers! But there’s more to com‐
puter vision than just classification. For example, what if you also want to know where
the flower is in the picture? Let’s look at this now.

## Classification and Localization

Localizing an object in a picture can be expressed as a regression task, as discussed in
Chapter 10: to predict a bounding box around the object, a common approach is to
predict the horizontal and vertical coordinates of the object’s center, as well as its
height and width. This means we have 4 numbers to predict. It does not require much
change to the model, we just need to add a second dense output layer with 4 units
(typically on top of the global average pooling layer), and it can be trained using the
MSE loss:

```
base_model = keras.applications.xception.Xception(weights="imagenet",
include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = keras.layers.Dense(n_classes, activation="softmax")(avg)
```

```
Classification and Localization | 469
```

22 “Crowdsourcing in Computer Vision,” A. Kovashka et al. (2016).

```
loc_output = keras.layers.Dense(4)(avg)
model = keras.models.Model(inputs=base_model.input,
outputs=[class_output, loc_output])
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
loss_weights=[0.8, 0.2], # depends on what you care most about
optimizer=optimizer, metrics=["accuracy"])
But now we have a problem: the flowers dataset does not have bounding boxes
around the flowers. So we need to add them ourselves. This is often one of the hard‐
est and most costly part of a Machine Learning project: getting the labels. It’s a good
idea to spend time looking for the right tools. To annotate images with bounding
boxes, you may want to use an open source image labeling tool like VGG Image
Annotator, LabelImg, OpenLabeler or ImgLab, or perhaps a commercial tool like
LabelBox or Supervisely. You may also want to consider crowdsourcing platforms
such as Amazon Mechanical Turk or CrowdFlower if you have a very large number of
images to annotate. However, it is quite a lot of work to setup a crowdsourcing plat‐
form, prepare the form to be sent to the workers, to supervise them and ensure the
quality of the bounding boxes they produce is good, so make sure it is worth the
effort: if there are just a few thousand images to label, and you don’t plan to do this
frequently, it may be preferable to do it yourself. Adriana Kovashka et al. wrote a very
practical paper^22 about crowdsourcing in Computer Vision, I recommend you check
it out, even if you do not plan to use crowdsourcing.
So let’s suppose you obtained the bounding boxes for every image in the flowers data‐
set (for now we will assume there is a single bounding box per image), you then need
to create a dataset whose items will be batches of preprocessed images along with
their class labels and their bounding boxes. Each item should be a tuple of the form:
(images, (class_labels, bounding_boxes)). Then you are ready to train your
model!
```

```
The bounding boxes should be normalized so that the horizontal
and vertical coordinates, as well as the height and width all range
from 0 to 1. Also, it is common to predict the square root of the
height and width rather than the height and width directly: this
way, a 10 pixel error for a large bounding box will not be penalized
as much as a 10 pixel error for a small bounding box.
```

```
The MSE often works fairly well as a cost function to train the model, but it is not a
great metric to evaluate how well the model can predict bounding boxes. The most
common metric for this is the Intersection over Union (IoU): it is the area of overlap
between the predicted bounding box and the target bounding box, divided by the
```

```
470 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

area of their union (see Figure 14-23). In tf.keras, it is implemented by the
tf.keras.metrics.MeanIoU class.

Figure 14-23. Intersection over Union (IoU) Metric for Bounding Boxes

Classifying and localizing a single object is nice, but what if the images contain multi‐
ple objects (as is often the case in the flowers dataset)?

## Object Detection

The task of classifying and localizing multiple objects in an image is called object
detection. Until a few years ago, a common approach was to take a CNN that was
trained to classify and locate a single object, then slide it across the image, as shown
in Figure 14-24. In this example, the image was chopped into a 6 × 8 grid, and we
show a CNN (the thick black rectangle) sliding across all 3 × 3 regions. When the
CNN was looking at the top left of the image, it detected part of the left-most rose,
and then it detected that same rose again when it was first shifted one step to the
right. At the next step, it started detecting part of the top-most rose, and then it detec‐
ted it again once it was shifted one more step to the right. You would then continue to
slide the CNN through the whole image, looking at all 3 × 3 regions. Moreover, since
objects can have varying sizes, you would also slide the CNN across regions of differ‐
ent sizes. For example, once you are done with the 3 × 3 regions, you might want to
slide the CNN across all 4 × 4 regions as well.

```
Object Detection | 471
```

Figure 14-24. Detecting Multiple Objects by Sliding a CNN Across the Image

This technique is fairly straightforward, but as you can see it will detect the same
object multiple times, at slightly different positions. Some post-processing will then
be needed to get rid of all the unnecessary bounding boxes. A common approach for
this is called non-max suppression:

- First, you need to add an extra objectness output to your CNN, to estimate the
  probability that a flower is indeed present in the image (alternatively, you could
  add a “no-flower” class, but this usually does not work as well). It must use the
  sigmoid activation function and you can train it using the "binary_crossen
  tropy" loss. Then just get rid of all the bounding boxes for which the objectness
  score is below some threshold: this will drop all the bounding boxes that don’t
  actually contain a flower.
- Second, find the bounding box with the highest objectness score, and get rid of
  all the other bounding boxes that overlap a lot with it (e.g., with an IoU greater
  than 60%). For example, in Figure 14-24, the bounding box with the max object‐
  ness score is the thick bounding box over the top-most rose (the objectness score
  is represented by the thickness of the bounding boxes). The other bounding box
  over that same rose overlaps a lot with the max bounding box, so we will get rid
  of it.

**472 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

23 “Fully Convolutional Networks for Semantic Segmentation,” J. Long, E. Shelhamer, T. Darrell (2015).
24 There is one small exception: a convolutional layer using VALID padding will complain if the input size is
smaller than the kernel size.

- Third, repeat step two until there are no more bounding boxes to get rid of.

```
This simple approach to object detection works pretty well, but it requires running
the CNN many times, so it is quite slow. Fortunately, there is a much faster way to
slide a CNN across an image: using a Fully Convolutional Network.
```

### Fully Convolutional Networks (FCNs)

```
The idea of FCNs was first introduced in a 2015 paper^23 by Jonathan Long et al., for
semantic segmentation (the task of classifying every pixel in an image according to
the class of the object it belongs to). They pointed out that you could replace the
dense layers at the top of a CNN by convolutional layers. To understand this, let’s look
at an example: suppose a dense layer with 200 neurons sits on top of a convolutional
layer that outputs 100 feature maps, each of size 7 × 7 (this is the feature map size, not
the kernel size). Each neuron will compute a weighted sum of all 100 × 7 × 7 activa‐
tions from the convolutional layer (plus a bias term). Now let’s see what happens if we
replace the dense layer with a convolution layer using 200 filters, each 7 × 7, and with
VALID padding. This layer will output 200 feature maps, each 1 × 1 (since the kernel
is exactly the size of the input feature maps and we are using VALID padding). In
other words, it will output 200 numbers, just like the dense layer did, and if you look
closely at the computations performed by a convolutional layer, you will notice that
these numbers will be precisely the same as the dense layer produced. The only differ‐
ence is that the dense layer’s output was a tensor of shape [batch size, 200] while the
convolutional layer will output a tensor of shape [batch size, 1, 1, 200].
```

```
To convert a dense layer to a convolutional layer, the number of fil‐
ters in the convolutional layer must be equal to the number of units
in the dense layer, the filter size must be equal to the size of the
input feature maps, and you must use VALID padding. The stride
may be set to 1 or more, as we will see shortly.
```

```
Why is this important? Well, while a dense layer expects a specific input size (since it
has one weight per input feature), a convolutional layer will happily process images of
any size^24 (however, it does expect its inputs to have a specific number of channels,
since each kernel contains a different set of weights for each input channel). Since an
FCN contains only convolutional layers (and pooling layers, which have the same
property), it can be trained and executed on images of any size!
```

```
Object Detection | 473
```

25 This assumes we used only SAME padding in the network: indeed, VALID padding would reduce the size of
the feature maps. Moreover, 448 can be neatly divided by 2 several times until we reach 7, without any round‐
ing error. If any layer uses a different stride than 1 or 2, then there may be some rounding error, so again the
feature maps may end up being smaller.

```
For example, suppose we already trained a CNN for flower classification and localiza‐
tion. It was trained on 224 × 224 images and it outputs 10 numbers: outputs 0 to 4 are
sent through the softmax activation function, and this gives the class probabilities
(one per class); output 5 is sent through the logistic activation function, and this gives
the objectness score; outputs 6 to 9 do not use any activation function, and they rep‐
resent the bounding box’s center coordinates, and its height and width. We can now
convert its dense layers to convolutional layers. In fact, we don’t even need to retrain
it, we can just copy the weights from the dense layers to the convolutional layers!
Alternatively, we could have converted the CNN into an FCN before training.
Now suppose the last convolutional layer before the output layer (also called the bot‐
tleneck layer) outputs 7 × 7 feature maps when the network is fed a 224 × 224 image
(see the left side of Figure 14-25). If we feed the FCN a 448 × 448 image (see the right
side of Figure 14-25), the bottleneck layer will now output 14 × 14 feature maps.^25
Since the dense output layer was replaced by a convolutional layer using 10 filters of
size 7 × 7, VALID padding and stride 1, the output will be composed of 10 features
maps, each of size 8 × 8 (since 14 - 7 + 1 = 8). In other words, the FCN will process
the whole image only once and it will output an 8 × 8 grid where each cell contains 10
numbers (5 class probabilities, 1 objectness score and 4 bounding box coordinates).
It’s exactly like taking the original CNN and sliding it across the image using 8 steps
per row and 8 steps per column: to visualize this, imagine chopping the original
image into a 14 × 14 grid, then sliding a 7 × 7 window across this grid: there will be 8
× 8 = 64 possible locations for the window, hence 8 × 8 predictions. However, the
FCN approach is much more efficient, since the network only looks at the image
once. In fact, You Only Look Once (YOLO) is the name of a very popular object detec‐
tion architecture!
```

```
474 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

26 “You Only Look Once: Unified, Real-Time Object Detection,” J. Redmon, S. Divvala, R. Girshick, A. Farhadi
(2015).
27 “YOLO9000: Better, Faster, Stronger,” J. Redmon, A. Farhadi (2016).
28 “YOLOv3: An Incremental Improvement,” J. Redmon, A. Farhadi (2018).

```
Figure 14-25. A Fully Convolutional Network Processing a Small Image (left) and a
Large One (right)
```

### You Only Look Once (YOLO)

```
YOLO is an extremely fast and accurate object detection architecture proposed by
Joseph Redmon et al. in a 2015 paper^26 , and subsequently improved in 2016^27
(YOLOv2) and in 2018^28 (YOLOv3). It is so fast that it can run in realtime on a video
(check out this nice demo).
YOLOv3’s architecture is quite similar to the one we just discussed, but with a few
important differences:
```

```
Object Detection | 475
```

- First, it outputs 5 bounding boxes for each grid cell (instead of just 1), and each
  bounding box comes with an objectness score. It also outputs 20 class probabili‐
  ties per grid cell, as it was trained on the PASCAL VOC dataset, which contains
  20 classes. That’s a total of 45 numbers per grid cell (5 \* 4 bounding box coordi‐
  nates, plus 5 objectness scores, plus 20 class probabilities).
- Second, instead of predicting the absolute coordinates of the bounding box cen‐
  ters, YOLOv3 predicts an offset relative to the coordinates of the grid cell, where
  (0, 0) means the top left of that cell, and (1, 1) means the bottom right. For each
  grid cell, YOLOv3 is trained to predict only bounding boxes whose center lies in
  that cell (but the bounding box itself generally extends well beyond the grid cell).
  YOLOv3 applies the logistic activation function to the bounding box coordinates
  to ensure they remain in the 0 to 1 range.
- Third, before training the neural net, YOLOv3 finds 5 representative bounding
  box dimensions, called anchor boxes (or bounding box priors): it does this by
  applying the K-Means algorithm (see ???) to the height and width of the training
  set bounding boxes. For example, if the training images contain many pedes‐
  trians, then one of the anchor boxes will likely have the dimensions of a typical
  pedestrian. Then when the neural net predicts 5 bounding boxes per grid cell, it
  actually predicts how much to rescale each of the anchor boxes. For example,
  suppose one anchor box is 100 pixels tall and 50 pixels wide, and the network
  predicts, say, a vertical rescaling factor of 1.5 and a horizontal rescaling of 0.9 (for
  one of the grid cells), this will result in a predicted bounding box of size 150 × 45
  pixels. To be more precise, for each grid cell and each anchor box, the network
  predicts the log of the vertical and horizontal rescaling factors. Having these pri‐
  ors makes the network more likely to predict bounding boxes of the appropriate
  dimensions, and it also speeds up training since it will more quickly learn what
  reasonable bounding boxes look like.
- Fourth, the network is trained using images of different scales: every few batches
  during training, the network randomly chooses a new image dimension (from
  330 × 330 to 608 × 608 pixels). This allows the network to learn to detect objects
  at different scales. Moreover, it makes it possible to use YOLOv3 at different
  scales: the smaller scale will be less accurate but faster than the larger scale, so
  you can choose the right tradeoff for your use case.

There are a few more innovations you might be interested in, such as the use of skip
connections to recover some of the spatial resolution that is lost in the CNN (we will
discuss this shortly when we look at semantic segmentation). Moreover, in the 2016
paper, the authors introduce the YOLO9000 model that uses hierarchical classifica‐
tion: the model predicts a probability for each node in a visual hierarchy called Word‐
Tree. This makes it possible for the network to predict with high confidence that an
image represents, say, a dog, even though it is unsure what specific type of dog it is.

**476 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

So I encourage you to go ahead and read all three papers: they are quite pleasant to
read, and it is an excellent example of how Deep Learning systems can be incremen‐
tally improved.

```
Mean Average Precision (mAP)
A very common metric used in object detection tasks is the mean Average Precision
(mAP). “Mean Average” sounds a bit redundant, doesn’t it? To understand this met‐
ric, let’s go back to two classification metrics we discussed in Chapter 3: precision and
recall. Remember the tradeoff: the higher the recall, the lower the precision. You can
visualize this in a Precision/Recall curve (see Figure 3-5). To summarize this curve
into a single number, we could compute its Area Under the Curve (AUC). But note
that the Precision/Recall curve may contain a few sections where precision actually
goes up when recall increases, especially at low recall values (you can see this at the
top left of Figure 3-5). This is one of the motivations for the mAP metric.
Suppose the classifier has a 90% precision at 10% recall, but a 96% precision at 20%
recall: there’s really no tradeoff here: it simply makes more sense to use the classifier
at 20% recall rather than at 10% recall, as you will get both higher recall and higher
precision. So instead of looking at the precision at 10% recall, we should really be
looking at the maximum precision that the classifier can offer with at least 10% recall.
It would be 96%, not 90%. So one way to get a fair idea of the model’s performance is
to compute the maximum precision you can get with at least 0% recall, then 10%
recall, 20%, and so on up to 100%, and then calculate the mean of these maximum
precisions. This is called the Average Precision (AP) metric. Now when there are more
than 2 classes, we can compute the AP for each class, and then compute the mean AP
(mAP). That’s it!
However, in an object detection systems, there is an additional level of complexity:
what if the system detected the correct class, but at the wrong location (i.e., the
bounding box is completely off )? Surely we should not count this as a positive predic‐
tion. So one approach is to define an IOU threshold: for example, we may consider
that a prediction is correct only if the IOU is greater than, say, 0.5, and the predicted
class is correct. The corresponding mAP is generally noted mAP@0.5 (or mAP@50%,
or sometimes just AP 50 ). In some competitions (such as the Pascal VOC challenge),
this is what is done. In others (such as the COCO competition), the mAP is computed
for different IOU thresholds (0.50, 0.55, 0.60, ..., 0.95), and the final metric is the
mean of all these mAPs (noted AP@[.50:.95] or AP@[.50:0.05:.95]). Yes, that’s a mean
mean average.
```

Several YOLO implementations built using TensorFlow are available on github, some
with pretrained weights. At the time of writing, they are based on TensorFlow 1, but
by the time you read this, TF 2 implementations will certainly be available. Moreover,
other object detection models are available in the TensorFlow Models project, many

```
Object Detection | 477
```

29 “SSD: Single Shot MultiBox Detector,” Wei Liu et al. (2015).
30 “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” Shaoqing Ren et al.
(2015).

```
with pretrained weights, and some have even been ported to TF Hub, making them
extremely easy to use, such as SSD^29 and Faster-RCNN.^30 , which are both quite popu‐
lar. SSD is also a “single shot” detection model, quite similar to YOLO, while Faster R-
CNN is more complex: the image first goes through a CNN, and the output is passed
to a Region Proposal Network (RPN) which proposes bounding boxes that are most
likely to contain an object, and a classifier is run for each bounding box, based on the
cropped output of the CNN.
The choice of detection system depends on many factors: speed, accuracy, available
pretrained models, training time, complexity, etc. The papers contain tables of met‐
rics, but there is quite a lot of variability in the testing environments, and the technol‐
ogies evolve so fast that it is difficulty to make a fair comparison that will be useful for
most people and remain valid for more than a few months.
Great! So we can locate objects by drawing bounding boxes around them. But per‐
haps you might want to be a bit more precise. Let’s see how to go down to the pixel
level.
```

## Semantic Segmentation

```
In semantic segmentation, each pixel is classified according to the class of the object it
belongs to (e.g., road, car, pedestrian, building, etc.), as shown in Figure 14-26. Note
that different objects of the same class are not distinguished. For example, all the bicy‐
cles on the right side of the segmented image end up as one big lump of pixels. The
main difficulty in this task is that when images go through a regular CNN, they grad‐
ually lose their spatial resolution (due to the layers with strides greater than 1): so a
regular CNN may end up knowing that there’s a person in the image, somewhere in
the bottom left of the image, but it will not be much more precise than that.
```

```
478 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

31 This type of layer is sometimes referred to as a deconvolution layer, but it does not perform what mathemati‐
cians call a deconvolution, so this name should be avoided.

```
Figure 14-26. Semantic segmentation
```

```
Just like for object detection, there are many different approaches to tackle this prob‐
lem, some quite complex. However, a fairly simple solution was proposed in the 2015
paper by Jonathan Long et al. we discussed earlier. They start by taking a pretrained
CNN and turning into an FCN, as discussed earlier. The CNN applies a stride of 32 to
the input image overall (i.e., if you add up all the strides greater than 1), meaning the
last layer outputs feature maps that are 32 times smaller than the input image. This is
clearly too coarse, so they add a single upsampling layer that multiplies the resolution
by 32. There are several solutions available for upsampling (increasing the size of an
image), such as bilinear interpolation, but it only works reasonably well up to ×4 or
×8. Instead, they used a transposed convolutional layer:^31 it is equivalent to first
stretching the image by inserting empty rows and columns (full of zeros), then per‐
forming a regular convolution (see Figure 14-27). Alternatively, some people prefer to
think of it as a regular convolutional layer that uses fractional strides (e.g., 1/2 in
Figure 14-27). The transposed convolutional layer can be initialized to perform some‐
thing close to linear interpolation, but since it is a trainable layer, it will learn to do
better during training.
```

```
Semantic Segmentation | 479
```

Figure 14-27. Upsampling Using a Transpose Convolutional Layer

```
In a transposed convolution layer, the stride defines how much the
input will be stretched, not the size of the filter steps, so the larger
the stride, the larger the output (unlike for convolutional layers or
pooling layers).
```

```
TensorFlow Convolution Operations
TensorFlow also offers a few other kinds of convolutional layers:
```

- keras.layers.Conv1D creates a convolutional layer for 1D inputs, such as time
  series or text (sequences of letters or words), as we will see in ???.
- keras.layers.Conv3D creates a convolutional layer for 3D inputs, such as 3D
  PET scan.
- Setting the dilation_rate hyperparameter of any convolutional layer to a value
  of 2 or more creates an à-trous convolutional layer (“à trous” is French for “with
  holes”). This is equivalent to using a regular convolutional layer with a filter dila‐
  ted by inserting rows and columns of zeros (i.e., holes). For example, a 1 × 3 filter
  equal to [[1,2,3]] may be dilated with a dilation rate of 4, resulting in a dilated
  filter [[1, 0, 0, 0, 2, 0, 0, 0, 3]]. This allows the convolutional layer to

**480 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks**

```
have a larger receptive field at no computational price and using no extra param‐
eters.
```

- tf.nn.depthwise_conv2d() can be used to create a depthwise convolutional layer
  (but you need to create the variables yourself ). It applies every filter to every
  individual input channel independently. Thus, if there are fn filters and fn′ input
  channels, then this will output fn × fn′ feature maps.

This solution is okay, but still too imprecise. To do better, the authors added skip con‐
nections from lower layers: for example, they upsampled the output image by a factor
of 2 (instead of 32), and they added the output of a lower layer that had this double
resolution. Then they upsampled the result by a factor of 16, leading to a total upsam‐
pling factor of 32 (see Figure 14-28). This recovered some of the spatial resolution
that was lost in earlier pooling layers. In their best architecture, they used a second
similar skip connection to recover even finer details from an even lower layer: in
short, the output of the original CNN goes through the following extra steps: upscale
×2, add the output of a lower layer (of the appropriate scale), upscale ×2, add the out‐
put of an even lower layer, and finally upscale ×8. It is even possible to scale up
beyond the size of the original image: this can be used to increase the resolution of an
image, which is a technique called super-resolution.

Figure 14-28. Skip layers recover some spatial resolution from lower layers

Once again, many github repositories provide TensorFlow implementations of
semantic segmentation (TensorFlow 1 for now), and you will even find a pretrained
instance segmentation model in the TensorFlow Models project. Instance segmenta‐
tion is similar to semantic segmentation, but instead of merging all objects of the
same class into one big lump, each object is distinguished from the others (e.g., it
identifies each individual bicycle). At the present, they provide multiple implementa‐
tions of the Mask R-CNN architecture, which was proposed in a 2017 paper: it
extends the Faster R-CNN model by additionally producing a pixel-mask for each
bounding box. So not only do you get a bounding box around each object, with a set
of estimated class probabilities, you also get a pixel mask that locates pixels in the
bounding box that belong to the object.

```
Semantic Segmentation | 481
```

32 “Matrix Capsules with EM Routing,” G. Hinton, S. Sabour, N. Frosst (2018).

```
As you can see, the field of Deep Computer Vision is vast and moving fast, with all
sorts of architectures popping out every year, all based on Convolutional Neural Net‐
works. The progress made in just a few years has been astounding, and researchers
are now focusing on harder and harder problems, such as adversarial learning (which
attempts to make the network more resistant to images designed to fool it), explaina‐
bility (understanding why the network makes a specific classification), realistic image
generation (which we will come back to in ???), single-shot learning (a system that can
recognize an object after it has seen it just once), and much more. Some even explore
completely novel architectures, such as Geoffrey Hinton’s capsule networks^32 (I pre‐
sented them in a couple videos, with the corresponding code in a notebook). Now on
to the next chapter, where we will look at how to process sequential data such as time
series using Recurrent Neural Networks and Convolutional Neural Networks.
```

## Exercises

1. What are the advantages of a CNN over a fully connected DNN for image classi‐
   fication?
2. Consider a CNN composed of three convolutional layers, each with 3 × 3 kernels,
   a stride of 2, and SAME padding. The lowest layer outputs 100 feature maps, the
   middle one outputs 200, and the top one outputs 400. The input images are RGB
   images of 200 × 300 pixels. What is the total number of parameters in the CNN?
   If we are using 32-bit floats, at least how much RAM will this network require
   when making a prediction for a single instance? What about when training on a
   mini-batch of 50 images?
3. If your GPU runs out of memory while training a CNN, what are five things you
   could try to solve the problem?
4. Why would you want to add a max pooling layer rather than a convolutional
   layer with the same stride?
5. When would you want to add a local response normalization layer?
6. Can you name the main innovations in AlexNet, compared to LeNet-5? What
   about the main innovations in GoogLeNet, ResNet, SENet and Xception?
7. What is a Fully Convolutional Network? How can you convert a dense layer into
   a convolutional layer?
8. What is the main technical difficulty of semantic segmentation?
9. Build your own CNN from scratch and try to achieve the highest possible accu‐
   racy on MNIST.

```
482 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
```

10. Use transfer learning for large image classification.

```
a.Create a training set containing at least 100 images per class. For example, you
could classify your own pictures based on the location (beach, mountain, city,
etc.), or alternatively you can just use an existing dataset (e.g., from Tensor‐
Flow Datasets).
b. Split it into a training set, a validation set and a test set.
c.Build the input pipeline, including the appropriate preprocessing operations,
and optionally add data augmentation.
d.Fine-tune a pretrained model on this dataset.
```

11. Go through TensorFlow’s DeepDream tutorial. It is a fun way to familiarize your‐
    self with various ways of visualizing the patterns learned by a CNN, and to gener‐
    ate art using Deep Learning.

Solutions to these exercises are available in ???.

```
Exercises | 483
```

##### About the Author

**Aurélien Géron** is a Machine Learning consultant. A former Googler, he led the You‐
Tube video classification team from 2013 to 2016. He was also a founder and CTO of
Wifirst from 2002 to 2012, a leading Wireless ISP in France; and a founder and CTO
of Polyconseil in 2001, the firm that now manages the electric car sharing service
Autolib’.

Before this he worked as an engineer in a variety of domains: finance (JP Morgan and
Société Générale), defense (Canada’s DOD), and healthcare (blood transfusion). He
published a few technical books (on C++, WiFi, and internet architectures), and was
a Computer Science lecturer in a French engineering school.

A few fun facts: he taught his three children to count in binary with their fingers (up
to 1023), he studied microbiology and evolutionary genetics before going into soft‐
ware engineering, and his parachute didn’t open on the second jump.

##### Colophon

The animal on the cover of Hands-On Machine Learning with Scikit-Learn and Ten‐
sorFlow is the fire salamander (Salamandra salamandra), an amphibian found across
most of Europe. Its black, glossy skin features large yellow spots on the head and
back, signaling the presence of alkaloid toxins. This is a possible source of this
amphibian’s common name: contact with these toxins (which they can also spray
short distances) causes convulsions and hyperventilation. Either the painful poisons
or the moistness of the salamander’s skin (or both) led to a misguided belief that these
creatures not only could survive being placed in fire but could extinguish it as well.

Fire salamanders live in shaded forests, hiding in moist crevices and under logs near
the pools or other freshwater bodies that facilitate their breeding. Though they spend
most of their life on land, they give birth to their young in water. They subsist mostly
on a diet of insects, spiders, slugs, and worms. Fire salamanders can grow up to a foot
in length, and in captivity, may live as long as 50 years.

The fire salamander’s numbers have been reduced by destruction of their forest habi‐
tat and capture for the pet trade, but the greatest threat is the susceptibility of their
moisture-permeable skin to pollutants and microbes. Since 2014, they have become
extinct in parts of the Netherlands and Belgium due to an introduced fungus.

Many of the animals on O’Reilly covers are endangered; all of them are important to
the world. To learn more about how you can help, go to animals.oreilly.com.

The cover image is from Wood’s Illustrated Natural History. The cover fonts are URW
Typewriter and Guardian Sans. The text font is Adobe Minion Pro; the heading font
is Adobe Myriad Condensed; and the code font is Dalton Maag’s Ubuntu Mono.
