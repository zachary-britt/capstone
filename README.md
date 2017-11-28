# text2slant

## Looking for bias in news articles via NLP.

By reading in sentences from a news article into an RNN can we mathematically map the ideology of the article? The issue of bias in journalism has long been a hot button issue, but it has blown up in the Trump era with vitrolic attacks coming from the very top on political reporting.

People can manually judge bias, but it inherantly involves subjective human assessment. People have also created maps of the biases of different news sources:

![](https://i.imgur.com/kP4Yax1.png "Partisan map")

This process can be automated by mapping the web via twitter shares:

![](https://thesocietypages.org/socimages/files/2017/09/4.png "Twitter Partisan map")


<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">We should have a contest as to which of the Networks, plus CNN and not including Fox, is the most dishonest, corrupt and/or distorted in its political coverage of your favorite President (me). They are all bad. Winner to receive the FAKE NEWS TROPHY!</p>&mdash; Donald J. Trump (@realDonaldTrump) <a href="https://twitter.com/realDonaldTrump/status/935147410472480769?ref_src=twsrc%5Etfw">November 27, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


![Even Donald Trump cares](https://github.com/zachary-britt/text2slant/blob/master/figures/Screenshot%20from%202017-11-28%2013-00-44.png "Trump Cares")

My goal is to tackle this by reading the text itself. 




### Web-scraping

The first major component of the project was data collection.

For training data I used political news articles from:

1. Right wing outlets:
	* Fox (fox) 
	* Breitbart (bb)
	* Trump election advertisements (ads)

2. Left wing outlets
	* the Huffington Post (hp)
	* Mother Jones (mj)
	* Occupy Democrats (od)
	* Clinton election advertisements (ads)

3. Neutral outlets
	* Reuters (reu)

I also scraped
* Addicting Info (ai) (left)
and
* Gateway Pundit (gp) (right)

but left them as a holdout set.


These were tediously web scraped. See [src/scrapers/](https://github.com/zachary-britt/text2slant/tree/master/src/scrapers "scrapers"). HP, Fox and reuters were the original dataset, but they provide too little variation in their writing style. This allows a model to easily identify what "an HP" article looks like, without having to learn anything about political sentiment.

By expanding the dataset with a distinct sources, the model can be leveraged into memorizing less and learning more. 

To further generalize I also downloaded a year of reddit comments and partitioned them into left wing, right wing, and non-political subreddits. The comments were filtered for length and and high score to ensure that they both fit the ethos of their subreddit and are long enough that they make sense out of context. 

The political comments are then filtered to include at least one recognized political keyword/name, while the non-political subreddits recieve the opposite of this filter. See [src/scrapers/database_cleaning](https://github.com/zachary-britt/text2slant/blob/master/src/scrapers/database_cleaning.py "cleaning")

All of this data is saved to a mongo database for convenient storage.

### Pre-processing

The next component of the project was pre-processing the text in a way which removed any obvious "tells" as to the source of the article. The text is loaded from mongo into a pandas dataframe and then stripped of source consistent introductory/ending sentences 

e.g.: 

	"Chris Stirewalt is the politics editor for Fox News. Brianna McClelland contributed to this report. Want FOX News Halftime Report in your inbox every day? Sign up here." 
	
gets cut, along with other references to the news source. (replacing 'Fox news' with 'this newspaper' and so on)

<br>

After links and strange introductory - conclusion punctuation are stripped, the text is checked to be at least 400 characters long to ensure the model isn't being punished for not understanding a short collection of sentence fragments.

At this stage the reddit comments are similarily also stripped of links, and the comments from political subreddits are filtered 

### spaCy NLP

With the text obfuscated we move on to processing the text in the spaCy NLP ecosystem. Instead of relearning how to read from scratch

<br>

### Model training

<br>

### Model Performance on validation set

<br>

### Model Performance on articles from new sources

<br>

### Model next steps:

Bin content by date and topic to leverage variance in reporting.  

By scrambling the dates and topics we lose a huge amount of valuable information. Covering Clinton scandals in November 2017 is humongously different than covering a scandal of a frontrunning presidential nominee.

<br>

# Obsolete from here on


As a quick example, right now (2017-11-21) HP, Fox and Reuters each have a headline on net neutrality:

	HP: In Major Win For Telecom Industry, FCC Announces Plans To Repeal Net Neutrality

	Fox: FCC chairman moves to dismantle Obama net neutrality rules

	Reu: FCC chief plans to ditch U.S. 'net neutrality' rules


<br>


Next the text is chunked and annotated with scpaCy.

this turns:

	'U.S. President Donald Trump will meet with Russian President Vladimir Putin next week at a summit in Germany.'
	
into:

	'U.S._President_Donald_Trump|ENT will|VERB meet|VERB with|ADP Russian_President_Vladimir_Putin|ENT next_week|DATE at|ADP a|DET summit|NOUN in|ADP Germany|ENT .|PUNCT'

Which is substantially easier to work with.

(At this point we split the dataset into train/test sets to prevent leakage)

<br>

These annotated words and chunks are then vectorized into dense array of 100 or so floats in a process called "embedding" carried out by Gensim Word2Vec. 

(Key hyperparameters: Dimensionality of word vectors, skip gram vs CBOW, negative sampling vs heirarchical softmax, window size) 

https://radimrehurek.com/gensim/models/word2vec.html

The word vectorizer drops words which show up less than 5 times, so words which aren't in the vectorizer vocabulary are assigned 'RARE|POS' (with POS being the part of speech label). It's important that the model be able to handle RARE words effectively as new news articles on breaking topics are likely to contain new words and phrases. To help handle this: (aside) While we can't initially train the word embedder on the words in the test set, Gensim lets us continue to add more words and update the model after it has already been trained. Thus after recieving a new test set article we can train the embedder on the new text, update the model, and then embed the text into vectors. This sounds like leakage, but is kosher as the embedder is an unsupervised model and totally indifferent to the labels.

<br>

i.e.:

	1) All words preprocessed and annotated.
	2) Train test split
	3) word2vec trained on training set
	4) training set vectorized
	5a) RNN cross validation training via training set
	5b) Search for optimal hyper-parameters 
	
	(model parameters frozen)

	6) for test_article in test_set:
		word2vec updated by adding test_article text
		test_article vectorized
		test_article_vector classified via RNN
	
	7) Produce confusion matrix and performance metrics
	
	8) Repeat model training pipeline, but with entire corpus in training set
	9) Ask model to predict bias for Salon/Brietart to see if model generalizes to articles from new sources

<br>


With the text in each article transformed into a sequence of dense vectors, we can now run an LSTM RNN over the sequences and train the network with softmax classification. For bias classification labels we just use the source of the article. HP being a proxy for left, Reuters a proxy for neutral, and Fox a proxy for right. While this asks the network to simply identify the source of the article, the goal is to be able to classify articles from arbitrary sources along the spectrum of left to right. Thus it's important that the model train on the political sentiment in the text, and not on the editorial style/formatting quirks of each source. 

A future goal is to incorporate politically charged, >paragraph length comments from reddit. Reddit is a highly rich source as A) comments are uniformly formatted, B) the Reddit hivemind produces upvoted comments which represent the collective views of a subreddit, C) the writing styles and diction are boundlessly varied, which should enormously increase the predictive strength of the model.
