# text2slant

# Table of Contents
1. [Introduction](#introduction)
2. [Web-scraping](#web_scraping)
3. [Pre-processing](#pre_processing)
4. [spaCy NLP](#spacy)
5. [Model Training](#training)
6. [Model In Action](#performance)
7. [Next Steps](#next_steps)


## Looking for bias in news articles via NLP. <a name="introduction"></a>

By reading in sentences from a news article into an RNN can we mathematically map the ideology of the article? The issue of bias in journalism has long been a hot button issue, but it has blown up in the Trump era with vitrolic attacks coming from the very top on political reporting.

People can manually judge bias, but it inherantly involves subjective human assessment. People have also created maps of the biases of different news sources:

![](https://i.imgur.com/kP4Yax1.png "Partisan map")

Made by Vanessa Otero, allgeneralizationsarefalse.com

My goal is to tackle this by reading the text itself. 



### Web-scraping <a name="web_scraping"></a>

The first major component of the project was data collection.

For training data I used political news articles from:

1. Right wing outlets:
	* Fox 
	* Breitbart
	* Trump election advertisements

2. Left wing outlets
	* the Huffington Post
	* Mother Jones
	* Occupy Democrats
	* Clinton election advertisements

3. Neutral outlets
	* Reuters

I also scraped
* Addicting Info (left)
* Gateway Pundit (right)
* NYT -reporting sections- (center)
* CNN -online articles- (center)

but left them as a holdout set.


These sources required customized web scraping [src/scrapers/](https://github.com/zachary-britt/text2slant/tree/master/src/scrapers "scrapers").

To further generalize I also downloaded a year of reddit comments from [pushshift](https://files.pushshift.io/reddit/comments/)and partitioned them into left wing, right wing, and non-political subreddits. The comments were filtered for length and and high score to ensure that they both fit the ethos of their subreddit and are long enough that they make sense out of context. 

1. Right wing subs:
	* /r/The_Donald 
	* /r/Conservative
	* /r/Republican

2. Left wing subs
	* /r/politics
	* /r/hillaryclinton
	* /r/progressive

3. Neutral (non-political) subs
	* /r/nfl
	* /r/nhl
	* /r/nba
	* /r/baseball
	* /r/soccer
	* /r/hockey
	* /r/Fitness
	* /r/WritingPrompts
	* /r/philosophy
	* /r/askscience
	* /r/gaming


The political comments are then filtered to include at least one recognized political keyword/name, while the non-political subreddits recieve the opposite of this filter. See [src/scrapers/database_cleaning](https://github.com/zachary-britt/text2slant/blob/master/src/scrapers/database_cleaning.py "cleaning")

All of this data is saved to a mongo database for convenient storage.


### Pre-processing <a name="pre_processing"></a>

The next component of the project was pre-processing the text in a way which removed any obvious "tells" as to the source of the article. The text is loaded from mongo into a pandas dataframe and then stripped of source consistent introductory/ending sentences 

e.g.: 

	"Chris Stirewalt is the politics editor for Fox News. Brianna McClelland contributed to this report. Want FOX News Halftime Report in your inbox every day? Sign up here." 
	
gets cut, along with other references to the news source. (replacing 'Fox news' with 'this newspaper' and so on) 

<br>

After links and strange introductory - conclusion punctuation are stripped, the text is checked to be at least 400 characters long to ensure the model isn't being punished for not understanding a short collection of sentence fragments.

At this stage the reddit comments are similarily also stripped of links, and the comments from political subreddits are filtered for length. See [src/formatter](https://github.com/zachary-britt/text2slant/blob/master/src/formatter.py "formatting")


### spaCy NLP <a name="spacy"></a>

With the text obfuscated we move on to processing the text in the spaCy NLP ecosystem. Instead of relearning how to read from scratch I used spaCy's [prebuilt model](https://spacy.io/models/en#en_core_web_lg) trained on the [common crawl](http://commoncrawl.org/). The model is useful in translating written text into dense vectors. I.e. spacy reads the text, annotates it (entity tags, noun chunks, part of speech tagging, syntax parsing ...) and uses these annotation to produce a 300 dimensional vector of floats for each word. 

'Flynn is the first member of Trump’s administration to plead guilty.'

is interpreted and transformed into a (13, 300) matrix of floats, one row vector for each word in:

[Flynn, is, the, first, member, of, Trump, ’s, administration, to, plead, guilty, .]


### Model training <a name="training"></a>

With the text embedded into vectors we can now either take those vectors to Keras for neural network training or we can stay in spaCy and use spaCy's neural network model. Because Keras requires padding each sequence to uniform length, I choose to stay in spaCy as spaCy's textcat is a quick learner (can classify reasonably well after only a few thousand training samples), and flexible (handles abritrarily lengthed sequences, single words - giant documents)

https://spacy.io/usage/training#section-textcat

[src/spacy_textcat](https://github.com/zachary-britt/text2slant/blob/master/src/spacy_textcat.py "textcat")

The spacy_textcat.Model class wraps and handles the training process. The model gets trained via scripts in [src/runner_script](https://github.com/zachary-britt/text2slant/blob/master/src/runner_script) which makes it easy to setup a training sequence which caches the model at each step.

spaCy is finiky about the formatting of the data so I also wrote a large data loading and configuring function in [src/zutils](https://github.com/zachary-britt/text2slant/blob/master/src/zutils.py). 

#### Article Based Training

If the model starts out training on articles it will easily learn to classify articles from each source based on formatting tells. 

e.g. 

1. "U.S. President Donald Trump" == Reuters. 
2. "Mr. Trump" == Fox 
*ect

I uniformized "U.S. President Donald Trump" to just always be "Trump".

You can try to pull out all of these formatting tells in the preprocessing step, but  you'll be playing an endless game of wack-a-mole (which you will lose)

![](https://github.com/zachary-britt/text2slant/blob/master/figures/overtrained_roc.png?raw=true)

#### Reddit Based Training

To help generalize the model, train on the much broader and more diverse Reddit comments. The downside of this is that by training on a related domain you lose the specificity of your model to the target domain and overall performance suffers: 

![](https://github.com/zachary-britt/text2slant/blob/master/figures/article_roc_trained_on_reddit_only.png?raw=true)

#### Reddit Training -> Mixed Training

To re-introduce artcile domain specific classifying I took the Reddit model as a base. The reddit model was trained on 5 epochs of undersampled comments. This was more than enough for the loss to level off. These weights were saved with the suffix .r5 (for 5 reddit epochs)

I then trained the model on mix of reddit comments and articles in a 5-1 ratio of comments to articles. This training was breif, to reduce the potential of the model to shift into memorizing article formatting. 

Each article training epoch was random choice undersampled to ten thousand samples from left, right, and neutral.

I trained the model on 3 epochs of this data (resampled each time) and saved the model after each epoch. This process was performed three times to get a total of 9 models. The final score for an article in the holdout set was the average of the 9 models. This formed an ensamble which reduced model variance at the cost of bias.

![](https://github.com/zachary-britt/text2slant/blob/master/figures/ensamble_roc.png?raw=true)



### Model In Action <a name="performance"></a>

On short text segments the model shows how predominantly negative press is. 

1. “Trump did something”		left: 0.779, 	right: 0.145
2. “Obama did something”		left: 0.120, 	right: 0.820
3. “Republicans did something”		left: 0.989, 	right: 0.008
4. “Democrats did something”		left: 0.126, 	right: 0.822

Right wing press predominantly reports on the actions of Democrats while left wing press predominantly reports on the actions of Republicans. Simply writing "Trump is good" doesn't affect this result, likely because no one writes like that. 

Running the model on single words shows partisan interest in specific topics:

1. “Islam”				left: 0.000, 	right: 0.999
2. “Evangelical”			left:  0.997, 	right: 0.002
3. “Immigration”			left: 0.002, 	right: 0.994
4. “Net neutrality”			left: 0.981, 	right: 0.018

Which matches the (eyes) test.

### Next steps: <a name="next_steps"></a>

Bin content by date and topic to leverage variance in reporting.  

By scrambling the dates and topics we lose a huge amount of valuable information. Covering Clinton scandals in November 2017 is humongously different than covering a scandal of a frontrunning presidential nominee.

