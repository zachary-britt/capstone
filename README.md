# text2slant

## Looking for bias in news articles via NLP.

By reading in sentences from a news article into an RNN can we mathematically map the ideology of the article? The issue of bias in journalism has long been a hot button issue, but it has blown up in the Trump era with vitrolic attacks coming from the very top on political reporting.

People can manually judge bias, but it inherantly involves subjective human assessment. People have also created maps of the biases of different news sources:

![](https://i.imgur.com/kP4Yax1.png "Partisan map")

Made by Vanessa Otero, allgeneralizationsarefalse.com

My goal is to tackle this by reading the text itself. 



### Web-scraping

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

To further generalize I also downloaded a year of reddit comments and partitioned them into left wing, right wing, and non-political subreddits. The comments were filtered for length and and high score to ensure that they both fit the ethos of their subreddit and are long enough that they make sense out of context.

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


### Pre-processing

The next component of the project was pre-processing the text in a way which removed any obvious "tells" as to the source of the article. The text is loaded from mongo into a pandas dataframe and then stripped of source consistent introductory/ending sentences 

e.g.: 

	"Chris Stirewalt is the politics editor for Fox News. Brianna McClelland contributed to this report. Want FOX News Halftime Report in your inbox every day? Sign up here." 
	
gets cut, along with other references to the news source. (replacing 'Fox news' with 'this newspaper' and so on) 

<br>

After links and strange introductory - conclusion punctuation are stripped, the text is checked to be at least 400 characters long to ensure the model isn't being punished for not understanding a short collection of sentence fragments.

At this stage the reddit comments are similarily also stripped of links, and the comments from political subreddits are filtered for length. See [src/formatter](https://github.com/zachary-britt/text2slant/blob/master/src/formatter.py "formatting")


### spaCy NLP

With the text obfuscated we move on to processing the text in the spaCy NLP ecosystem. Instead of relearning how to read from scratch I used spaCy's [prebuilt model](https://spacy.io/models/en#en_core_web_lg) trained on the [common crawl](http://commoncrawl.org/). The model is useful in translating written text into dense vectors. I.e. spacy reads the text, annotates it (entity tags, noun chunks, part of speech tagging, syntax parsing ...) and uses these annotation to produce a 300 dimensional vector of floats for each word. 

'Flynn is the first member of Trump’s administration to plead guilty.'

is interpreted and transformed into a (13, 300) matrix of floats, one row vector for each word in:

[Flynn, is, the, first, member, of, Trump, ’s, administration, to, plead, guilty, .]


### Model training

With the text embedded into vectors we can now either take those vectors to Keras for neural network training or we can stay in spaCy and use spaCy's neural network model. 

https://spacy.io/usage/training#section-textcat

[src/spacy_textcat](https://github.com/zachary-britt/text2slant/blob/master/src/spacy_textcat.py "textcat")

The spacy_textcat.Model class wraps and handles the training process. The model gets trained via scripts in [src/runner_script](https://github.com/zachary-britt/text2slant/blob/master/src/runner_script) which makes it easy to setup a training sequence which caches the model at each step.

spaCy is finiky about the formatting of the data so I also wrote a long 

### Model Performance 




### Model next steps:

Bin content by date and topic to leverage variance in reporting.  

By scrambling the dates and topics we lose a huge amount of valuable information. Covering Clinton scandals in November 2017 is humongously different than covering a scandal of a frontrunning presidential nominee.

