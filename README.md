# capstone

Looking for bias in news articles.

By reading in sentences from a news article into an RNN can we mathematically map the ideology of the article? The issue of bias in journalism has long been a hot button issue, but it has blown up in the Trump era with vitrolic attacks coming from the very top on political reporting.

People can manually judge bias, but it inherantly involves subjective human assessment. People have also created maps of the biases of different news sources:

https://i.imgur.com/kP4Yax1.png

This process can be automated by mapping the web via twitter shares:

https://thesocietypages.org/socimages/2017/09/28/the-different-media-spheres-of-the-right-and-the-left-and-how-theyre-throwing-elections-to-the-republicans/


My goal is to tackle this by reading the text itself. For training data I used political news articles from Fox, the Huffington Post (HP), and Reuters over the past year. These were web scraped (see "src/scrapers/"). HP/Fox were chosen as they provide news coverage in a consistently partisan fashion for the left/right. While their journalistic styles differ, they conveniently mirror each other as partisan cheerleaders. As a neutral label I chose Reuters as they embody intensly unsentimental and unbiased news reporting.

As a quick example, right now (2017-11-21) HP, Fox and Reuters each have a headline on net neutrality:

	HP: In Major Win For Telecom Industry, FCC Announces Plans To Repeal Net Neutrality

	Fox: FCC chairman moves to dismantle Obama net neutrality rules

	Reu: FCC chief plans to ditch U.S. 'net neutrality' rules

<br>

First the text is pre-processed and obfuscated  by removing sentences which clearly identify the source of the article. 

e.g.: 

	"Chris Stirewalt is the politics editor for Fox News. Brianna McClelland contributed to this report. Want FOX News Halftime Report in your inbox every day? Sign up here." 
	
gets cut, along with other references to the news source. (replacing 'HuffPost' with 'this newspaper' and so on)

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

The word vectorizer drops words which show up less than 5 times, so words which aren't in the vectorizer vocabulary are assigned 'RARE|POS' (with POS being the part of speech label). It's important that the model be able to handle RARE words effectively as new news articles on breaking topics are likely to contain new words and phrases. To help handle this: (aside) While we can't train the inital model on the words in the test set, Gensim lets us continue to add more words and update the model after it has already been trained. Thus after recieving a new test set article we can train the embedder on the new text, update the model, and then embed the text into vectors. This sounds like leakage, but is kosher as the embedder is an unsupervised model and totally indifferent to the labels.

<br>

 
With the text in each article transformed into a sequence of dense vectors, we can now run an LSTM RNN over the sequences and train the network with softmax classification. For bias classification labels we just use the source of the article. HP being a proxy for left, Reuters a proxy for neutral, and Fox a proxy for right. While this asks the network to simply identify the source of the article, the goal is to be able to classify articles from arbitrary sources along the spectrum of left to right. Thus it's important that the model train on the political sentiment in the text, and not on the editorial style/formatting quirks of each source. 

A future goal is to incorporate politically charged, >paragraph length comments from reddit. Reddit is a highly rich source as A) comments are uniformly formatted, B) the Reddit hivemind produces upvoted comments which represent the collective views of a subreddit, C) the writing styles and diction are boundlessly varied, which should enormously increase the predictive strength of the model.
