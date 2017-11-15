# capstone

Looking for bias in news articles.

By reading in sentences from a news article into an RNN can we mathematically map the ideology of the article? The issue of bias in journalism has long been a hot button issue, but it has blown up in the Trump era with vitrolic attacks coming from the very top on political reporting.

People can manually judge bias, but it inherantly involves subjective human assessment. People have also created maps of the biases of different news sources:

https://i.imgur.com/kP4Yax1.png

This process can be automated by mapping the web via twitter shares:

https://thesocietypages.org/socimages/2017/09/28/the-different-media-spheres-of-the-right-and-the-left-and-how-theyre-throwing-elections-to-the-republicans/


My goal is to tackle this by reading the text itself.

My plan is to organize the text around key figures, groups and concepts. E.g. ['Elizabeth Warren', 'Democrats', 'Socialism']

When these entities are mentioned in the text, they can be associated with sentiment in their context. By comparint the sentiment in one article to another article on a similar topic, we can find a sentimental variance. Thus we can try to find and article with no sentiment and articles with extreme sentiment.

By cross referencing with other forms of bias indication, we can associate each article with a conservative or liberal bias. 

Data:

News articles: Fox, HP.

Political ad transcripts.



MVP: Model which can assign similary scores between news articles and political advertisements.

Bonus: Web app which can accept text and return a political metric.




IN PROGRESS THOUGHTS:

So we have:
  Republican ads.
  Democratic ads.

  Republican leaning political news articles
  Democratic leaning political news articles

  
  The simplest thing might be to train on the ad data, building a network that distinguishes between republican and democratic ads, and then use the news articles purely as a test. If the ads themselves can sort the articles, that would be great. 
  
  My concern is that the domain shift between transcripts of spoken advertising and written news articles will be too great. Thus it would help to also train on the news articles. However that creates a plethora of other issues. Once we're training on news articles we have to either be pre-assigning labels to the articles, or be performing some sort of unsupervised learning. If we just ask the network to classify an article as "FOX" or "HP" and then assign a loss based on missed classifications, we could just be learning the writing style of FOX and HP. There's also a leakage nightmare, as articles have clues like, "woman house told FOX news," in their text.
  This also weakens the value of the model, because it's looking for similarity to FOX/HP rather than similarity to actual propoganda. I can live with it, but it's less interesting.
  
	
  A possible solution is to cut articles into sentences and remove sentences with compromising information (Author names, Journal names). We could further limit the data to just sentences with identified Republican/Democratic entities. 
	
	
  The original idea of how to make this work was to not directly ask for party identification, but to rather use advertisement as entity sentiment labels. They revolve around entities they support/attack and have intense negative/positive sentiment. By training for entity sentiment, you can then look for entity sentiment in news articles, perhaps a few sentences before to a few sentences after a tagged entity is mentioned. We then collect all of the tagged entities mentioned and produce a vector of entity-sentiment pairs for the article. 
  We can then concatenate each document vector into a big ol matrix and run SVD. Ideally the first component will be the left/right partisan divide.
	
	


