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

News articles: NYT, WSJ.

Political ad transcripts.



MVP: Model which can assign similary scores between news articles and political advertisements.

Bonus: Web app which can accept text and return a political metric.

