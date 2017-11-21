# capstone

Looking for bias in news articles.

By reading in sentences from a news article into an RNN can we mathematically map the ideology of the article? The issue of bias in journalism has long been a hot button issue, but it has blown up in the Trump era with vitrolic attacks coming from the very top on political reporting.

People can manually judge bias, but it inherantly involves subjective human assessment. People have also created maps of the biases of different news sources:

https://i.imgur.com/kP4Yax1.png

This process can be automated by mapping the web via twitter shares:

https://thesocietypages.org/socimages/2017/09/28/the-different-media-spheres-of-the-right-and-the-left-and-how-theyre-throwing-elections-to-the-republicans/


My goal is to tackle this by reading the text itself.

First the text is pre-processed and obfuscated  by removing sentences which clearly identify the source of the article. 

e.g.: 

	"Chris Stirewalt is the politics editor for Fox News. Brianna McClelland contributed to this report. Want FOX News Halftime Report in your inbox every day? Sign up here." 
	
gets cut, along with other references to the news source. (replacing 'HuffPost' with 'this newspaper' and so on)


Next the text is chunked and annotated with scpaCy.

this turns:

	U.S. President Donald Trump will meet with Russian President Vladimir Putin next week at a summit in Germany that brings two world leaders whose political fortunes have become intertwined face-to-face for the first time. 
	
into:

	U.S._President_Donald_Trump|ENT will|VERB meet|VERB with|ADP Russian_President_Vladimir_Putin|ENT next_week|DATE at|ADP a|DET summit|NOUN in|ADP Germany|ENT that|ADJ brings|VERB two|CARDINAL world_leaders|NOUN whose|ADJ political_fortunes|NOUN have|VERB become|VERB intertwined|ADJ face|NOUN -|PUNCT to|ADP -|PUNCT face|NOUN for|ADP the|DET first_time|NOUN .|PUNCT'


My plan is to organize the text around key figures, groups and concepts. E.g. ['Elizabeth Warren', 'Democrats', 'Socialism']

When these entities are mentioned in the text, they can be associated with sentiment in their context. By comparint the sentiment in one article to another article on a similar topic, we can find a sentimental variance. Thus we can try to find and article with no sentiment and articles with extreme sentiment.

By cross referencing with other forms of bias indication, we can associate each article with a conservative or liberal bias. 

Data:

News articles: Fox, HP.

Political ad transcripts.



MVP: Model which can assign similary scores between news articles and political advertisements.

Bonus: Web app which can accept text and return a political metric.




