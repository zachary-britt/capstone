from data_loader import load_toy, load
import pandas as pd
from datetime import datetime as dt

if __name__ == '__main__':
    ads, fox, hp, reu = load()
    #ads, fox, hp, nyt = load_toy()

    ads_df = pd.DataFrame.from_dict(ads)
    fox_df = pd.DataFrame.from_dict(fox)
    hp_df = pd.DataFrame.from_dict(hp)
    reu_df = pd.DataFrame.from_dict(reu)

    ads_df=ads_df[ads_df['supports'].isin(('Hillary Clinton','Donald Trump'))]
    ads_df['source'] = ads_df['supports']
    ads_df.drop('supports', axis=1, inplace=True)

    fox_df['date'] = fox_df.date.apply( lambda date_str: dt.date(dt.strptime(date_str, '%Y-%m-%d')))
    fox_df['source'] = 'fox'

    hp_df.drop('author', axis=1, inplace=True)
    hp_df['date'] = hp_df.date.apply( dt.date )
    hp_df['source'] = 'hp'

    reu_df
