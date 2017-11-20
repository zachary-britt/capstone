import pandas as pd
import data_loader as dl
from multiprocessing import Pool

def universal_text_cleaner(text):

    replacements =  \
        {   '\n': ' \n ',    # space around newlines
            '\xa0':' ',     # space char?
            '\'': '’',      # universal apostrophe
            "\\'": '’',
            "'": '’',
            '`': '’',
            ' "': " “",     # open quote
            '" ': "” "      # close quote
        }

    for k in replacements:
        text = text.replace(k, replacements[k])

    return text


def universal_cleaner(df):
    from formatter import universal_text_cleaner
    pool = Pool(4)
    df['content'] = pool.map(universal_text_cleaner,  df['content'])
    pool.close()
    return df


def fox_clean_text(text):
    '''
    sign in: article summary (sometimes)
            FILE (sometimes)

    sign out (sometimes): \n Chris Stirewalt is the politics editor for Fox News.
    Brianna McClelland contributed to this report. Want FOX News Halftime Report
    in your inbox every day? Sign up here. \n Chris Stirewalt joined Fox News Channel
    (FNC) in July of 2010 and serves as politics editor based in Washington, D.C. \n '
    '''

    sign_ins =  [   ' \n                                      \n                                  \n ',
                    '’Fox & Friends First.’',
                    '’Special Report.’ \n'

                ]

    for sign_in in sign_ins:
        i = text.find(sign_in)
        if i != -1:
            text = text[i+len(sign_in):]

    sign_offs = [   ' \n Jake Gibson is a producer',
                    ' \n  Chris Stirewalt',
                    ' \n Chris Stirewalt',
                    ' \n Follow Fox News’',
                    ' \n Fox News’',
                    ' \n The Associated Press contributed',
                    ' Want FOX News',
                    ' \n Brooke Singman is a Politics Reporter ',
                    ' \n Howard Kurtz ',
                    ' \n  News’ Chad Pergram',
                    'Share your color commentary:'
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]

    text = text.replace('Fox News', 'this newspaper')
    text = text.replace('Fox', 'this newspaper')
    text = text.replace('FNC', 'this newspaper')
    text = text.replace('\n \t\t\t\t\t\t\t\t \n \t\t\t\t\t         \n', '\n')
    text = text.replace('(Copyright 2017 The Associated Press. All rights reserved.)', '')
    text = text.replace('(Copyright 2016 The Associated Press. All rights reserved.)', '')

    return text

def fox_clean(df):
    from formatter import fox_clean_text
    pool = Pool(4)
    df['content'] = pool.map(fox_clean_text, df['content'])
    pool.close()
    return df

def hp_clean_text(text):
    text.replace('HuffPost', 'this newspaper')
    text.replace('the Huffington Post', 'this newspaper')
    return text

def hp_clean(df):
    '''
    sign in: 'NEW YORK
    '''
    from formatter import hp_clean_text
    pool = Pool(6)
    df['content'] = pool.map(hp_clean_text, df['content'])
    return df

def reu_clean(df):
    return df

def nyt_clean(df):
    return df

def ads_clean(df):
    return df


def find_leak(df, keyword):
    for content in list(df.content.values):
        try:
            i = content.find(keyword)
            section = content[i-100:]
            yield section
        except:
            continue

if __name__ == '__main__':
    dfs = dl.load_dfs()
    dfs = [universal_cleaner(df) for df in dfs]

    fox_df, hp_df, reu_df, nyt_df, ads_df = dfs

    fox_df = fox_clean(fox_df)
    hp_df = hp_clean(hp_df)
    reu_df = reu_clean(reu_df)
    nyt_df = nyt_clean(nyt_df)
    ads_df = ads_clean(ads_df)
