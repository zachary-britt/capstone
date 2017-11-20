import pandas as pd
import numpy as np
import data_loader as dl
from multiprocessing import Pool
import ipdb
import re

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
    pool.join()
    return df

def kill_from_(text, keyword, startword):
    i = text.find(keyword)
    if i != -1:
        j = text[:i].rfind(startword)
        return text[:j]
    else:
        return text


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
                    'Share your color commentary:',
                    '\n Click for more'
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]

    kill_from_args = [  ('is a Reporter for Fox', '\n'),
                        ('is a White House Producer for FOX', '\n')
                     ]

    for kill_from_arg in kill_from_args:
        kill_args = [text]
        kill_args.extend(kill_from_arg)
        text = kill_from_(*kill_args)


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
    pool.join()
    return df

def hp_clean_text(text):
    '''
    sign in: 'NEW YORK ― '
    '''

    #ipdb.set_trace()

    aborts = ['(Reuters)']

    for abort in aborts:
        if text.find(abort) != -1:
            return ''

    intro_text = text[:50]

    found = re.search(r'—', intro_text)
    if found:
        text = text[found.span()[1]:]
    # i = intro_text.find(' — ')
    # if i != -1:
    #     text = text[i + len(' — '):]

    sign_ins =  [   'Sign up here.) \n '
                ]

    for sign_in in sign_ins:
        i = text.find(sign_in)
        if i != -1:
            text = text[i+len(sign_in):]


    sign_offs = [   '\n This story has been updated'
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]

    kill_from_args = [  (' is HuffPost’s ', '\n'),
                        ('Have a tip? ', '\n'),
                        ('Check out the full', '\n')
                     ]

    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg[0], kill_from_arg[1]]
        text = kill_from_(*kill_args)


    text = text.replace('HuffPost', 'this newspaper')
    text = text.replace('the Huffington Post', 'this newspaper')

    return text

def hp_clean(df):

    from formatter import hp_clean_text
    # pool = Pool(4)
    # df['content'] = pool.map(hp_clean_text, df['content'])
    df['content'] = list(map(hp_clean_text, df['content']))
    return df

def reu_clean_text(text):
    '''
    sign in: 'WASHINGTON (Reuters) -

    sign out \n Reporting by David Morgan and Amanda Becker; Additional reporting
    by Ginger Gibson, Jeff Mason, Susan Cornwell and Richard Cowan; Editing by Diane
     Craft and Peter Cooney'
    '''

    sign_ins =  [  ' (Reuters) -'
                ]

    for sign_in in sign_ins:
        i = text.find(sign_in)
        if i != -1:
            text = text[i+len(sign_in):]

    sign_offs = [   ' \n Reporting by ',
                    ' \n reporting by ',
                    ' \n Reporting By ',
                    ' \n Additional reporting by',
                    ' \n Additional Reporting by',
                    ' \n Additional Reporting By',
                    ' \n writing by'
                    ' \n Writing by'
                    ' \n Writing By'

                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]


    text = text.replace('Reuters', 'this newspaper')
    return text

def reu_clean(df):
    from formatter import reu_clean_text
    pool = Pool(4)
    df['content'] = pool.map(reu_clean_text, df['content'])
    return df
#
# def nyt_clean(df):
#     return df
#
# def ads_clean(df):
#     return df


def find_leak(df, keyword):
    for content in list(df.content.values):
        i = content.find(keyword)
        section = content[i-100:]
        if i != -1:
            yield section

def cull_shorts(df):
    lens = np.array(df.content.apply(len))
    inds = np.argwhere(lens > 100).ravel()
    return df.iloc[inds]

def loader_formatter():
    dfs = dl.load_dfs()
    dfs = [universal_cleaner(df) for df in dfs]

    fox_df, hp_df, reu_df = dfs

    fox_df = fox_clean(fox_df)
    hp_df = hp_clean(hp_df)
    reu_df = reu_clean(reu_df)

    fox_df = cull_shorts(fox_df)
    hp_df = cull_shorts(hp_df)
    reu_df = cull_shorts(reu_df)

    return fox_df, hp_df, reu_df

if __name__ == '__main__':

    dfs = dl.load_dfs()
    dfs = [universal_cleaner(df) for df in dfs]

    fox_df, hp_df, reu_df = dfs

    fox_df = fox_clean(fox_df)
    hp_df = hp_clean(hp_df)
    reu_df = reu_clean(reu_df)
    # nyt_df = nyt_clean(nyt_df)
    # ads_df = ads_clean(ads_df)
