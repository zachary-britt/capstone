import pandas as pd
import numpy as np
import data_loader as dl
from multiprocessing import Pool
import ipdb
import re
import string
import plac
import os
DATA_PATH = os.environ['DATA_PATH']


'''
TODO

replace dataframe specific functions with a general function that takes:
df:     (w/ 'source' column)

    create: {source: {sign_ins: [], kill_froms: [], sign_outs: [], replacements: []}}
    e.g.:
    {
        'fox':
            {
                'sign_ins': [
                                    '’Fox & Friends First.’',
                                    '’Special Report.’ \n',
                                    ' AP, File)',
                                    ' has the details. \n'
                            ],

                'kill_froms': [
                                    'is a Reporter for Fox',
                                    'is a White House Producer for FOX'
                            ],

                'sign_outs': [
                                    'Share your color commentary:',
                                    '\n Click for more'
                                    '\n Editor’s Note:'
                            ],

                'replacements': [
                                    ('Fox News', 'this newspaper'),
                                    ('FNC', 'this newspaper')
                            ]
            }
    }

It won't be THAT much shorter, but it will be a little cleaner
'''


def universal_text_cleaner(text):

    replacements =  \
        {   '\n': ' \n ',    # space around newlines
            '\xa0':' ',     # space char?
            '\'': '’',      # universal apostrophe
            "\\'": '’',
            "'": '’',
            '`': '’',
            ' "': " “",     # open quote
            '" ': "” ",      # close quote
            '&gt;':'' ,     # using > in stories is a 4chan thing
            '&lt;':'',
            """â""":'’',
            """â\\x80\\x93""":'’',
            '''â\x80\x99''':'’',
            """â""":'“',
            """â\\x80\\x9c'""":'“',
            """â""":'”',
            '''â\\x80\\x9d''':'”',

            # Trump Aliases

            'U.S. President Donald Trump': 'Trump',
            'U.S. President Elect Donald Trump': 'Trump',
            'U.S. President elect Donald Trump': 'Trump',
            'President Donald Trump': 'Trump',
            'President Elect Donald Trump': 'Trump',
            'President elect Donald Trump': 'Trump',
            'President Trump':'Trump',
            'President Elect Trump': 'Trump',
            'President elect Trump': 'Trump',
            'Mr Trump':'Trump',
            'Mr. Trump':'Trump',
            'Donald Trump':'Trump',
            'Donald':'Trump',

            'EXCLUSIVE':'',
            'LISTEN':'',
            'VIDEO':'',
            ' - Reuters interview':'',
            'Reuters interview' :'',
            'WATCH':'',
            'BREAKING':'',
            'REPORT':'',

            # Maybe do the same for senators, representatives, Clinton.........

        }

    for k in replacements:
        text = text.replace(k, replacements[k])

    #strip links
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)

    return text


def universal_cleaner(df):
    from formatter import universal_text_cleaner
    pool = Pool(4)
    df['content'] = pool.map(universal_text_cleaner,  df['content'])

    if 'title' in df.columns:
        df['title'] = pool.map(universal_text_cleaner,  df['title'])

    pool.close()
    pool.join()
    return df


def universal_text_stripper(text):
    start_afters = [' - ', ' ― ', 'EXCLUSIVE:', 'WASHINGTON', 'Raw video: ']
    intro = text[:20]
    for start_after in start_afters:
        i = intro.find(start_after)
        if i != -1:
            text = text[i+len(start_after):]
            intro = intro[i+len(start_after):]

    #consistent representative abbreviations
    # state_abbers = ['AL','AK','AZ','AR','CA',
    #  'CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
    #  'MA','MI','MN','MS','MO','MT', 'NE', 'NV', 'NH','NJ','NM','NY','NC','ND','OH',
    #  'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','WV','WA','VA','WI','WY']
    #
    # for ab in state_abbers:
    #     text.lower().find_all(ab)

    text = text.replace('\t',' ')
    text = text.replace(' – ', ', ')
    text = text.replace(' - ', ', ')
    s = ' \n                           \n                       \n '
    i = text.find(s)
    if i != -1:
        text = text[i+len(s):]


    strip_chars = ' \n,-―:'
    text = text.strip(strip_chars)

    start_afters = ['WASHINGTON, ', '(AP),  ', '(REUTERS)']
    intro = text[:20]
    for start_after in start_afters:
        i = intro.find(start_after)
        if i != -1:
            text = text[i+len(start_after):]
            intro = intro[i+len(start_after):]



    # remove double_new_lines
    lines = text.split('\n')
    lines = [line.strip(' ') for line in lines]
    lines = [line for line in lines if line != '']
    text = '\n'.join(lines)

    return text

def universal_stripper(df):
    from formatter import universal_text_stripper
    pool = Pool(4)
    df['content'] = pool.map(universal_text_stripper, df['content'])

    if 'title' in df.columns:
        df['title'] = pool.map(universal_text_stripper,  df['title'])

    pool.close()
    pool.join()
    return df


def kill_from_line(text, keyword):
    ''' Look for keyword in text, if so, reverse search from keyword to newline,
        end the text before the newline'''

    i = text.find(keyword)
    if i != -1:
        j = text[:i].rfind('\n')
        if j!=-1:
            return text[:j]
        else:
            return ''
    else:
        return text


def fox_clean_text(text):
    ''' Clean Fox Text '''


    sign_ins =  [   ' \n                                      \n                                  \n ',
                    ' \n                           \n                       \n ',
                    ' \\n                           \\n                       \\n',
                    '\n                           \n                       \n'
                    '’Fox & Friends First.’',
                    '’Special Report.’ \n',
                    ' AP, File)',
                    ' has the details. \n'
                ]

    for sign_in in sign_ins:
        i = text.find(sign_in)
        if i != -1:
            text = text[i+len(sign_in):]

    sign_offs = [   ' \n Jake Gibson is a producer',
                    ' \n  Chris Stirewalt',
                    ' \n Chris Stirewalt',
                    ' \n Follow F',
                    ' \n Fox News’',
                    ' \n The Associated Press contributed',
                    'Want FOX News',
                    ' \n Brooke Singman is a Politics Reporter ',
                    ' \n Howard Kurtz ',
                    ' \n  News’ Chad Pergram',
                    'Share your color commentary:',
                    '\n Click for more'
                    '\n Editor’s Note:'
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]

    kill_from_args = [  'is a Reporter for F',
                        'is a reporter for F',
                        'is a White House Producer for FOX'
                     ]
    #ipdb.set_trace()
    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg]
        text = kill_from_line(*kill_args)


    text = text.replace('Fox News', 'this newspaper')
    text = text.replace('Fox', 'this newspaper')
    text = text.replace('FNC', 'this newspaper')
    text = text.replace('\n \t\t\t\t\t\t\t\t \n \t\t\t\t\t         \n', '\n')
    text = text.replace('(Copyright 2017 The Associated Press. All rights reserved.)', '')
    text = text.replace('(Copyright 2016 The Associated Press. All rights reserved.)', '')

    return text


def fox_clean(df):
    ''' clean Fox Dataframe '''
    from formatter import fox_clean_text
    pool = Pool(4)
    df['content'] = pool.map(fox_clean_text, df['content'])
    pool.close()
    pool.join()
    return df


def hp_clean_text(text):
    ''' clean Huffington Post Text '''
    aborts = [  '(Reuters)',
                'HUFFPOST HILL',
                'The Morning Email']

    for abort in aborts:
        if text.find(abort) != -1:
            return ''


    sign_ins =  [   'Sign up here.) \n '
                    'WASHINGTON ―'
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

    kill_from_args = [  ' is HuffPost’s ',
                        'Have a tip? ',
                        'Check out the full'
                     ]

    for kill_from_arg in kill_from_args:
        kill_args = (text, kill_from_arg)
        text = kill_from_line(*kill_args)


    text = text.replace('HuffPost', 'this newspaper')
    text = text.replace('the Huffington Post', 'this newspaper')

    return text


def hp_clean(df):
    ''' clean Huffington Post Dataframe '''
    from formatter import hp_clean_text
    # pool = Pool(4)
    # df['content'] = pool.map(hp_clean_text, df['content'])
    df['content'] = list(map(hp_clean_text, df['content']))
    return df



def reu_clean_text(text):
    ''' clean Reuters text '''

    sign_ins =  [  ' (Reuters) -',
                    'WASHINGTON'
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
    ''' clean Reuters dataframe '''
    from formatter import reu_clean_text
    pool = Pool(4)
    df['content'] = pool.map(reu_clean_text, df['content'])
    return df




def bb_clean_text(text):
    ''' clean Breitbart text '''
    sign_ins =  [  ' (Reuters) -',
                    'WASHINGTON'
                ]
    for sign_in in sign_ins:
        i = text.find(sign_in)
        if i != -1:
            text = text[i+len(sign_in):]



    kill_from_args = [  'Follow him on Twitter ',
                        'is a reporter for Breitbart News',
                        'Check out the full',
                        'is a Breitbart News',
                        'Daily airs on ',
                        'columnist for Br'
                     ]

    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg]
        text = kill_from_line(*kill_args)


    sign_offs = [   '\n Follow ', '\n Follow ',
                    '\n iFrameResize(',
                    '\n You can follow', '\n You can follow',
                    '\n (h/t'
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]

    text = text.replace('Breitbart News', 'this newspaper')
    text = text.replace('Breitbart', 'this newspaper')
    return text

def bb_clean(df):
    ''' clean Breitbart dataframe '''
    from formatter import bb_clean_text
    pool = Pool(4)
    df['content'] = pool.map(bb_clean_text, df['content'])
    return df



def mj_clean_text(text):
    ''' clean motherjones text '''
    sign_ins =  [
                ]
    for sign_in in sign_ins:
        i = text.find(sign_in)
        if i != -1:
            text = text[i+len(sign_in):]

    kill_from_args = [  'Follow him on Twitter ',
                        'Sign up for '
                     ]

    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg]
        text = kill_from_line(*kill_args)


    sign_offs = [   '\n Read the full ',
                    '\n This is a developing story'
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]

    text = text.replace('Mother Jones', 'this newspaper')
    text = text.replace('MJ', 'this newspaper')
    text = text.replace('&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;lt;span data-mce-type=”bookmark” style=”display: inline-block; width: 0px; overflow: hidden; line-height: 0;” class=”mce_SELRES_start”&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;\ufeff&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;lt;/span&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;','')
    return text

def mj_clean(df):
    ''' clean motherjones dataframe '''
    from formatter import mj_clean_text
    pool = Pool(4)
    df['content'] = pool.map(mj_clean_text, df['content'])
    return df


def od_clean_text(text):
    ''' Clean occupydemocrats text '''
    kill_from_args = [  'Follow him on Twitter ',
                        'Sign up for ',
                     ]

    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg]
        text = kill_from_line(*kill_args)


    sign_offs = [   '\n Download our NEW',
                    '\n Download our NEW',
                    '\n Watch his remarks here:',
                    'Add your name to millions',
                    '\n A post shared by '
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]

    return text

def od_clean(df):
    ''' Clean occupydemocrats dataframe '''
    from formatter import od_clean_text
    pool = Pool(4)
    df['content'] = pool.map(od_clean_text, df['content'])
    return df


def ai_clean_text(text):
    ''' Clean addicting info text '''
    #TODO Fill this out

    kill_from_args = [  'Getty Images',
                        'Image via',
                        'Featured Image',

                        ]

    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg]
        text = kill_from_line(*kill_args)


    text = text.replace('Addicting Info', 'this newspaper')

    return text

def ai_clean(df):
    ''' Clean addicting info dataframe '''
    from formatter import ai_clean_text
    pool = Pool(4)
    df['content'] = pool.map(ai_clean_text, df['content'])
    return df

def gp_clean_text(text):
    #TODO
    ''' Clean gateway pundit text '''

    kill_from_args = [  'UPDATE: ',
                        'Read the whole thing here.',
                        'Photo image via',
                        'H/T:'
                        ]

    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg]
        text = kill_from_line(*kill_args)


    text = text.replace('Gateway Pundit', 'this newspaper')

    return text

def gp_clean(df):
    ''' Clean gateway pundit dataframe '''
    from formatter import gp_clean_text
    pool = Pool(4)
    df['content'] = pool.map(gp_clean_text, df['content'])
    return df

def nyt_clean_text(text):
    ''' Clean new york times text '''

    aborts = ['ourpicks@nytimes.com','CAtoday@nytimes.com.']

    for abort in aborts:
        if text.find(abort) != -1:
            return ''


    sign_ins =  [   'WASHINGTON — ',
                    'CHICAGO — '
                ]
    for sign_in in sign_ins:
        i = text.find(sign_in)
        if i != -1:
            text = text[i+len(sign_in):]

    kill_from_args = [  'Follow him on Twitter ',
                        'Sign up for ',
                        'contributed reporting',
                        'Get politics and Washington news updates',
                        'reported from Washington'
                     ]

    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg]
        text = kill_from_line(*kill_args)


    sign_offs = [   '\n Read the full ',
                    '\n A version of this article ',
                    ' \n Follow '
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]

    text = text.replace('\n Please verify you’re not a robot by clicking the box. \n Invalid email address. Please re-enter. \n You must select a newsletter to subscribe to. \n View all New York Times newsletters. \n','\n')
    text = text.replace('\n Advertisement \n', '\n')
    text = text.replace('New York Times', 'this newspaper')
    text = text.replace('NYT', 'this newspaper')
    text = text.replace('nytimes', 'this newspaper')
    return text

def nyt_clean(df):
    ''' Clean new york times text '''
    from formatter import nyt_clean_text
    pool = Pool(4)
    df['content'] = pool.map(nyt_clean_text, df['content'])
    return df

def cnn_clean_text(text):
    ''' Clean cnn text '''


    sign_ins =  [   'WASHINGTON — ',
                    'CHICAGO — '
                ]
    for sign_in in sign_ins:
        i = text.find(sign_in)
        if i != -1:
            text = text[i+len(sign_in):]

    kill_from_args = [  'Follow him on Twitter ',
                        'Sign up for ',
                        'contributed reporting',
                        'Get politics and Washington news updates',
                        'reported from Washington'
                     ]

    for kill_from_arg in kill_from_args:
        kill_args = [text, kill_from_arg]
        text = kill_from_line(*kill_args)


    sign_offs = [   '\n Read the full ',
                    '\n A version of this article ',
                    ' \n Follow ',
                    'CORRECTION',
                    'UPDATE',
                ]

    for sign_off in sign_offs:
        i = text.find(sign_off)
        if i != -1:
            text = text[:i]
    text = text.replace('CNN', 'this newspaper')
    text = text.replace(' --', ',')
    text.replace('  ', '\n')

    return text



def cnn_clean(df):
    ''' Clean cnn text '''
    from formatter import cnn_clean_text
    pool = Pool(4)
    df['content'] = pool.map(cnn_clean_text, df['content'])
    return df


def find_leak(df, keyword):
    ''' For data munging, find a potentially leaky text segment '''
    for content in list(df.content.values):
        i = content.find(keyword)
        section = content[i-100:]
        if i != -1:
            yield section


def cull_shorts(df, min_length=400):
    ''' Cut out rows with less than min_length chrs of text '''
    lens = np.array(df.content.apply(len))
    inds = np.argwhere(lens > min_length).ravel()
    df = df.iloc[inds]
    df.reset_index(inplace=True, drop=True)
    return df


def save_full_corpus():
    in_dir=DATA_PATH
    standard_art_df = pd.read_pickle(in_dir+'articles.pkl')
    hold_art_df = pd.read_pickle(in_dir+'holdout.pkl')
    cnn_df = pd.read_pickle(in_dir+'cnn.pkl')

    full_df = pd.concat([standard_art_df, hold_art_df, cnn_df], ignore_index=True)
    full_df.to_pickle(in_dir+'full_corpus.pkl')
    print('In total, your full article corpus has: {} articles. Be careful not to leak!'
            .format(full_df.shape[0]))


def main(out_dir=DATA_PATH):
    '''
    Load dfs from table using data_loader
    Run universal_cleaner on text

    Run source specific cleaning on text
        (depends on consistent formatting, i.e. don't change universal_cleaner)

    Run universal_stripper on text
     '''


    ''' Article Dfs'''
    dfs = dl.load_dfs()
    dfs = {name:universal_cleaner(dfs[name]) for name in dfs}

    dfs['fox'] = fox_clean(dfs['fox'])
    dfs['hp'] = hp_clean(dfs['hp'])
    dfs['reu'] = reu_clean(dfs['reu'])

    dfs['bb'] = bb_clean(dfs['bb'])
    dfs['mj'] = mj_clean(dfs['mj'])
    dfs['od'] = od_clean(dfs['od'])

    dfs = {name:universal_stripper(dfs[name]) for name in dfs}
    dfs = {name:cull_shorts(dfs[name], min_length=1000) for name in dfs}
    df = pd.concat( list(dfs.values()), ignore_index=True )
    df.to_pickle(out_dir+'articles.pkl')

    print("Cleaned articles, you've got {} of them".format(df.shape[0]))

    dfs.pop('reu')
    dfnc = pd.concat( list(dfs.values()), ignore_index=True )
    dfnc.to_pickle(out_dir+'articles_no_center.pkl')

    print("Cleaned articles no center, you've got {} of them".format(dfnc.shape[0]))


    '''Reddit dfs'''
    rdf = dl.load_reddit()
    rdf = cull_shorts(rdf, min_length=600)
    rdf = universal_cleaner(rdf)
    rdf = universal_stripper(rdf)
    rdf = cull_shorts(rdf, min_length=600)
    rdf.to_pickle(out_dir+'reddit.pkl')

    print("Cleaned reddit comments, you've got {} of them".format(rdf.shape[0]))


    rdf = pd.read_pickle('../data/reddit.pkl')
    '''Reddit dfs no center'''
    rncdf = rdf[rdf.bias != 0]
    rncdf.reset_index(inplace=True, drop=True)
    rncdf.to_pickle(out_dir+'reddit_no_center.pkl')

    print("Cleaned reddit no center comments, you've got {} of them".format(rncdf.shape[0]))


    mixed_df = pd.concat( [df, rdf], ignore_index=True )
    mixed_df.to_pickle(out_dir+'mixed.pkl')


    ''' Holdout Dfs'''
    hdfs = dl.load_holdout()
    hdfs = {name:universal_cleaner(hdfs[name]) for name in hdfs}

    hdfs['nyt'] = nyt_clean(hdfs['nyt'])
    hdfs['nyt']['source'] = 'nyt'
    hdfs = {name:universal_stripper(hdfs[name]) for name in hdfs}
    hdfs = {name:cull_shorts(hdfs[name], min_length=1000) for name in hdfs}

    hdf = pd.concat( list(hdfs.values()), ignore_index=True )
    hdf.to_pickle(out_dir+'holdout.pkl')

    print("Cleaned holdout articles, you've got {} of them".format(hdf.shape[0]))

    udf = dl.ultra_holdout()
    udf = universal_cleaner(udf)
    udf = cnn_clean(udf)
    udf = universal_stripper(udf)
    udf = cull_shorts(udf, min_length=1000)
    udf.to_pickle(out_dir+'udf.pkl')

    # now instead set as central
    cnn_df = udf
    cnn_df['bias']=0
    cnn_df['orient'] = 'center'
    cnn_df.to_pickle(out_dir+'cnn.pkl')

    print("Cleaned cnn, you have {} cnn articles, have fun".format(udf.shape[0]))

    save_full_corpus()


if __name__ == '__main__':
    plac.call(main)
