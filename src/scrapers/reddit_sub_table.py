import scrape_tools as st





if __name__ == '__main__':

    # table = st.open_database_collection('reddit')
    # table2 = st.open_database_collection('political_reddit')
    #
    # left_subs = ['hillaryclinton', 'politics','progressive']
    # right_subs = ['the_donald', 'conservative','republican']
    #
    # subs = ['hillaryclinton', 'politics','progressive', 'the_donald', 'conservative','republican']


    # mongo db command to create collection of comments from these subs and with a score of at least 5
    '''

    var fil =
    {
        $and:[
                    {subreddit: {$in: ['hillaryclinton', 'politics','progressive', 'the_donald', 'conservative','republican']}},
                    {score: {$gte: 5} }
             ]
    }

    var bulkInsert = db.political_reddit.initializeUnorderedBulkOp()
    var x = 10000
    var counter = 0

    var proj =
    {
        body: 1,
        subreddit: 1,
        author: 1,
        score: 1,
        created_utc: 1
    }

    db.reddit.find(fil, proj).forEach(
        function(doc){
            bulkInsert.insert(doc);
            counter ++
            if( counter % x == 0){
                bulkInsert.execute()
                bulkInsert = db.political_reddit.initializeUnorderedBulkOp()
          }
        }
      )
    bulkInsert.execute()





        function copyDocuments(sourceCollection, targetCollection, filter){
            var bulkInsert = targetCollection.initializeUnorderedBulkOp();
            sourceCollection.find(filter).forEach(
                function(doc) { bulkInsert.insert(doc); }
            )
            bulkInsert.execute();
        }

        function copy5(sourceCollection, targetCollection, filter){
            var bulkInsert = targetCollection.initializeUnorderedBulkOp();
            sourceCollection.find(filter).limit(5).forEach(
                function(doc) { bulkInsert.insert(doc); }
            )
            bulkInsert.execute();
        }

        var fil =
        {
            $and:[
                        {subreddit: {$in: ['hillaryclinton', 'politics','progressive', 'the_donald', 'conservative','republican']}},
                        {score: {$gte: 5} }
                 ]
        }

        copyDocuments(db.reddit, db.political_reddit, x)
    '''

    table = st.open_database_collection('political_reddit')
