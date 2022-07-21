from collections import defaultdict
'''
Assume we have some function called get_bm25_score()
'''
def pseudo_relevance(scores, k=3):
    '''
    Returns topk docs after BM25
    '''
    pass
    # Rank scores
    # Get top k documents
    # Return top k docs
    
def queryexp(query, topkdocs):
    '''
    Returns:
        Expanded query
    '''
    pass

    # Get all unique terms from top k docs
    # Create co-occurrence matrix (association matrix?)
    # Populate matrix
    '''
    for uniqueword in topkdocsvocab
        for doc in topkdocs
            add to association matrix
        
    '''

    # Normalize matrix (termi-termj frequency / (sum of termii-termjj diagonals - termi-termj frequency))

    # get relevant words to query from association matrix
    '''
    expandedquery = query
    for word in query
        if word in matrix
            get top m words from matrix
            add to expanded query
        else 
            use wordnet function for synonym
    '''
    # Return expanded 