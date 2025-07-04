from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_KeyBert_result(document_list):
    result = []
    
    if not document_list or len(document_list) == 0:
        return result
    
    kw_model = KeyBERT()

    for doc in document_list:
        keywords = kw_model.extract_keywords(doc)
        if len(keywords) == 0:
            print("None in keyBert")
            result.append([" "])
        else:
            result.append(keywords)
    return result

def get_tf_idf_result(document_list, max_features=1000, top_k=5):
    result = []
    
    if not document_list or len(document_list) == 0:
        return result
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(document_list)
        feature_names = vectorizer.get_feature_names_out()
        
        for i, doc in enumerate(document_list):
            doc_tfidf = tfidf_matrix[i].toarray().flatten()
            
            top_indices = np.argsort(doc_tfidf)[-top_k:][::-1]
            
            keywords = []
            for idx in top_indices:
                if doc_tfidf[idx] > 0:
                    keywords.append((feature_names[idx], doc_tfidf[idx]))
            
            if len(keywords) == 0:
                print("None in TF-IDF")
                result.append([" "])
            else:
                result.append(keywords)
                
    except Exception as e:
        print(f"Error in TF-IDF processing: {e}")
        result = [[" "] for _ in document_list]
    
    return result
