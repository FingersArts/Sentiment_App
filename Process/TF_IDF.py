# TF-IDF calculation
from preprocessing import df_cleaned
import pandas as pd
import numpy as np

def calc_TF(document):
    TF_dict = {}
    for term in document:
        TF_dict[term] = TF_dict.get(term, 0) + 1
    for term in TF_dict:
        TF_dict[term] /= len(document)
    return TF_dict

df_cleaned["TF_dict"] = df_cleaned["tokenize"].apply(calc_TF)

def calc_DF(tfDict):
    count_DF = {}
    for document in tfDict:
        for term in document:
            count_DF[term] = count_DF.get(term, 0) + 1
    return count_DF

DF = calc_DF(df_cleaned["TF_dict"])
n_document = len(df_cleaned)

def calc_IDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
    return IDF_Dict

IDF = calc_IDF(n_document, DF)

def calc_TF_IDF(TF):
    TF_IDF_Dict = {key: TF[key] * IDF[key] for key in TF}
    return TF_IDF_Dict

df_cleaned["TF-IDF_dict"] = df_cleaned["TF_dict"].apply(calc_TF_IDF)

sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:50]
unique_term = [item[0] for item in sorted_DF]

def calc_TF_IDF_Vec(__TF_IDF_Dict):
    TF_IDF_vector = [__TF_IDF_Dict.get(term, 0.0) for term in unique_term]
    return TF_IDF_vector

df_cleaned["TF_IDF_Vec"] = df_cleaned["TF-IDF_dict"].apply(calc_TF_IDF_Vec)

TF_IDF_Vec_List = np.array(df_cleaned["TF_IDF_Vec"].to_list())
sums = TF_IDF_Vec_List.sum(axis=0)

data = [(term, sums[col]) for col, term in enumerate(unique_term)]

ranking = pd.DataFrame(data, columns=['term', 'rank'])
ranking = ranking.sort_values('rank', ascending=False)

ranking.head()  # print top ranking terms
