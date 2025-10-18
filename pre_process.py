import pandas as pd
import ast
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = 'Top_10000_Movies.csv'

# Use Python engine to avoid C parser buffer overflow
df = pd.read_csv(file_path, engine='python')

"""print(df.head())                        # print 1st few columns
print(df.columns)                       # shows the columns
date_rel=df['release_date']             # retrieving a specific column
print(date_rel.head())                  # prints specific column 1st few columns
date_lst=date_rel.to_list()             # makes the column into a list
print(date_lst[:5])                     # prints the list upto 5 
print(df.dtypes)                        # type of data i.e; int,float or stings , object etc;
print(df["release_date"].dtype)"""         # specific column data type
missing_values=df['release_date'].isna().sum()# total no of missing values in release_date column
#print(missing_values)
df['release_date']=pd.to_datetime(df["release_date"],errors='coerce')# object into datetime format 
#print(df["release_date"].dtype)
movie_null_rel=df[df['release_date'].isna()][['original_title','id']]    # Finding movie names and id's where release dates are null 
#print(movie_null_rel)
#print(len(df))                          #Total num of rows

chng = {
    610150: "11 June 2022",
    553301: "26 August 2016",
    533535: "26 July 2024",
    891060: "5 August 2023",
    496450: "5 July 2023",
    815762: "31 December 2018",
    617126: "25 July 2025",
    663712: "6 October 2022"
}
for movie_id,date_str in chng.items():    #filling some of the release dates using their id's
    new_date=pd.to_datetime(date_str)
    df.loc[df['id']==movie_id,'release_date']=new_date
#print(df.loc[df['id'].isin(chng.keys()),['id','release_date']])

df.dropna(subset=['release_date'],inplace=True)    # dropping all null values in release_date column
#print(len(df))

df.drop_duplicates(subset=['original_title'],keep='first',inplace=True)   #remove movie title dupliactes 
#print(len(df))

df.reset_index(drop=True,inplace=True)    # re-assign index values 

#print(df.duplicated(subset=['original_title']).sum()) # check if any movie with duplicates

df['genre']=df['genre'].apply(lambda x : ast.literal_eval(x) if isinstance(x,str) else []) # genre conatins strings "[1,2,"abc"]" into lists [1,2,"abc"]
df=df[df['genre'].map(len)>0]                           # dropping empty genres 

df['combined_text']=df['overview']+' '+df['tagline']
df['combined_text']=df['combined_text'].fillna('')

transactions=df['genre'].tolist()           # converting to lists for encoding 

te=TransactionEncoder()                     # instance of Transaction encoder
te_ary=te.fit_transform(transactions)       # fit learns all unique values  and transforms into binary numpy array
df_genres=pd.DataFrame(te_ary,columns=te.columns_)      #converts into pandas dataframe and columns to remember the original coulmn names

frequent_itemsets=apriori(df_genres,min_support=0.02,use_colnames=True)       # apriori algorithm for finding the frequent_itmesets
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1.0)    # finding association rules

vector=TfidfVectorizer(stop_words='english',max_features=5000)
tidf_matrix=vector.fit_transform(df['combined_text'])
cosine_sim=cosine_similarity(tidf_matrix,tidf_matrix)

def hybrid_recomm(title,top_n=10):
    movie_row=df[df['original_title'].str.lower()==title.lower()]
    if movie_row.empty:
        return None,f"Movie {title} not found"
    
    idx=movie_row.index[0]
    movie_lang=movie_row.iloc[0]['original_language']
    movie_genres=set(movie_row.iloc[0]['genre'])
    movie_rating=movie_row.iloc[0]['vote_average']

    lang_filtered=df[df['original_language']==movie_lang].copy()
    lang_filtered['genre_overlap']=lang_filtered['genre'].apply(lambda g: len(movie_genres & set(g)))

    genre_filtered=lang_filtered[lang_filtered['genre_overlap']>0]
    if len(genre_filtered) < top_n:
        genre_filtered=lang_filtered

    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_df=pd.DataFrame(sim_scores,columns=['index','similarity'])
    sim_df=sim_df.merge(df[['original_title']],left_on='index',right_index=True)

    filtered_df=genre_filtered.merge(sim_df,on='original_title')
    filtered_df=filtered_df.sort_values(by=['genre_overlap','similarity','vote_average'],ascending=[False,False,False])

    filtered_df = filtered_df[filtered_df['original_title'].str.lower() != title.lower()]
    reason = (
        f"\n **Because you watched '{title}'**\n"
        f" Language match: *{movie_lang.upper()}*\n"
        f" Genre overlap: *{', '.join(movie_genres)}*\n"
        f" Your movie rating: *{movie_rating}*\n\n"
        "These movies were chosen because they share:\n"
        "- The same language\n"
        "- Similar genres\n"
        "- Related storylines (overview + tagline)\n"
        "- And have high audience ratings.\n")
    #print(reason)
    return filtered_df[['original_title', 'genre', 'vote_average']].head(top_n),None


    


# print(rules.sort_values("lift",ascending=False).head(5))                      # top 5 strong assciation rules










