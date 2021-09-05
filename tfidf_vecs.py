from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
 
 
def clean_title(x):
    tokens = x.split()
    tokens = [
        token for token in tokens
        if (
                not any([c in token for c in string.punctuation + '0123456789'])
            and not token.isupper()
        )
    ]
 
    return ' '.join(tokens[:4]).lower()
 
def parse_json_(x):
    try:
        return json.loads(x) if not pd.isnull(x) else ''
    except:
        return ''
 
 
df['features_clean'] = (
    df['Характеристики СТЕ']
    .apply(parse_json_)
    .apply(lambda l: ' '.join([x['Name'] for x in l]).lower())
)
df['clean_title'] = df['Наименование СТЕ'].apply(clean_title)
df['text'] = df['clean_title'] + ' ' + df['features_clean']
 
 
mystem = Mystem()
 
df['text'] = df['text'].apply(lambda x: ''.join(mystem.lemmatize(x)[:-1]))
 
index_valid = (
    df
    .set_index('Идентификатор СТЕ')
    .join(
        df_pred
        .set_index('Идентификатор СТЕ'),
        how='left'
    )
    .dropna(subset=['Сопутствующие товары'])
).index
 
 
tfidf = TfidfVectorizer(max_features=5000)
 
tfidf.fit(df['text'].tail(180000))
vectors = tfidf.transform(df['text'])
vectors_valid = tfidf.transform(df.set_index('Идентификатор СТЕ').loc[index_valid]['text'])
 
sparse.save_npz('tmp/vectors.npz', vectors)
sparse.save_npz('tmp/vectors_valid.npz', vectors_valid)
