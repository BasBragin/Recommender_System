import os
from re import I
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet, Response
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger
import hashlib

CONN = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"\
        "postgres.lab.karpov.courses:6432/startml"

SALT = 'ab_salt'

app = FastAPI()

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(CONN)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def get_exp_group(id: int) -> str:
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "Unknown"


def get_model_path(exp_group: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        if exp_group == 'control':
            path = '/workdir/user_input/model_control'
        if exp_group == 'test':
            path = '/workdir/user_input/model_test'
    else:
        if exp_group == 'control':
            path = 'C:/catboost_coef'
        if exp_group == 'test':
            path = 'C:/catboost_model_with_embedding_new'
    return path


def load_models():

    loaded_model_control = CatBoostClassifier()
    control_model_path = get_model_path('control')

    loaded_model_test = CatBoostClassifier()
    test_model_path = get_model_path('test')

    return (
            loaded_model_control.load_model(control_model_path),
            loaded_model_test.load_model(test_model_path)
        )


def load_features() -> pd.DataFrame:
    logger.info('loading liked posts')
    liked_posts_query = '''
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action = 'like'
        LIMIT 200000
        '''
    liked_posts = batch_load_sql(liked_posts_query)

    logger.info('loading user features')
    user_info = pd.read_sql('''
        SELECT *
        FROM public.user_data ''',
                                con=CONN
                                )

    logger.info('loading posts features test')
    posts_features_test = pd.read_sql(f'''
            SELECT * FROM public.bragin_posts_info_features_with_pca_embeddings''',
                                 con=CONN
                                 )

    logger.info('loading posts features control')
    posts_features_control = pd.read_sql(f'''
                SELECT * FROM public.post_text_df''',
                                 con=CONN
                                 )


    return [liked_posts, user_info, posts_features_test, posts_features_control]


def get_recommended_feed_test(id: int, time: datetime, limit: int, exp_group: str):
    logger.info(f'user_id: {id}')
    logger.info('reading_features')
    user_info = features[1].loc[features[1].user_id == id]
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    posts_features = features[2]
    posts_features = posts_features[~posts_features.index.isin(liked_posts)]



    logger.info('add topic')

    user_posts_features = pd.merge(
        posts_features, user_info,
        how='cross')

    content = features[2][['post_id', 'text', 'topic']]

    logger.info('droppping_columns')
    user_posts_features = user_posts_features.drop([
        'text',
    ],
        axis=1)

    logger.info('add time info')
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    user_posts_features = user_posts_features.drop('index', axis=1)

    user_posts_features_ = pd.DataFrame()
    user_posts_features_['user_id'] = user_posts_features['user_id']
    user_posts_features_['post_id'] = user_posts_features['post_id']
    user_posts_features_['topic'] = user_posts_features['topic']
    user_posts_features_['TextCluster'] = user_posts_features['TextCluster']
    user_posts_features_['DistanceTo1thCluster'] = user_posts_features['DistanceTo1thCluster']
    user_posts_features_['DistanceTo2thCluster'] = user_posts_features['DistanceTo2thCluster']
    user_posts_features_['DistanceTo3thCluster'] = user_posts_features['DistanceTo3thCluster']
    user_posts_features_['DistanceTo4thCluster'] = user_posts_features['DistanceTo4thCluster']
    user_posts_features_['DistanceTo5thCluster'] = user_posts_features['DistanceTo5thCluster']
    user_posts_features_['DistanceTo6thCluster'] = user_posts_features['DistanceTo6thCluster']
    user_posts_features_['DistanceTo7thCluster'] = user_posts_features['DistanceTo7thCluster']
    user_posts_features_['DistanceTo8thCluster'] = user_posts_features['DistanceTo8thCluster']
    user_posts_features_['DistanceTo9thCluster'] = user_posts_features['DistanceTo9thCluster']
    user_posts_features_['DistanceTo10thCluster'] = user_posts_features['DistanceTo10thCluster']
    user_posts_features_['DistanceTo11thCluster'] = user_posts_features['DistanceTo11thCluster']
    user_posts_features_['DistanceTo12thCluster'] = user_posts_features['DistanceTo12thCluster']
    user_posts_features_['DistanceTo13thCluster'] = user_posts_features['DistanceTo13thCluster']
    user_posts_features_['DistanceTo14thCluster'] = user_posts_features['DistanceTo14thCluster']
    user_posts_features_['DistanceTo15thCluster'] = user_posts_features['DistanceTo15thCluster']
    user_posts_features_['gender'] = user_posts_features['gender']
    user_posts_features_['age'] = user_posts_features['age']
    user_posts_features_['country'] = user_posts_features['country']
    user_posts_features_['city'] = user_posts_features['city']
    user_posts_features_['exp_group'] = user_posts_features['exp_group']
    user_posts_features_['os'] = user_posts_features['os']
    user_posts_features_['source'] = user_posts_features['source']
    user_posts_features_['hour'] = user_posts_features['hour']
    user_posts_features_['month'] = user_posts_features['month']

    model = model_cat[1]
    predicts = model.predict_proba(user_posts_features_)[:, 1]

    user_posts_features_['predicts'] = predicts

    recommended_posts = user_posts_features_.sort_values('predicts')[-limit:].post_id.values

    return Response(
         recommendations=[
             PostGet(**{
                'id': i,
                'text': content[content.post_id == i].text.values[0],
                'topic': content[content.post_id == i].topic.values[0]
            })
             for i in recommended_posts
         ],
         exp_group=exp_group,
    )


def get_recommended_feed_control(id: int, time: datetime, limit: int, exp_group: str):
    logger.info(f'user_id: {id}')
    logger.info('reading_features')
    user_features = features[1].loc[features[1].user_id == id]
    user_features = user_features.drop('exp_group', axis=1)

    logger.info('droppping_columns')

    posts_features = features[3].drop(['text'], axis=1)
    content = features[3][['post_id', 'text', 'topic']]

    logger.info('add topic')
    user_posts_features = pd.merge(
        user_features, posts_features,
        how='cross')

    logger.info('add time info')
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    user_posts_features_ = pd.DataFrame()
    user_posts_features_['user_id'] = user_posts_features['user_id']
    user_posts_features_['post_id'] = user_posts_features['post_id']
    user_posts_features_['topic'] = user_posts_features['topic']
    user_posts_features_['gender'] = user_posts_features['gender']
    user_posts_features_['age'] = user_posts_features['age']
    user_posts_features_['country'] = user_posts_features['country']
    user_posts_features_['city'] = user_posts_features['city']
    user_posts_features_['os'] = user_posts_features['os']
    user_posts_features_['source'] = user_posts_features['source']
    user_posts_features_['hour'] = user_posts_features['hour']
    user_posts_features_['month'] = user_posts_features['month']

    model = model_cat[0]
    predicts = model.predict_proba(user_posts_features_)[:, 1]

    user_posts_features['predicts'] = predicts

    logger.info('deleting liked posts')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values('predicts')[-limit:].post_id.values

    return Response(
         recommendations=[
             PostGet(**{
                'id': i,
                'text': content[content.post_id == i].text.values[0],
                'topic': content[content.post_id == i].topic.values[0]
            })
             for i in recommended_posts
         ],
         exp_group=exp_group,
    )

logger.info('loading model')
model_cat = load_models()

logger.info('loading_features')
features = load_features()


@app.get("/post/recommendations/", response_model=Response)
def get_recommendations(id: int, time: datetime = datetime.now(), limit: int = 5) -> Response:
    exp_group = get_exp_group(id)
    if exp_group == 'control':
        return get_recommended_feed_control(id, time, limit, exp_group)
    elif exp_group == 'test':
        return get_recommended_feed_test(id, time, limit, exp_group)
    else:
        raise ValueError('unknown group')