from functools import partial

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from xfeat import (ArithmeticCombinations, ConcatCombination,
                   GBDTFeatureExplorer, GBDTFeatureSelector, LabelEncoder,
                   Pipeline, SelectCategorical, SelectNumerical, TargetEncoder,
                   aggregation)


# カテゴリカルデータのみ抽出
def extracting_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    return SelectCategorical().fit_transform(df)


# 数値データのみ抽出
def extracting_number_data(df: pd.DataFrame) -> pd.DataFrame:
    return SelectNumerical().fit_transform(df)


# Lable Encoding
# カテゴリカルデータの各カテゴリを整数に置き換える変換
def label_encode(
    df: pd.DataFrame,
    exclude_cols: list[str] = ['Name', 'Ticket'],
    output_suffix=''
) -> pd.DataFrame:
    encoder = Pipeline([
        SelectCategorical(exclude_cols=exclude_cols),
        LabelEncoder(output_suffix=output_suffix),
    ])
    return encoder.fit_transform(df)


# Target Encoding
# 目的変数を用いてカテゴリカルデータを数値に変換
def target_encode(
    df: pd.DataFrame,
    n_splits: int = 5,
    input_cols: list[str] = ['Cabin'],
    target_col: str = 'Survived',
    output_suffix: str = '_te'
) -> pd.DataFrame:
    fold = KFold(n_splits=n_splits, shuffle=False)
    encoder = TargetEncoder(
        input_cols=input_cols,
        target_col=target_col,
        fold=fold,
        output_suffix=output_suffix
    )
    ret_df = encoder.fit_transform(df).loc[
        :, [x+output_suffix for x in input_cols]
    ]
    return ret_df


# カテゴリカルデータの組み合わせ
def combine_categorical_data(
    df: pd.DataFrame,
    exclude_cols: list[str] = ['Ticket', 'Name'],
    output_suffix: str = '_re',
    r: int = 2  # 組み合わせる変数の数
) -> pd.DataFrame:
    encoder = Pipeline([
        SelectCategorical(exclude_cols=exclude_cols),
        ConcatCombination(
            output_suffix=output_suffix,
            r=r
        ),
    ])
    return encoder.fit_transform(df)


# 指定した変数の数値データの加算
def additional_num_data(
    df: pd.DataFrame,
    input_cols: list[str] = ['SibSp', 'Parch'],
    drop_origin: bool = True,
    operator: str = '+',
    r: int = 2  # 組み合わせる変数の数
) -> pd.DataFrame:
    encoder = Pipeline([
        SelectNumerical(),
        ArithmeticCombinations(
            input_cols=input_cols,
            drop_origin=drop_origin,
            operator=operator,
            r=r,
        ),
    ])
    return encoder.fit_transform(df)


# 指定した変数の値の集約
def aggregation_num_data(
    df: pd.DataFrame,
    group_key: str = 'Sex',
    group_values: list[str] = ['Age', 'Pclass'],
    agg_methods: list[str] = ['mean', 'max'],  # maxが利くことあるのか…？
) -> pd.DataFrame:
    aggregated_df, aggregated_cols = aggregation(
        df,
        group_key=group_key,
        group_values=group_values,
        agg_methods=agg_methods
    )
    return aggregated_df.loc[:, aggregated_cols]


def pipe_all_process(df: pd.DataFrame) -> pd.DataFrame:
    te_df = target_encode(df)
    an_df = additional_num_data(df, drop_origin=False)
    agg_df = aggregation_num_data(df)
    le_df = label_encode(
        combine_categorical_data(df, r=2, exclude_cols=['Name', 'Ticket']),
        exclude_cols=[]
    )
    cleaned_df = pd.concat([an_df,  le_df, te_df, agg_df], axis=1)
    return cleaned_df


# optunaとLightGBMのfeature importanceを用いた特徴量選択
def select_features_by_lgbm(
    cleaned_df: pd.DataFrame,
    target_col: str = 'Survived',
    num_boost_round: int = 100,
    stratified: bool = False,
    seed: int = 1,
    LGBM_PARAMS: dict = None
) -> None:
    if LGBM_PARAMS is None:
        LGBM_PARAMS = {
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1
        }

    def objective(
        df: pd.DataFrame,
        selector: GBDTFeatureSelector,
        trial,
        label_name: str = target_col,
        num_boost_round: int = num_boost_round,
        stratified: bool = stratified,
        seed: int = seed
    ):
        selector.set_trial(trial)
        selector.fit(df)
        input_cols = selector.get_selected_cols()

        # チューニングするパラメータとその範囲を設定
        lgbm_params = {
            'num_leaves': trial.suggest_int('num_leaves', 3, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 10)
        }
        lgbm_params.update(LGBM_PARAMS)

        # 選択した特徴量により評価
        train_set = lgb.Dataset(df[input_cols], label=df[label_name])
        scores = lgb.cv(
            lgbm_params,
            train_set,
            num_boost_round=num_boost_round,
            stratified=stratified,
            seed=seed
        )
        binary_error_score = scores['binary_error-mean'][-1]
        return 1 - binary_error_score

    # 探索する特徴量を設定
    input_cols = list(cleaned_df.columns)
    input_cols.remove(target_col)

    # 特徴量探索器を作成
    selector = GBDTFeatureExplorer(
        input_cols=input_cols,
        target_col=target_col,
        fit_once=True,
        threshold_range=(0.5, 1.0),
        lgbm_params=LGBM_PARAMS
    )

    # optunaによるパラメータチューニング
    study = optuna.create_study(direction='minimize')
    study.optimize(partial(objective, cleaned_df, selector), n_trials=100)

    # 選択された特徴量を確認
    selector.from_trial(study.best_trial)
    print('selected columns: ', selector.get_selected_cols())

    print('best_params: ', study.best_params)
    print('best_value: ', study.best_value)


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print('train_df: ', train_df.columns)

    cleaned_df = pipe_all_process(train_df)
    cleaned_df = cleaned_df.drop(['PassengerId'], axis=1)

    print('cl_df: ', cleaned_df.columns)

    # LightGBMのfeature importanceを用いた特徴量選択
    # LightGBMのパラメータ設定
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_error',
    }
    fit_kwargs = {'num_boost_round': 10}
    # 特徴量選択器を作成
    selector = GBDTFeatureSelector(
        target_col='Survived',
        threshold=0.5,  # 入力データの変数のうちいくつを選択するかの割合
        lgbm_params=lgbm_params,
        lgbm_fit_kwargs=fit_kwargs
    )
    selected_df = selector.fit_transform(cleaned_df)
    print('Selected columns: ', selector._selected_cols)
    print('seleted_df: ', selected_df.columns)  # 超便利では…？

    select_features_by_lgbm(cleaned_df)
