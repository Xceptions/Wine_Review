import numpy as np
import pandas as pd
import luigi
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle


task_make_path = '../data_root/Make/'
task_FE_path = '../data_root/FE/'
task_train_path = '../data_root/Train/'
task_split_path = '../data_root/Split/'
directories = [task_make_path,
                task_FE_path,
                task_train_path,
                task_split_path
                ]
for i in directories:
    if not os.path.exists(i):
        os.makedirs(i)


class ReadData(luigi.Task):
    """
    Read the original csv file and output the same file.
    This is done to be sure that the file can be successfully read and worked
    on. Save it as ReadData.csv in the data_root/Make folder
    """
    input_data = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                task_make_path, "{}.csv".format("ReadData")
            )
        )

    def run(self):
        data = pd.read_csv(self.input_data)
        with self.output().open('w') as write_out:
            print(data.to_csv(index=False), file=write_out)


class SplitData(luigi.Task):
    """
    Split the data into train and test sets on the basis of percentages.
    80% for training and 20% for testing. Training will be used to
    build the model and testing for evaluating the model.
    Save as TrainData.csv and TestData.csv
    """
    input_data = luigi.Parameter()

    def output(self):
        return {
            'out1':  luigi.LocalTarget(
                os.path.join(
                    task_split_path,
                    "{}.csv".format("TrainData")
                )
            ),
            'out2': luigi.LocalTarget(
                os.path.join(
                    task_split_path,
                    "{}.csv".format("TestData")
                )
            )
        }

    def requires(self):
        return ReadData(input_data=self.input_data)

    def run(self):
        data = pd.read_csv(os.path.join(
            task_make_path, "ReadData.csv")
        )
        split_1, split_2 = np.split(data, [int(.8*len(data))])
        train_data = pd.DataFrame(split_1)
        test_data = pd.DataFrame(split_2)
        with self.output()['out1'].open('w') as write_out1:
            print(train_data.to_csv(index=False), file=write_out1)
        with open(self.output()['out2'].path, 'w') as write_out2:
            print(test_data.to_csv(index=False), file=write_out2)


class DeduplicateData(luigi.Task):
    """
    Read the TrainData.csv, check if there are duplicate rows
    in the data based on the taster_name, title and description,
    and drop them
    """
    input_data = luigi.Parameter()

    def requires(self):
        return SplitData(input_data=self.input_data)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                task_make_path,
                "{}.csv".format("DeduplicateData")
            )
        )

    def run(self):
        df = pd.read_csv(os.path.join(task_split_path, "TrainData.csv"))
        dedup_cols = ['taster_name', 'title', 'description']
        data = df.drop_duplicates(dedup_cols)
        with self.output().open('w') as write_out:
            print(data.to_csv(index=False), file=write_out)


class FeatureSelection(luigi.Task):
    """
    My model is based on the insight that a person's rating of
    wine is based on price, culture and individual differences
    so select the features that can help to extract these features
    for building the model on.
    These features are price, points, country and province
    save as FeatureSelection.csv
    """
    input_data = luigi.Parameter()

    def requires(self):
        return DeduplicateData(input_data=self.input_data)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                task_make_path,
                "{}.csv".format("FeatureSelection")
            )
        )

    def run(self):
        df = pd.read_csv(os.path.join(
            task_make_path,
            "DeduplicateData.csv")
        )
        cols = ['price', 'points', 'country', 'province']
        data = df[cols].copy()
        with self.output().open('w') as write_out:
            print(data.to_csv(index=False), file=write_out)


class HandleMissingData(luigi.Task):
    """
    To handle missing data:
    Convert every infinite value (if they exist) to NaN
    Fill the NaN values with the mean.
    The rows which have country and province empty can
    be discarded since it is text data that is missing and word2vec
    is not permitted in this challenge
    save as HandleMissingData.csv
    """
    input_data = luigi.Parameter()

    def requires(self):
        return FeatureSelection(input_data=self.input_data)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                task_make_path,
                "{}.csv".format("HandleMissingData")
            )
        )

    def run(self):
        data = pd.read_csv(os.path.join(
            task_make_path,
            "FeatureSelection.csv")
        )
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.mean())
        # drop the rows where country and province are null
        data = data.dropna()
        with self.output().open('w') as write_out:
            print(data.to_csv(index=False), file=write_out)


class FeatureEngineering(luigi.Task):
    """
    Generate the features that will directly be
    used in building the model based on the features we
    selected.
    Then save these features for persistence when predicting
    new data
    save file as FeatureEngineering.csv
    save features as Persistence.pkl
    """
    input_data = luigi.Parameter()

    def requires(self):
        return HandleMissingData(input_data=self.input_data)

    def output(self):
        return {
            'out1': luigi.LocalTarget(
                os.path.join(
                    task_FE_path,
                    "{}.csv".format("FeatureEngineering")
                )
            ),
            'out2': luigi.LocalTarget(
                os.path.join(
                    task_FE_path,
                    "{}.pkl".format("Persistence")
                )
            )
        }

    def run(self):
        data = pd.read_csv(os.path.join(
            task_make_path,
            "HandleMissingData.csv")
        )
        COUNTRY = data.groupby('country')
        PROVINCE = data.groupby('province')
        data['price_per_country_mean'] = COUNTRY['price'].transform('mean')
        data['price_per_country_mean_diff'] = data['price'] - data[
                                                'price_per_country_mean']
        data['price_per_country_median'] = COUNTRY['price'].transform(
                                                                'median'
                                                            )
        data['price_per_country_median_diff'] = data['price'] - data[
                                                'price_per_country_median']
        data['price_per_province_mean'] = PROVINCE['price'].transform('mean')
        data['price_per_province_mean_diff'] = data['price'] - data[
                                                'price_per_province_mean']
        data['price_per_province_median'] = PROVINCE['price'].transform(
                                                                'median'
                                                            )
        data['price_per_province_median_diff'] = data['price'] - data[
                                                'price_per_province_median']
        data['points_per_country_mean'] = COUNTRY['points'].transform('mean')
        data['points_per_country_median'] = COUNTRY['points'].transform(
                                                                'median'
                                                            )
        data['points_per_province_mean'] = PROVINCE['points'].transform('mean')
        data['points_per_province_median'] = PROVINCE['points'].transform(
                                                                    'median'
                                                                )
        country_cols = [
            'country',
            'price_per_country_mean',
            'price_per_country_mean_diff',
            'price_per_country_median',
            'price_per_country_median_diff',
            'points_per_country_mean',
            'points_per_country_median',
        ]
        province_cols = [
            'province',
            'price_per_province_mean',
            'price_per_province_mean_diff',
            'price_per_province_median',
            'price_per_province_median_diff',
            'points_per_province_mean',
            'points_per_province_median',
        ]
        var_persistence = data.drop_duplicates(['country', 'province'])
        country_persistence = var_persistence[country_cols].copy()
        province_persistence = var_persistence[province_cols].copy()
        country_dict = country_persistence.set_index('country').T.to_dict('list')
        province_dict = province_persistence.set_index('province').T.to_dict('list')
        var_persistence = [country_dict, province_dict]
        # var_dict = var_persistence.set_index('country').T.to_dict('list')
        with open(self.output()['out1'].path, 'w') as write_out:
            print(data.to_csv(index=False), file=write_out)
        with open(self.output()['out2'].path, 'wb') as write_out2:
            pickle.dump(var_persistence, write_out2)


class Make(luigi.Task):
    """
    Create the final engineered data for training.
    save as Make.csv
    """
    input_data = luigi.Parameter()

    def requires(self):
        return FeatureEngineering(input_data=self.input_data)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                task_make_path,
                "{}.csv".format("Make")
            )
        )

    def run(self):
        data = pd.read_csv(os.path.join(
            task_FE_path,
            "FeatureEngineering.csv")
        )
        model_train_cols = [
            'price',
            'price_per_country_mean',
            'price_per_country_mean_diff',
            'price_per_country_median',
            'price_per_country_median_diff',
            'price_per_province_mean',
            'price_per_province_mean_diff',
            'price_per_province_median',
            'price_per_province_median_diff',
            'points_per_country_mean',
            'points_per_country_median',
            'points_per_province_mean',
            'points_per_province_median',
            'points'
        ]
        data = data[model_train_cols].copy()
        with open(self.output().path, 'w') as final_train:
                print(data.to_csv(index=False), file=final_train)


class TrainModel(luigi.Task):
    input_data = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                task_train_path,
                "{}.pkl".format("Model")
            )
        )

    def requires(self):
        return Make(input_data=self.input_data)
    
    def run(self):
        data = pd.read_csv(os.path.join(
            task_make_path,
            "Make.csv")
        )
        train_features = [
            'price',
            'price_per_country_mean',
            'price_per_country_mean_diff',
            'price_per_country_median',
            'price_per_country_median_diff',
            'price_per_province_mean',
            'price_per_province_mean_diff',
            'price_per_province_median',
            'price_per_province_median_diff',
            'points_per_country_mean',
            'points_per_country_median',
            'points_per_province_mean',
            'points_per_province_median',
        ]
        target_feature = 'points'
        train_data = data[train_features].values
        target_data = data[target_feature].values
        # train the model using cross validation of 10 folds
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(train_data):
            X_train, X_valid = train_data[train_index], train_data[test_index]
            y_train, y_valid = target_data[train_index], target_data[test_index]
            xgb_model = xgb.XGBRegressor(
                            n_estimators=1000,
                            max_depth=20,
                            importance_type="gain",
                            learning_rate=0.01,
                            n_jobs=4
                        )
            xgb_model.fit(X_train, y_train,
                        early_stopping_rounds=5,
                        eval_set=[(X_valid, y_valid)],
                        eval_metric="rmse",
                        verbose=True)
        with open(self.output().path, 'wb') as file:
            pickle.dump(xgb_model, file)
        # print('saved')
