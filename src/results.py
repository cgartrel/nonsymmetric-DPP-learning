import os
import pickle
import numpy as np
import scipy.stats

class Results(object):
    mpr_column = "test-all-baskets.MPR"
    prec_ten_column = "test-all-baskets.Prec@10"
    prec_five_column = "test-all-baskets.Prec@5"
    all_baskets_AUC = "test-all-baskets.AUC"

    processed_columns = {"mpr": mpr_column,
                         "prec@5": prec_five_column,
                         "prec@10": prec_ten_column,
                         "auc": all_baskets_AUC}

    @staticmethod
    def read_df(df_path):
        with open(df_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_path(cls, path):
        name_with_ext = os.path.split(path)[1]
        name = os.path.splitext(name_with_ext)[0]
        df = cls.read_df(path)
        return cls(name, df)

    def __init__(self, name, df):
        """
        :param name: name of the result, that should contain the
                     name of the dataset in it (cbs_{retailer}_{1d|10d})
        :param df: the dataframe
        """
        self.name = name
        self.df = df
        self.results = {}
        for name_in_res, column_name in self.processed_columns.items():
            mean, min, max = self.mean_confidence_interval(self.df[column_name])
            curr_res = {"mean": mean,
                        "min": min,
                        "max": max}
            self.results[name_in_res] = curr_res

        self.dataset = self.name

    @staticmethod
    def group_by_dataset(results):
        res = {}
        for r in results:
            res.setdefault(r.dataset, []).append(r)
        return res

    @staticmethod
    def sort(results):
        return sorted(results,
                      key=lambda x: -x.results["mpr"]["mean"])

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = np.array(data)
        n = len(a)
        m = np.mean(a)
        if n == 1:
            return m, m, m
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    def __str__(self):
        res = self.name
        values = []
        for column in self.processed_columns:
            value = self.results[column]
            value_mean, value_min, value_max = value["mean"], value["min"], value["max"]
            values.append(self._mean_min_max_to_str(column,
                                                    value_mean,
                                                    value_min,
                                                    value_max))
        return "%s: %s" % (self.name, ", ".join(values))

    @staticmethod
    def _mean_min_max_to_str(distrib_name, mean, min, max):
        return "%s(%.2f [%.2f, %.2f])" % (distrib_name, mean, min, max)

    def __repr__(self):
        return self.__str__()