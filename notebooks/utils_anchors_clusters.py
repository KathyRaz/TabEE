from typing import Dict, List, Any, Union
import pandas as pd
import numpy as np
import random


def transform_explanation_to_df(exp):
    """
    :param exp: explanation of one example
    :return: transformation of a single explanation to df with columns (idx:name of field): min, max, eql (in case if categorical feature)
    """
    dict_exp = {}
    for field_exp in exp:
        field_exp_split = field_exp.split(' ')
        if len(field_exp_split) == 3:
            if field_exp_split[1] == '>' or field_exp_split[1] == '>=':
                dict_exp.update({field_exp_split[0]: [float(field_exp_split[2]), np.nan, np.nan]})
            elif field_exp_split[1] == '<' or field_exp_split[1] == '<=':
                dict_exp.update({field_exp_split[0]: [np.nan, float(field_exp_split[2]), np.nan]})
            elif field_exp_split[1] == '=':
                dict_exp.update({field_exp_split[0]: [np.nan, np.nan, field_exp_split[2]]})
        elif len(field_exp_split) == 5:
            dict_exp.update({field_exp_split[2]: [float(field_exp_split[0]), float(field_exp_split[4]), np.nan]})
    anchors_df = pd.DataFrame.from_dict(dict_exp, orient='index', columns=['min', 'max', 'equal'])
    anchors_df = anchors_df.sort_index()
    return anchors_df


def update_anchors_dict(potential_fix, anchors_df):
    """
    :param potential_fix: dictionary with fields values and min/max to update the anchor
    :param anchors_df: original anchor df
    :return: updated anchors df
    """

    for field_name in potential_fix.keys():
        if type(potential_fix[field_name][0]) == list:
            anchors_df.loc[field_name][potential_fix[field_name][0][0]] = potential_fix[field_name][0][1]
            anchors_df.loc[field_name][potential_fix[field_name][1][0]] = potential_fix[field_name][1][1]
        else:
            anchors_df.loc[field_name][potential_fix[field_name][0]] = potential_fix[field_name][1]
    return anchors_df


def assign_anchors_to_cluster(exp_array, verbose=0):
    """
    :param verbose: printing option
    :param exp_array: array of explanation previously created for each example in the cluster
    :return: anchors that represent this cluster
    """
    anchors_dict: Dict[int, Dict[str, Union[List[
                                                int], Any]]] = {}  # in form of {exp_num: {df:exp_df, examples:[1,
    # 2]}, }. (for each explanation in exp_of_array we will assign one exp_df and remember which one is it)
    for idx, exp_raw in enumerate(exp_array):
        if verbose:
            print("looking at new exp")
        exp = transform_explanation_to_df(exp_raw)
        found_a_match = 0
        if len(anchors_dict.keys()) == 0:  # initialize the dictionary
            anchors_dict = {1: {'df': exp, 'examples': [idx]}}  # first insertion to dictionary
            if verbose:
                print("updated anchors_dict, now it has {} keys.".format(len(anchors_dict.keys())))
                print("-----------------------------------------------")
                continue
        # the dictionary already has anchors
        for anchor_num in anchors_dict.keys():
            anchor_df: pd.DataFrame = anchors_dict[anchor_num]['df'].sort_index()
            if set(anchor_df.index) != set(exp.index):
                if verbose:
                    print("fields doesn't match")
                    print(set(anchor_df.index) - set(exp.index))
                    print(set(exp.index) - set(anchor_df.index))
                    print("______")
                    continue  # this anchor doesn't match the explanation
            # found a candidate for matching
            if anchor_df.equals(exp):  # the explanation matched completely
                anchors_dict[1]['examples'].append(idx)
                found_a_match = 1
                break  # we found a matching anchor, no need to look further
            # else: the explanation matched partially
            potential_fix = {}
            could_not_be_matched = 0
            for field_name in list(anchor_df.index):
                if anchor_df.loc[[field_name]].equals(exp.loc[[field_name]]):
                    continue
                # found the mismatch
                if (anchor_df.loc[[field_name]]['min'].isna()[0] == True) and (
                        anchor_df.loc[[field_name]]['max'].isna()[0] == True) \
                        and anchor_df.loc[[field_name]]['equal'] != exp.loc[[field_name]]['equal']:
                    could_not_be_matched = 1
                if exp.loc[[field_name]]['min'].isna()[0]:
                    if anchor_df.loc[[field_name]]['max'][0] > exp.loc[[field_name]]['max'][0]:
                        potential_fix[field_name] = ['max', exp.loc[[field_name]]['max'][0]]
                else:
                    if anchor_df.loc[[field_name]]['min'][0] < exp.loc[[field_name]]['min'][0]:  # making the anchor
                        # smaller
                        if anchor_df.loc[[field_name]]['max'].isna()[0]:  # the only constraint is minimum
                            potential_fix[field_name] = ['min', exp.loc[[field_name]]['min'][0]]

                        else:  # there is a constraint on the maximum as well
                            if anchor_df.loc[[field_name]]['max'][0] > exp.loc[[field_name]]['max'][0]:
                                potential_fix[field_name] = [['min', exp.loc[[field_name]]['min'][0]],
                                                             ['max', exp.loc[[field_name]]['max'][0]]]

            if could_not_be_matched == 0:  # didn't find a reason not to match between this anchor and explanation
                anchors_dict[anchor_num]['df'] = update_anchors_dict(potential_fix,
                                                                     anchors_dict[anchor_num]['df'])
                anchors_dict[anchor_num]['examples'].append(idx)
                found_a_match = 1

        if found_a_match == 0:  # there is no anchor in the dictionary that matches this explanation
            next_anchor_id = int(max(anchors_dict.keys())) + 1
            new_anchor = {next_anchor_id: {'df': exp, 'examples': [idx]}}
            anchors_dict.update(new_anchor)  # adding a new anchor to the dictionary
            if verbose:
                print("updated anchors_dict, now it has {} keys.".format(len(anchors_dict.keys())))
                print("-----------------------------------------------")

    return anchors_dict


def anchor_to_filter(anchor_exp, df, verbose=0):
    """
    Take the dataframe and filter it according to anchor explanation
    """
    df_filtered = df.copy()
    if verbose:
        print(anchor_exp)
    for field_exp in anchor_exp:
        field_exp_split: str = field_exp.split(' ')
        if len(field_exp_split) == 3:
            if field_exp_split[1] == '<' or field_exp_split[1] == '<=':
                if field_exp_split[1] == '<':
                    if verbose:
                        print(field_exp)
                        print("df_filtered= df_filtered[df_filtered[{field_name}] < {val} ]".format(
                            val=float(field_exp_split[2]), field_name=field_exp_split[0]))
                    df_filtered = df_filtered[df_filtered[field_exp_split[0]] < float(field_exp_split[2])]
                else:
                    if verbose:
                        print(field_exp)
                        print("df_filtered= df_filtered[df_filtered[{field_name}] <= {val} ]".format(
                            val=float(field_exp_split[2]), field_name=field_exp_split[0]))
                    df_filtered = df_filtered[df_filtered[field_exp_split[0]] <= float(field_exp_split[2])]
            elif (field_exp_split[1] == '>') or (field_exp_split[1] == '>='):
                if field_exp_split[1] == '>':
                    if verbose:
                        print(field_exp)
                        print("df_filtered= df_filtered[df_filtered[{field_name}] > {val} ]".format(
                            val=float(field_exp_split[2]), field_name=field_exp_split[0]))
                    df_filtered = df_filtered[df_filtered[field_exp_split[0]] > float(field_exp_split[2])]
                else:
                    if verbose:
                        print(field_exp)
                        print("df_filtered= df_filtered[df_filtered[{field_name}] >= {val} ]".format(
                            val=float(field_exp_split[2]), field_name=field_exp_split[0]))
                    df_filtered = df_filtered[df_filtered[field_exp_split[0]] >= float(field_exp_split[2])]
            elif field_exp_split[1] == '=':
                if verbose:
                    print(field_exp)
                    print("df_filtered= df_filtered[df_filtered[{field_name}] == {val} ]".format(
                        val=float(field_exp_split[2]), field_name=field_exp_split[0]))
                df_filtered = df_filtered[df_filtered[field_exp_split[0]] == float(field_exp_split[2])]
        elif len(field_exp_split) == 5:  # '1331.00 < WHEELS_OFF <= 1757.00'
            if field_exp_split[1] == '<':
                df_filtered = df_filtered[df_filtered[field_exp_split[2]] < float(field_exp_split[0])]
            elif field_exp_split[1] == '<=':
                df_filtered = df_filtered[df_filtered[field_exp_split[2]] <= float(field_exp_split[0])]
            if field_exp_split[3] == '>':
                df_filtered = df_filtered[df_filtered[field_exp_split[2]] > float(field_exp_split[4])]
            elif field_exp_split[3] == '>=':
                df_filtered = df_filtered[df_filtered[field_exp_split[2]] >= float(field_exp_split[4])]

    return df_filtered


def cover_cluster_w_anchors(df, explainer, predict_func, features, cluster_num, verbose=0, threshold=0.95,
                            num_samples=20, keep_dict=False, num_iterations=5):
    """

    :param num_iterations:
    :param predict_func:
    :param explainer:
    :param df:
    :param features:
    :param cluster_num:
    :param verbose:
    :param threshold:
    :param num_samples:
    :param keep_dict:
    :return:
    """
    index_covered_so_far = set()
    size_cluster = df[df['cluster'] == cluster_num].shape[0]
    cluster_num_index = list(df[df['cluster'] == cluster_num].index)
    all_cluster_idx = cluster_num_index
    if keep_dict:
        exp_dict = {}
    cluster_exp = []
    for iteration in range(0, num_iterations):
        sampled_cluster_num_idx = random.sample(cluster_num_index, num_samples)
        exp_dict_max: dict = {}  # finding the best anchor per iteration (for the tuples that hasn't been covered so far

        for num_example, idx_tuple in enumerate(sampled_cluster_num_idx):
            num_example_iter = num_example + iteration * num_samples
            if verbose:
                print("looking at sample num: ",num_example_iter)
            if df.iloc[idx_tuple]['cluster'] != cluster_num:
                print("error")
                return 0
            exp = explainer.explain_instance(df[features].values[idx_tuple], predict_func, threshold=threshold)
            exp_str = exp.names()
            filtered_df = anchor_to_filter(exp_str, df.iloc[cluster_num_index], verbose=0)
            if num_example == 0:
                if verbose:
                    print("updated max_dict for first time for the iteration")
                exp_dict_max = {'exp_num': num_example_iter, 'exp_str': exp.names(),
                                'idx_list': list(filtered_df.index),
                                'len_idx': filtered_df.shape[0]}
            if keep_dict:
                exp_dict[num_example_iter] = dict(exp_str=exp_str, idx_list=list(filtered_df.index),
                                                  len_idx=filtered_df.shape[0])
            if exp_dict_max['len_idx'] < filtered_df.shape[0]:
                if verbose:
                    print("found a better explanation for the cluster")
                exp_dict_max = {'exp_num': num_example_iter, 'exp_str': exp_str, 'idx_list': list(filtered_df.index),
                                'len_idx': filtered_df.shape[0]}
            if verbose:
                print("finished looking at sample num:", num_example_iter)

        index_covered_so_far = set(index_covered_so_far).union(set(exp_dict_max['idx_list']))
        if verbose:
            print(exp_dict_max['exp_str'])
            print("covers {}% of the cluster".format(str(round(len(index_covered_so_far) / size_cluster, 2) * 100)))
        cluster_num_index = list(set(all_cluster_idx) - set(index_covered_so_far))
        cluster_exp.append(exp_dict_max['exp_str'])
        if len(cluster_num_index) < (1 - threshold) * size_cluster:
            if verbose:
                print("reached coverage of cluster after {} iterations.".format(iteration))
            break
    if verbose:
        print("finished finding anchors to cluster num {num} with coverage of {cov}% after {iterations} iterations"
          .format(num=cluster_num, cov=round(len(index_covered_so_far) / size_cluster, 2) * 100, iterations=iteration))
    if keep_dict:
        return cluster_exp, exp_dict
    else:
        return cluster_exp
