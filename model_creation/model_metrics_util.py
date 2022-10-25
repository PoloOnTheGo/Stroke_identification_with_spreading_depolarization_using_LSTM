import pandas as pd
import datetime as dt
import os


def get_metrics(full_df):
    tp = len(full_df[(full_df.test_y == 1) & (full_df.predicted == 1)])
    fp = len(full_df[(full_df.test_y == 0) & (full_df.predicted == 1)])
    tn = len(full_df[(full_df.test_y == 0) & (full_df.predicted == 0)])
    fn = len(full_df[(full_df.test_y == 1) & (full_df.predicted == 0)])
    p = len(full_df[full_df.test_y == 1])
    n = len(full_df[full_df.test_y == 0])
    pp = len(full_df[full_df.predicted == 1])
    pn = len(full_df[full_df.predicted == 0])
    tpr = tp / p
    fnr = fn / p
    ppv = tp / pp
    tnr = tn / n
    fpr = fp / n
    return {
        'PPV': ppv,
        'TPR': tpr,
        'TNR': tnr
    }


def get_block_no(df, diff_col=None, block_col='block_no', diff_len=1, start_idx=1):
    delete_diff = False
    block_idx = start_idx
    if diff_col is None:
        df['diff_col'] = df.index.to_series().diff()
        diff_col = 'diff_col'
        delete_diff = True
    # print(df[diff_col].dtype)
    if df[diff_col].dtype == 'float64':
        comparator = diff_len
    else:
        comparator = dt.timedelta(minutes=diff_len)
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        # print(row)
        if row[diff_col] > comparator:
            block_idx += 1
        df.loc[row.name, block_col] = block_idx
    if delete_diff:
        df.drop(columns=[diff_col, ], inplace=True)
    return df


def get_detection_data(df, col, cmp_col, threshold=0, verbose=0):
    correct = 0
    wrong = 0
    overlaps = []
    for sd_no in df[col].unique():
        single_sd = df[df[col] == sd_no]
        if len(single_sd) > threshold:
            overlap = len(single_sd[single_sd[cmp_col] == 1])
            if verbose > 0:
                print(f'{sd_no}: len: {len(single_sd)}, overlap: {overlap}')
            if overlap > 0:
                correct += 1
                overlaps.append(overlap)
            else:
                wrong += 1
    return {
        'correct': correct,
        'wrong': wrong,
        'min_overlap': min(overlaps, default=0),
        'max_overlap': max(overlaps, default=0)
    }


def overall_model_analysis(folder, file_name):
    results = {}
    for path_main in sorted(os.scandir(folder), key=lambda ent: ent.name):
        pat_id = path_main.name[:]
        print(pat_id)
        sub_folder = str(folder + '/' + path_main.name)
        for path in os.scandir(sub_folder):
            if path.name.endswith('csv'):
                df = pd.read_csv(path.path, index_col=0, header=0, names=['channels', 'test_y', 'predicted'])
                metrices = get_metrics(df)
                sddf = df[df.test_y == 1].copy()
                sddf = get_block_no(sddf, block_col='sd_no')
                try:
                    detected = get_detection_data(sddf, 'sd_no', 'predicted')
                    sd_no = sddf.sd_no.max()
                except (KeyError, ValueError) as e:
                    print(f'{pat_id}: {e}')
                    detected = {'correct': 0, 'wrong': 0, 'min_overlap': 0, 'max_overlap': 0}
                    sd_no = 0
                det_df = df[df.predicted == 1].copy()
                det_df = get_block_no(det_df, block_col='dsd_no')
                detection_metrics = get_detection_data(det_df, 'dsd_no', 'test_y', 2)
                results[pat_id] = {
                    'PPV': metrices['PPV'],
                    'TPR': metrices['TPR'],
                    'TNR': metrices['TNR'],
                    'sd_block_count': sd_no,
                    'detected': detected['correct'],
                    'missed': detected['wrong'],
                    'correct_hits': detection_metrics['correct'],
                    'wrong_hits': detection_metrics['wrong'],
                    'min_overlap': detected['min_overlap'],
                }
    res_df = pd.DataFrame.from_dict(results, orient='index')
    res_df['ppv'] = res_df.correct_hits / (res_df.correct_hits + res_df.wrong_hits)
    res_df['tpr'] = res_df.detected / res_df.sd_block_count
    res_df.to_csv(file_name)


