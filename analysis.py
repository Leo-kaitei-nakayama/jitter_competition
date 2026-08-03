"""
Analyze per-sample CD/P2S scores from evaluate.py's --save_csv output.

Usage:
    python analyze.py --csv ./per_sample_scores.csv --worst_k 8
"""
import argparse
import pandas as pd

# Best-effort ShapeNet synset -> human name mapping for readability.
# (Not exhaustive / not guaranteed for every ID -- just for display.)
SYNSET_NAMES = {
    '02691156': 'airplane',
    '02871439': 'bookshelf',
    '02876657': 'bottle',
    '02942699': 'camera',
    '03046257': 'clock',
    '03642806': 'laptop',
    '04074963': 'remote_control',
    '04256520': 'sofa',
    '04330267': 'stove',
    '04379243': 'table',
    '04401088': 'phone',
    '04468005': 'watercraft',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='./per_sample_scores.csv')
    parser.add_argument('--meta_csv', type=str, default='./eval_meta.csv',
                         help='eval_meta.csv from make_eval_set.py (has noise_std per sample); pass "" to skip')
    parser.add_argument('--worst_k', type=int, default=8)
    parser.add_argument('--metric', type=str, default='cd_score',
                         choices=['cd_score', 'p2s_score'],
                         help='which score to sort worst-samples by')
    args = parser.parse_args()

    # category is a synset ID like "04379243" -- read as string or the
    # leading zero gets silently dropped by pandas' type inference.
    df = pd.read_csv(args.csv, dtype={'category': str, 'model_id': str})
    df['category_name'] = df['category'].map(lambda c: SYNSET_NAMES.get(c, c))

    have_noise = False
    if args.meta_csv:
        try:
            meta = pd.read_csv(args.meta_csv, dtype={'category': str, 'model_id': str})
            df = df.merge(meta[['key', 'noise_std']], on='key', how='left')
            have_noise = df['noise_std'].notna().any()
        except FileNotFoundError:
            print(f'(no meta CSV found at {args.meta_csv} -- skipping noise-level breakdown)\n')

    pd.set_option('display.width', 140)
    pd.set_option('display.max_rows', None)

    print('=' * 70)
    print(f'Overall ({len(df)} samples)')
    print('=' * 70)
    print(df[['cd_score', 'p2s_score']].agg(['mean', 'std', 'min', 'max']).to_string())

    print()
    print('=' * 70)
    print('By category')
    print('=' * 70)
    grp = df.groupby('category_name')[['cd_score', 'p2s_score']].agg(['mean', 'std', 'count'])
    grp = grp.sort_values(('cd_score', 'mean'))
    print(grp.to_string())

    cols = ['key', 'category_name', 'cd_score', 'p2s_score']
    if have_noise:
        cols.append('noise_std')

    print()
    print('=' * 70)
    print(f'Worst {args.worst_k} samples by {args.metric}')
    print('=' * 70)
    worst = df.sort_values(args.metric).head(args.worst_k)
    print(worst[cols].to_string(index=False))

    print()
    print('=' * 70)
    print(f'Best {args.worst_k} samples by {args.metric}')
    print('=' * 70)
    best = df.sort_values(args.metric, ascending=False).head(args.worst_k)
    print(best[cols].to_string(index=False))

    if have_noise:
        print()
        print('=' * 70)
        print('By noise level (quartile bins)')
        print('=' * 70)
        df['noise_bin'] = pd.qcut(df['noise_std'], q=4, duplicates='drop')
        bin_grp = df.groupby('noise_bin', observed=True)[['cd_score', 'p2s_score']].agg(['mean', 'std', 'count'])
        print(bin_grp.to_string())

        print()
        print('=' * 70)
        print('Correlation: noise_std vs scores')
        print('=' * 70)
        corr_cd = df['noise_std'].corr(df['cd_score'])
        corr_p2s = df['noise_std'].corr(df['p2s_score'])
        print(f'  corr(noise_std, cd_score)  = {corr_cd:+.3f}')
        print(f'  corr(noise_std, p2s_score) = {corr_p2s:+.3f}')
        print('  (positive = higher noise tends to score BETTER -- a sign the')
        print('   scoring formula rewards big improvements more than genuine quality)')


if __name__ == '__main__':
    main()