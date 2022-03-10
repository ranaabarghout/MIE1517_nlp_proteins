# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 18:48:55 2021

@author: ranam
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from scipy.stats import norm

sns.set(style='darkgrid', font_scale=1.2)


def main():
    EC_data = pd.read_csv('./data/results.csv')
    EC_data.sort_values(by=['Column1'], inplace=True, ascending = [True])

    EC_num_counts = []
    EC_num_counts=EC_data["Column1"].value_counts()
    EC_num_counts.sort_index(axis=0, inplace=True, ascending=True)
    EC_num_counts=EC_num_counts.to_frame()
    EC_num_counts.reset_index(level=0, inplace=True)
    EC_num_counts.rename(columns={'index': 'EC Number', 'Column1':'Count'}, inplace=True)
    EC_num_counts[['Level1', 'Level2', 'Level3', 'Level4']] = EC_num_counts['EC Number'].str.split('.', -1, expand=True)

    Level4_count = len(EC_num_counts)-(EC_num_counts['Level4'].isna().sum())
    Level3_count = len(EC_num_counts)-(EC_num_counts['Level3'].isna().sum())
    Level2_count = len(EC_num_counts)-(EC_num_counts['Level2'].isna().sum())
    Level1_count = len(EC_num_counts)-(EC_num_counts['Level1'].isna().sum())

    print('Number of ECs available with ATLEAST level 1, 2, 3, and 4, respectively: ', Level1_count, Level2_count, Level3_count, Level4_count)

    by_EC_level1 = EC_num_counts.groupby(by='Level1').agg({'Count': 'sum'})
    by_EC_level1.reset_index(level=0, inplace=True)

    ax = by_EC_level1.plot.bar(x='Level1', y='Count', rot=0)
    plt.xlabel('Enzyme Class')
    plt.ylabel('Count')
    # plt.xticks(ticks=[1, 2, 3, 4, 5, 6, 7],labels=['Oxireductase', 'Transferases', 'Hydrolases', 'Lyases', 'Isomerases', 'Ligases', 'Translocases'])

    EC_data_species_count = EC_data.groupby(by='Column3').count()
    EC_data_sequence_count = EC_data.groupby(by='Column2').count()
    x_ticks = np.arange(0,40,41)
    ax = sns.displot(data=EC_data_sequence_count, x='Column1', kind='hist', aspect=1.4, log_scale=(False, True), bins=40)
    plt.title('Repeated Sequences')
    plt.ylabel('Count')
    plt.xlabel('Number of Repeated Sequences')

    postcdhit_data = pd.read_csv('./data/results-fasta-80percent.csv')
    postcdhit_data.loc[-1] = ['>1']
    postcdhit_data.index = postcdhit_data.index + 1
    postcdhit_data = postcdhit_data.sort_index()

    new_post_data = pd.DataFrame({'Number': postcdhit_data['>1'].iloc[::2].values, 'sequence': postcdhit_data['>1'].iloc[1::2].values})
    new_post_data['Number'] = new_post_data['Number'].map(lambda x: x.lstrip('>'))

    reader = csv.reader(open('./data/ECnum-dictionary.csv', 'r'))
    d = {}
    for row in reader:
        k, v = row
        d[k] = v

    new_post_data_EC = new_post_data.replace({'Number': d})

    #### SAME ANALYSIS ON POST SIMILARITY DATA
    post_data_EC = new_post_data_EC.sort_values(by=['Number'], ascending = [True])

    EC_num_counts_post = []
    EC_num_counts_post=post_data_EC["Number"].value_counts()
    EC_num_counts_post.sort_index(axis=0, inplace=True, ascending=True)
    EC_num_counts_post=EC_num_counts_post.to_frame()
    EC_num_counts_post.reset_index(level=0, inplace=True)
    EC_num_counts_post.rename(columns={'index': 'EC Number', 'Number':'Count'}, inplace=True)
    EC_num_counts_post[['Level1', 'Level2', 'Level3', 'Level4']] = EC_num_counts_post['EC Number'].str.split('.', -1, expand=True)

    Level4_count_post = len(EC_num_counts_post)-(EC_num_counts_post['Level4'].isna().sum())
    Level3_count_post = len(EC_num_counts_post)-(EC_num_counts_post['Level3'].isna().sum())
    Level2_count_post = len(EC_num_counts_post)-(EC_num_counts_post['Level2'].isna().sum())
    Level1_count_post = len(EC_num_counts_post)-(EC_num_counts_post['Level1'].isna().sum())

    print('Number of ECs available with ATLEAST level 1, 2, 3, and 4 POST PROCESSING, respectively: ', Level1_count_post, Level2_count_post, Level3_count_post, Level4_count_post)

    by_EC_level1_post = EC_num_counts_post.groupby(by='Level1').agg({'Count': 'sum'})
    by_EC_level1_post.reset_index(level=0, inplace=True)

    ax = by_EC_level1_post.plot.bar(x='Level1', y='Count', rot=0)
    plt.xlabel('Enzyme Class')
    plt.ylabel('Count')
    # plt.xticks(ticks=[1, 2, 3, 4, 5, 6, 7],labels=['Oxireductase', 'Transferases', 'Hydrolases', 'Lyases', 'Isomerases', 'Ligases', 'Translocases'])


if __name__ == '__main__':
    main()










