import os
import pdb
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

class AnalyseCorr():
    def __init__(self):
        pass
    
    def train_corr(self, fold_num, train_path):
        train_pred_df = pd.read_csv(train_path)
        print(train_pred_df)
        ax1 = train_pred_df.plot.scatter(x='Score', y='Pred Score', c='C0', marker='.', s=80, lw=0, alpha=0.7)
        plt.title('Scatter Plot for Scores Distribution in ' + fold_num + ' Training Dataset')
        plt.savefig('./datainfo/plot/scatter_trainpred_corr.png', dpi=300)

    def test_corr(self, fold_num, test_path):
        test_pred_df = pd.read_csv(test_path)
        print(test_pred_df)
        ax1 = test_pred_df.plot.scatter(x='Score', y='Pred Score', c='C0', marker='.', s=80, lw=0, alpha=0.7)
        plt.title('Scatter Plot for Scores Distribution in ' + fold_num + ' Test Dataset')
        plt.savefig('./datainfo/plot/scatter_testpred_corr.png', dpi=300)

    def pred_result(self, fold_n, epoch_name):
        ### TRAIN PRED JOINTPLOT
        train_pred_df = pd.read_csv('./datainfo/result/tsgnn/' + epoch_name + '/BestTrainingPred.txt')
        sns.set_style('whitegrid')
        sns.jointplot(data=train_pred_df, x='Score', y='Pred Score', size=10, kind='reg')
        train_pearson = train_pred_df.corr(method='pearson')['Pred Score'][0]
        plt.legend(['Training Pearson =' + str(train_pearson)])
        plt.savefig('./datainfo/plot/trainpred_corr.png', dpi=300)
        ### TEST PRED JOINTPLOT
        test_pred_df = pd.read_csv('./datainfo/result/tsgnn/' + epoch_name + '/BestTestPred.txt')
        comb_testpred_df = pd.read_csv('./datainfo/filtered_data/split_input_' + str(fold_n) + '.csv')
        comb_testpred_df['Pred Score'] = list(test_pred_df['Pred Score'])
        comb_testpred_df.to_csv('./datainfo/result/tsgnn/' + epoch_name + '/combine_testpred.csv', index=False, header=True)
        sns.set_style('whitegrid')
        sns.jointplot(data=comb_testpred_df, x='Score', y='Pred Score', size=10, kind='reg')
        test_pearson = test_pred_df.corr(method='pearson')['Pred Score'][0]
        plt.legend(['Test Pearson =' + str(test_pearson)])
        plt.savefig('./datainfo/plot/testpred_corr.png', dpi=300)
        ### HISTOGRAM
        hist = test_pred_df.hist(column=['Score', 'Pred Score'], bins=20)
        plt.savefig('./datainfo/plot/testpred_hist.png', dpi=300)
        ### BOX PLOT
        testpred_df = comb_testpred_df[['Cell Line Name', 'Pred Score']]
        testpred_df['Type'] = ['Prediction Score']*testpred_df.shape[0]
        testpred_df = testpred_df.rename(columns={'Pred Score': 'Drug Score'})
        test_df = comb_testpred_df[['Cell Line Name', 'Score']]
        test_df['Type'] = ['Input Score']*test_df.shape[0]
        test_df = test_df.rename(columns={'Score': 'Drug Score'})
        comb_score_df = pd.concat([testpred_df, test_df])
        a4_dims = (20, 15)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.set_context('paper')
        sns.boxplot(ax=ax, x='Cell Line Name', y='Drug Score', hue='Type', data=comb_score_df)
        plt.xticks(rotation = 90, ha = 'right')
        plt.savefig('./datainfo/plot/testpred_compare_cell_line_boxplot.png', dpi=600)
        # plt.show()

    def comparison(self):
        labels = ['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold', 'Average']
        gcn_decoder_test = [0.472962, 0.523855, 0.542053, 0.576385, 0.5266995, 0.528391]
        gat_decoder_test = [0.438654, 0.500450, 0.55021, 0.562448, 0.552173, 0.520787]
        tsgnn_decoder_test = [0.576044, 0.55865, 0.602628, 0.627989, 0.589877, 0.591038]

        x = np.arange(len(labels))
        width = 0.25 
        dnn = plt.bar(x - 1*width, gcn_decoder_test, width, label='GCN Decoder')
        gat = plt.bar(x, gat_decoder_test, width, label='GAT Decoder')
        tsgnn = plt.bar(x + 1*width, tsgnn_decoder_test, width, label='TSGNN')
        plt.ylabel('Pearson Correlation')
        # plt.title('Pearson Correlation Comparison For 3 GNN Models')

        plt.ylim(0.0, 0.8)
        plt.xticks(x, labels=labels)
        plt.legend()
        plt.show()


# fold_num = '5th'
# epoch_name = 'epoch_60_4'
# train_path = './datainfo/result/tsgnn/' + epoch_name + '/BestTrainingPred.txt'
# AnalyseCorr().train_corr(fold_num=fold_num, train_path=train_path)
# test_path = './datainfo/result/tsgnn/' + epoch_name + '/BestTestPred.txt'
# AnalyseCorr().test_corr(fold_num=fold_num, test_path=test_path)

# fold_n = 5
# epoch_name = 'epoch_60_4'
# AnalyseCorr().pred_result(fold_n=fold_n, epoch_name=epoch_name)

AnalyseCorr().comparison()