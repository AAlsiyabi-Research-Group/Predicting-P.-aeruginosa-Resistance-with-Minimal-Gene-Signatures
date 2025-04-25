import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import math

from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.callbacks import ConsecutiveStopping

#load resistance data
resistance_data = pd.read_excel('pseudomonas_MIC.xlsx', index_col=0)

#load gene expression data
expression = pd.read_excel('pseudomonas_expression-m.xlsx','Sheet1')
expression = expression.set_index('Gene')

X = expression
y = resistance_data.reindex(X.index)

#remove all null values
y_nullRemoved = y.dropna()
X_yNullRemoved = X.reindex(y_nullRemoved.index)

#removing null values from specific columns
y_TOB = y['TOB'].dropna()
y_TOB = pd.to_numeric(y_TOB, downcast='integer')
X_TOB = X.reindex(y_TOB.index)

y_CAZ = y['CAZ'].dropna()
y_CAZ = pd.to_numeric(y_CAZ, downcast='integer')
X_CAZ = X.reindex(y_CAZ.index)

y_CIP = y['CIP'].dropna()
y_CIP = pd.to_numeric(y_CIP, downcast='integer')
X_CIP = X.reindex(y_CIP.index)

y_MNM = y['MNM'].dropna()
y_MNM = pd.to_numeric(y_MNM, downcast='integer')
X_MNM = X.reindex(y_MNM.index)

y_COL = y['Colistin'].dropna()
y_COL = pd.to_numeric(y_COL, downcast='integer')
X_COL = X.reindex(y_COL.index)


io_folder = 'batch_parallel_output'
iterations = 1
feature_sample = [40]
scoring_list = ['f1']
ab = 'CAZ'
algo = 'LR'
ratio = '91'
processors = 25
processes = 1000
pop_size=1000
gen_count=300


if ab == 'CAZ':
    print(f'X_CAZ: {X_CAZ.shape}, y_CAZ: {y_CAZ.shape}')
    X = X_CAZ
    y = y_CAZ
elif ab == 'TOB':
    print(f'X_TOB: {X_TOB.shape}, y_TOB: {y_TOB.shape}')
    X = X_TOB
    y = y_TOB
elif ab == 'CIP':
    print(f'X_CIP: {X_CIP.shape}, y_CIP: {y_CIP.shape}')
    X = X_CIP
    y = y_CIP
elif ab == 'MNM':
    print(f'X_MNM: {X_MNM.shape}, y_MNM: {y_MNM.shape}')
    X = X_MNM
    y = y_MNM
elif ab == 'COL':
    print(f'X_COL: {X_COL.shape}, y_COL: {y_COL.shape}')
    X = X_COL
    y = y_COL


cross_probability = float('0.'+ratio[0])
mut_probability = float('0.'+ratio[1])


def run_generation(parameter):
    
    gen_start_time = time.time()
    print(f'X: {X.shape}, y: {y.shape}')
    X_columns = X.columns
  
    callback = ConsecutiveStopping(generations=10, metric='fitness_max')

    file_name_counter = parameter

    for metric in scoring_list:
        
        path = str(os.getcwd())+f'/{io_folder}/pseudomonas_scoring-{metric}_{file_name_counter}.xlsx'
        writer = pd.ExcelWriter(path, engine = 'openpyxl')
                      
        for item in feature_sample:
            
            if algo == 'SVM':
                clf = svm.SVC()
            elif algo == 'LR':
                clf = LogisticRegression()
                
            top_iteration_df = pd.DataFrame()
            top_iteration_dict = []
            gene_count = {key: 0 for key in X_columns}

            for iteration in range(iterations):
                print(f"===== AB: {ab} *** Algo: {algo} *** Scoring: {metric} *** Max Feature Size: {item} *** Iteration: {iteration+1} *** Ratio: {ratio} =====")
                evolved_estimator = GAFeatureSelectionCV(
                    estimator=clf,
                    cv=4,
                    scoring=metric,
                    population_size=pop_size,
                    max_features=item,
                    generations=gen_count,
                    verbose=True,
                    keep_top_k=2,
                    elitism=True,
                    crossover_probability=cross_probability,
                    mutation_probability=mut_probability,
                    criteria='max',
                )

                evolved_estimator.fit(X, y, callbacks=callback)
                features = evolved_estimator.support_
                best_features = evolved_estimator.best_features_
                feature_count = sum(best_features)
                feature_list = []
                for index,value in enumerate(best_features):
                    if value == True:
                        feature_list.append(X_columns[index])

                history_dict = evolved_estimator.history
                history_df = pd.DataFrame(history_dict)
                history_df = history_df.sort_values(by=['fitness_max'], ascending=False)

                top_iteration_row = history_df.iloc[0]
                top_iteration_row_dict = {'gen': int(top_iteration_row['gen']),'genes': str(feature_list), 'fitness': top_iteration_row['fitness'], \
                                          'fitness_std': top_iteration_row['fitness_std'], 'fitness_max': top_iteration_row['fitness_max'], \
                                          'fitness_min': top_iteration_row['fitness_min']}

                print(f'**************************{top_iteration_row_dict}******************************')

                feature_df = pd.DataFrame(feature_list)

                for feature in feature_list:
                    if feature in gene_count:
                        gene_count[feature] += 1
                gene_count_df = pd.DataFrame.from_dict(gene_count, orient='index', columns=['count'])
                gene_count_df = gene_count_df.sort_values(by=['count'], ascending=False)


            history_df.to_excel(writer, index=False, sheet_name=f'{item}-result')
            feature_df.to_excel(writer, index=False, sheet_name=f'{item}-best_features')

        writer.close()
        print(f"========= Generation Runtime: {round((time.time() - gen_start_time)/60,2)} minutes =========")


def run_parallel():
    
    process_count = [x for x in range(processes)]
    
    processor_count = processors

    for iteration in range(math.ceil(len(process_count)/processor_count)):
        print(iteration*processor_count)
        small_process_list = [x for x in range(iteration*processor_count, iteration*processor_count+processor_count,1) if x < len(process_count)]
        pool = Pool(processes=processor_count)
        pool.map(run_generation, small_process_list)
    
def combine_result():
    file_counter = 0
    file_list = os.listdir(io_folder)
    top_iteration_df = pd.DataFrame()
    top_iteration_dict = []
    gene_count = {key: 0 for key in X.columns}
    
    path = str(os.getcwd())+f'/{io_folder}/master#TPM#{ab}#({algo}-{feature_sample[0]}-{scoring_list[0]}-{ratio})#{processes}.xlsx'
    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    
    for file in file_list:
        file_size = os.path.getsize(f'{io_folder}/{file}')
        if os.path.isfile(f'{io_folder}/{file}'):
            if 'master' not in file and file_size != 0 and 'xlsx' in file:
                file_counter += 1
                result_df = pd.read_excel(f'{io_folder}/{file}', sheet_name=0)
                feature_df = pd.read_excel(f'{io_folder}/{file}', sheet_name=1)

                #result df creation
                top_iteration_row = result_df.iloc[0:1]
                gene_names = feature_df[0].tolist()
                top_iteration_row_dict = {'gen': top_iteration_row['gen'].values[0],'genes': gene_names, 'gene_count': len(gene_names),\
                                          'fitness': top_iteration_row['fitness'].values[0], 'fitness_std': top_iteration_row['fitness_std'].values[0], \
                                          'fitness_max': top_iteration_row['fitness_max'].values[0], 'fitness_min': top_iteration_row['fitness_min'].values[0]}
                
                top_iteration_dict.append(top_iteration_row_dict)
                

                #feature count df creation
                for gene in feature_df.itertuples():
                    feature = gene[1]
                    if feature in gene_count:
                        gene_count[feature] += 1
                os.remove(f'{io_folder}/{file}')
                
    top_iteration_df = pd.DataFrame(top_iteration_dict)
    top_iteration_df = top_iteration_df.sort_values(by=['fitness_max'], ascending=False)
    gene_count_df = pd.DataFrame.from_dict(gene_count, orient='index', columns=['count'])
    gene_count_df = gene_count_df.sort_values(by=['count'], ascending=False)
    print("**********************************************")
    print(top_iteration_df.head)
    print(gene_count_df.head())
    print("**********************************************")
    top_iteration_df.to_excel(writer, index=False, sheet_name='top_iteration_rows')
    gene_count_df.to_excel(writer, index=True, sheet_name='gene_count')
    writer.close()
    print(f"Total traversed files: {file_counter}")
    
# Driver code
if __name__ == '__main__':
    start_time = time.time()
    run_parallel()
    combine_result()
    print(f"Runtime: {round((time.time() - start_time)/60,2)} minutes")

