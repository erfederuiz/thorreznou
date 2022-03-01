# FUNCION 9
def visualizeME_scores_models(y_true,models,bin_multi_classifier,vis_pallete='tab10',save=True):
    ''' 
    This is a function that allows you to quickly identify the best metrics from your Machine Learning models whether is binary or non binary target
    ### Parameters(4):
        * y_true: -- `array_like` 
            True labels.
        * models: 
            here you should put a diccionary that contains for keys, the model names (as a str) and for values their predictions to compare with true labels.
             This will help to join all the metrics in one data frame
             EXAMPLE: {'Linear Regression':preds_multi,'KNN Classifier':preds_multi2}
        * vis_pallete:
            Seaborn color pallete we have selected before
        * bin_multi_classifier:
            True if it is a binomial model (0,1) and  False if it is multicategory
        * save=False -- `bool` 
            Saves plot and metrics (if metrics=True) to disk. If title='' default is 'classifier'.
    '''
    # extracting values and keys from models dictionary:
    df = pd.DataFrame()
    keys_modelo = list(models.keys())
    values_modelo = list(models.values())   
    
    #If we have selected a binomial classifier then:
    if bin_multi_classifier == True:
        #Elaborating a Data Frame that will give you 5 metrics from you ML Model:
        for i,j in enumerate(models):
            metrics_df = pd.DataFrame({str(j): [float(f'{accuracy_score(y_true, values_modelo[i]):.10f}'),
                                            float(f'{precision_score(y_true, values_modelo[i]):.10f}'),
                                            float(f'{recall_score(y_true, values_modelo[i]):.10f}'),
                                            float(f'{f1_score(y_true, values_modelo[i]):.10f}'),
                                            float(f'{roc_auc_score(y_true, values_modelo[i]):.10f}')]},
                                index=[['Accuracy: (TP + TN) / TOTAL',
                                           'Precision: TP / (TP + FP)',
                                           'Recall: TP / (TP + FN)',
                                           'F1: harmonic mean (accuracy, recall)',
                                           'ROC AUC']])
            # Here we are joining all the dataframe metrics from all the models we have selected:
            if i == 0:
                df=metrics_df
            else:
                df = df.join(other=metrics_df)

        #FINDING METRICS
        for y in range(len(df.index)):
            #Finding the best Accuracy:
            if y == 0:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_acc = df.columns[i]  

                # Setting a bar plot to see the best Accuracy:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf1= sns.barplot(x=df.columns,y=df.iloc[y],palette=vis_pallete)
                graf1.set_xlabel('Metrics from Models',fontsize = 12)
                graf1.set_title('Accuracy',fontsize=15,color='blue')
                for i in graf1.patches:
                    graf1.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best Accuracy value is '   + str(df.iloc[y].max())+' from Model: '+ str(best_acc))

                # Saving Bar plot
                if save == True:
                    path=os.path.join('visualizeME_Accuracy_barplot' + '.' + 'png')
                    plt.savefig(path, format='png', dpi=300)
                plt.show()
            
            #Finding the best Precision:
            elif y == 1:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_prec = df.columns[i]    
            

                # Setting a bar plot to see the best Precision:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf2= sns.barplot(x=df.columns,y=df.iloc[y],palette=vis_pallete)
                graf2.set_xlabel('Metrics from Models',fontsize = 12)
                graf2.set_title('Precision',fontsize=15,color='blue')
                for i in graf2.patches:
                    graf2.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best Precision value is '   + str(df.iloc[y].max())+' from Model: '+ best_prec)
                
                
                # Saving Bar plot
                if save == True:
                    name = 'visualizeME_Precision_barplot' 
                    path=os.path.join(name + '.' + 'jpg')
                    plt.savefig(path, format='jpg', dpi=300)
                plt.show()

            #Finding the best Recall:
            elif y == 2:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_rec = df.columns[i]    

                
                # Setting a bar plot to see the best Recall:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf3= sns.barplot(x=df.columns,y=df.iloc[y],palette=vis_pallete)
                graf3.set_xlabel('Metrics from Models',fontsize = 12)
                graf3.set_title('Recall',fontsize=15,color='blue')
                for i in graf3.patches:
                    graf3.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best Recall value is '   + str(df.iloc[y].max())+' from Model: '+ best_rec)
                
                # Saving Bar plot
                if save == True:
                    name = 'visualizeME_Recall_barplot'
                    path=os.path.join(name + '.' + 'jpg')
                    plt.savefig(path, format='jpg', dpi=300)
                plt.show()
            
            #Finding the best F1 score:
            elif y == 3:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_f1 = df.columns[i]    
            

                # Setting a bar plot to see the best F1 Score:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf4= sns.barplot(x=df.columns,y=df.iloc[y],palette=vis_pallete)
                graf4.set_xlabel('Metrics from Models',fontsize = 12)
                graf4.set_title('F1 Scores',fontsize=15,color='blue')
                for i in graf4.patches:
                    graf4.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best F1 Score value is '   + str(df.iloc[y].max())+' from Model: '+ best_f1)
                
                # Saving Bar plot
                if save:
                    name = 'visualizeME_F1Score_barplot' + '.png'
                    path=os.path.join(name + '.' + 'png')
                    plt.savefig(path, format='png', dpi=300)
                plt.show()
            
            #Finding the best ROC:
            elif y == 4:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_roc = df.columns[i]    
            
                # Setting a bar plot to see the best ROC AUC Score:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf5= sns.barplot(x=df.columns,y=df.iloc[y],palette= vis_pallete)
                graf5.set_xlabel('Metrics from Models',fontsize = 12)
                graf5.set_title('ROC / AUC',fontsize=15,color='blue')
                for i in graf5.patches:
                    graf5.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best ROC / AUC value is '   + str(df.iloc[y].max())+' from Model: '+ best_prec)
                
                # Saving Bar plot
                if save == True:
                    name = 'visualizeME_ROC_AUC_barplot' + '.png'
                    #path=os.path.join(name + '.' + 'png')
                    plt.savefig(name,format='png', dpi=300)
                plt.show()


    # bin_multi_classifier is False ->>>>> Multicategorical:
    else:
        for i,j in enumerate(models):
            report = classification_report(y_multi,values_modelo[i])
            report = [line.split(' ') for line in report.splitlines()]

            header = [x.upper()+' : '+keys_modelo[i] for x in report[0] if x!='']

            index = []
            values = []

            for row in report[1:-5]:
                row = [value for value in row if value!='']
                if row!=[]:
                    index.append(row[0].upper())
                    values.append(row[1:])

            index.append('ACCURACY')
            values.append(['-', '-'] + [x for x in report[-3] if x != ''][-2:])
            index.append('MACRO AVG.')
            values.append([x for x in report[-2] if x != ''][-4:])
            index.append('WEIGHTED AVG.')
            values.append([x for x in report[-1] if x != ''][-4:])

            metrics_df = pd.DataFrame(data=values, columns=header, index=index)
            if i == 0:
                df=metrics_df
            elif i == 1:
                df = df.join(other=metrics_df)
            else:
                df = df.join(other=metrics_df,) 
        
        #Creating a Data Frame with the Accuracy metrics from all the models:
        acc_list = []
        
        for i in range(2,len(df.loc['ACCURACY']),4):
                acc_list.append(float(df.loc['ACCURACY'][i]))
        acc_dic={}
        for j,k in enumerate(models):
            acc_dic[k] = acc_list[j]
        
        acc = pd.DataFrame(data= acc_dic,index=keys_modelo)
    
        # Setting a BarPlot to show the Best model Accuracy:
        num=len(acc.index)
        plt.figure(figsize=(num*1.9,4))
        graf1= sns.barplot(data=acc,palette=vis_pallete)
        graf1.set_xlabel('ML Models - Results',fontsize = 12)
        graf1.set_title('Accuracy',fontsize=15,color='blue')
        for i in graf1.patches:
            graf1.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
        if save:
            name = 'visualizeME_Accuracy_barplot' + '.png'
            path=os.path.join(name + '.' + 'png')
            plt.savefig(path, format='png', dpi=300)
        plt.show()
    
    # Saving Metrics
    if save == True:
        name = 'visualizeME_df_ML_metrics' + '.csv'
        df.to_csv(name, header=True)
