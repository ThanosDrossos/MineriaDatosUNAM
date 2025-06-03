from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
import math
import numpy as np
import pandas as pd
import time

def build_incremental_naivebayes(df, feature_list, target_col, train_indices, test_indices):
    """Construye modelos Naive Bayes incrementales añadiendo características una a una."""
    # Preparar dataframes de train y test
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    
    # Obtener probabilidad previa de obesidad en conjunto de entrenamiento
    prior_probability = df_train[target_col].mean()
    prior_odds = prior_probability / (1 - prior_probability) if prior_probability < 1 else 1.0
    log_prior = math.log(prior_odds) if prior_odds > 0 else 0
    
    print(f"Probabilidad previa en conjunto de entrenamiento: {prior_probability:.4f}")
    
    # Diccionario para almacenar rendimiento para cada incremento de características
    performance = {
        'num_features': [],
        'features': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'pr_auc': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'train_auc': [],
        'train_time': [],
        'positive_features': [],
        'negative_features': []
    }
    
    # Análisis incremental: añadimos características una por una
    for k in range(1, len(feature_list) + 1):
        start_time = time.time()
        
        # Seleccionar k primeras características
        selected_features = feature_list[:k]
        feature_names = [f['Feature'] for f in selected_features]
        
        # Registrar características positivas y negativas
        pos_features = sum(1 for f in selected_features if f['Epsilon'] > 0)
        neg_features = sum(1 for f in selected_features if f['Epsilon'] < 0)
        
        # Inicializar predicciones para train y test (log-odds ratio)
        train_scores = np.ones(len(df_train)) * log_prior
        test_scores = np.ones(len(df_test)) * log_prior
        
        # Calcular probabilidades para cada característica en el conjunto
        for feature_info in selected_features:
            feature = feature_info['Feature']
            value = feature_info['Value']
            is_threshold = feature_info['Is_Threshold']
            
            # Contar instancias positivas y negativas en train
            positive_instances = df_train[target_col].sum()
            negative_instances = len(df_train) - positive_instances
            
            # Crear condición
            if is_threshold:
                train_condition = df_train[feature] >= value
                test_condition = df_test[feature] >= value
            else:
                train_condition = df_train[feature] == value
                test_condition = df_test[feature] == value
            
            # Calcular probabilidades condicionales con suavizado de Laplace
            # P(X|C) - Probabilidad de la característica dado la clase
            count_with_C = ((df_train[target_col] == 1) & train_condition).sum() + 1
            count_with_not_C = (train_condition & (df_train[target_col] == 0)).sum() + 1
            
            P_X_given_C = count_with_C / (positive_instances + 2)
            P_X_given_not_C = count_with_not_C / (negative_instances + 2)
            
            # Calcular razón de log-verosimilitud
            log_ratio = math.log(P_X_given_C / P_X_given_not_C) if P_X_given_not_C > 0 else 0
            
            # Aplicar a pacientes que coinciden - tanto en train como en test
            train_scores[train_condition] += log_ratio
            test_scores[test_condition] += log_ratio
            
            # Para pacientes que no coinciden, aplicar la verosimilitud complementaria
            # P(not X|C) - Probabilidad de NO tener la característica dada la clase
            comp_count_with_C = positive_instances - ((df_train[target_col] == 1) & train_condition).sum() + 1
            comp_count_with_not_C = negative_instances - (train_condition & (df_train[target_col] == 0)).sum() + 1
            
            comp_P_X_given_C = comp_count_with_C / (positive_instances + 2)
            comp_P_X_given_not_C = comp_count_with_not_C / (negative_instances + 2)
            
            # Calcular razón de log-verosimilitud complementaria
            comp_log_ratio = math.log(comp_P_X_given_C / comp_P_X_given_not_C) if comp_P_X_given_not_C > 0 else 0
            
            # Aplicar a pacientes que no coinciden
            train_scores[~train_condition] += comp_log_ratio
            test_scores[~test_condition] += comp_log_ratio
        
        # Convertir log-odds a probabilidades
        train_probs = 1 / (1 + np.exp(-train_scores))
        test_probs = 1 / (1 + np.exp(-test_scores))
        
        # Convertir probabilidades a predicciones binarias (umbral 0.5)
        train_preds = (train_probs >= 0.5).astype(int)
        test_preds = (test_probs >= 0.5).astype(int)
        
        # Calcular métricas en conjunto de test
        acc = accuracy_score(df_test[target_col], test_preds)
        prec = precision_score(df_test[target_col], test_preds, zero_division=0)
        rec = recall_score(df_test[target_col], test_preds, zero_division=0)
        f1 = f1_score(df_test[target_col], test_preds, zero_division=0)
        auc_score = roc_auc_score(df_test[target_col], test_probs)
        
        # Calcular métricas en conjunto de train
        train_acc = accuracy_score(df_train[target_col], train_preds)
        train_prec = precision_score(df_train[target_col], train_preds, zero_division=0)
        train_rec = recall_score(df_train[target_col], train_preds, zero_division=0)
        train_f1 = f1_score(df_train[target_col], train_preds, zero_division=0)
        train_auc = roc_auc_score(df_train[target_col], train_probs)
        
        # Calcular Precision-Recall AUC para test
        precision_curve, recall_curve, _ = precision_recall_curve(df_test[target_col], test_probs)
        pr_auc_score = auc(recall_curve, precision_curve)
        
        # Tiempo de entrenamiento
        train_time = time.time() - start_time
        
        # Guardar resultados
        performance['num_features'].append(k)
        performance['features'].append(feature_names)
        performance['accuracy'].append(acc)
        performance['precision'].append(prec)
        performance['recall'].append(rec)
        performance['f1'].append(f1)
        performance['auc'].append(auc_score)
        performance['pr_auc'].append(pr_auc_score)
        performance['train_accuracy'].append(train_acc)
        performance['train_precision'].append(train_prec)
        performance['train_recall'].append(train_rec)
        performance['train_f1'].append(train_f1)
        performance['train_auc'].append(train_auc)
        performance['train_time'].append(train_time)
        performance['positive_features'].append(pos_features)
        performance['negative_features'].append(neg_features)
        
        # Mostrar progreso
        if k % 10 == 0 or k == 1 or k == len(feature_list):
            print(f"Modelo con {k} features")
            print(f"Train - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
            print(f"Test - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    
    return performance

def build_fixed_naivebayes(df, feature_list, target_col, train_indices, test_indices):
    """Construye un modelo Naive Bayes con un conjunto fijo de características."""
    # Preparar dataframes de train y test
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    
    # Obtener probabilidad previa de obesidad en conjunto de entrenamiento
    prior_probability = df_train[target_col].mean()
    prior_odds = prior_probability / (1 - prior_probability) if prior_probability < 1 else 1.0
    log_prior = math.log(prior_odds) if prior_odds > 0 else 0
    
    print(f"Probabilidad previa en conjunto de entrenamiento: {prior_probability:.4f}")
    
    # Medir tiempo
    start_time = time.time()
    
    # Usar todas las características disponibles
    selected_features = feature_list
    feature_names = [f['Feature'] for f in selected_features]
    
    # Registrar características positivas y negativas
    pos_features = sum(1 for f in selected_features if f['Epsilon'] > 0)
    neg_features = sum(1 for f in selected_features if f['Epsilon'] < 0)
    
    print(f"Usando {len(selected_features)} características: {pos_features} positivas, {neg_features} negativas")
    
    # Inicializar predicciones para train y test (log-odds ratio)
    train_scores = np.ones(len(df_train)) * log_prior
    test_scores = np.ones(len(df_test)) * log_prior
    
    # Calcular probabilidades para cada característica en el conjunto
    for feature_info in selected_features:
        feature = feature_info['Feature']
        value = feature_info['Value']
        is_threshold = feature_info['Is_Threshold']
        
        # Contar instancias positivas y negativas en train
        positive_instances = df_train[target_col].sum()
        negative_instances = len(df_train) - positive_instances
        
        # Crear condición
        if is_threshold:
            train_condition = df_train[feature] >= value
            test_condition = df_test[feature] >= value
        else:
            train_condition = df_train[feature] == value
            test_condition = df_test[feature] == value
        
        # Calcular probabilidades condicionales con suavizado de Laplace
        count_with_C = ((df_train[target_col] == 1) & train_condition).sum() + 1
        count_with_not_C = (train_condition & (df_train[target_col] == 0)).sum() + 1
        
        P_X_given_C = count_with_C / (positive_instances + 2)
        P_X_given_not_C = count_with_not_C / (negative_instances + 2)
        
        # Calcular razón de log-verosimilitud
        log_ratio = math.log(P_X_given_C / P_X_given_not_C) if P_X_given_not_C > 0 else 0
        
        # Aplicar a pacientes que coinciden - tanto en train como en test
        train_scores[train_condition] += log_ratio
        test_scores[test_condition] += log_ratio
        
        # Para pacientes que no coinciden, aplicar la verosimilitud complementaria
        comp_count_with_C = positive_instances - ((df_train[target_col] == 1) & train_condition).sum() + 1
        comp_count_with_not_C = negative_instances - (train_condition & (df_train[target_col] == 0)).sum() + 1
        
        comp_P_X_given_C = comp_count_with_C / (positive_instances + 2)
        comp_P_X_given_not_C = comp_count_with_not_C / (negative_instances + 2)
        
        # Calcular razón de log-verosimilitud complementaria
        comp_log_ratio = math.log(comp_P_X_given_C / comp_P_X_given_not_C) if comp_P_X_given_not_C > 0 else 0
        
        # Aplicar a pacientes que no coinciden
        train_scores[~train_condition] += comp_log_ratio
        test_scores[~test_condition] += comp_log_ratio
    
    # Convertir log-odds a probabilidades
    train_probs = 1 / (1 + np.exp(-train_scores))
    test_probs = 1 / (1 + np.exp(-test_scores))
    
    # Convertir probabilidades a predicciones binarias (umbral 0.5)
    train_preds = (train_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)
    
    # Calcular métricas en conjunto de test
    acc = accuracy_score(df_test[target_col], test_preds)
    prec = precision_score(df_test[target_col], test_preds, zero_division=0)
    rec = recall_score(df_test[target_col], test_preds, zero_division=0)
    f1 = f1_score(df_test[target_col], test_preds, zero_division=0)
    auc_score = roc_auc_score(df_test[target_col], test_probs)
    
    # Calcular métricas en conjunto de train
    train_acc = accuracy_score(df_train[target_col], train_preds)
    train_prec = precision_score(df_train[target_col], train_preds, zero_division=0)
    train_rec = recall_score(df_train[target_col], train_preds, zero_division=0)
    train_f1 = f1_score(df_train[target_col], train_preds, zero_division=0)
    train_auc = roc_auc_score(df_train[target_col], train_probs)
    
    # Calcular Precision-Recall AUC para test
    precision_curve, recall_curve, _ = precision_recall_curve(df_test[target_col], test_probs)
    pr_auc_score = auc(recall_curve, precision_curve)
    
    # Tiempo de entrenamiento
    train_time = time.time() - start_time
    
    # Crear y retornar resultados
    performance = {
        'num_features': [len(feature_names)],
        'features': [feature_names],
        'accuracy': [acc],
        'precision': [prec],
        'recall': [rec],
        'f1': [f1],
        'auc': [auc_score],
        'pr_auc': [pr_auc_score],
        'train_accuracy': [train_acc],
        'train_precision': [train_prec],
        'train_recall': [train_rec],
        'train_f1': [train_f1],
        'train_auc': [train_auc],
        'train_time': [train_time],
        'positive_features': [pos_features],
        'negative_features': [neg_features]
    }
    
    print(f"Modelo con {len(feature_names)} features")
    print(f"Train - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
    print(f"Test - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    print(f"Tiempo de entrenamiento: {train_time:.2f} segundos")
    
    return performance