def compare_models(results):
    """Compara los cuatro modelos creados y muestra sus métricas de evaluación."""
    print("\n===== COMPARACIÓN DE LOS 4 MODELOS =====\n")

    # Crear DataFrames para cada modelo con sus mejores resultados
    model_comparison = []

    for model_name, model_results in results.items():
        # Para modelos incrementales, identificar el número óptimo de características (basado en AUC)
        best_idx = np.argmax(model_results['auc'])
        
        model_comparison.append({
            'Modelo': model_name,
            'Num_Características': model_results['num_features'][best_idx],
            'Train_Accuracy': model_results['train_accuracy'][best_idx],
            'Train_Precision': model_results['train_precision'][best_idx],
            'Train_Recall': model_results['train_recall'][best_idx],
            'Train_F1': model_results['train_f1'][best_idx],
            'Train_AUC': model_results['train_auc'][best_idx],
            'Test_Accuracy': model_results['accuracy'][best_idx],
            'Test_Precision': model_results['precision'][best_idx],
            'Test_Recall': model_results['recall'][best_idx],
            'Test_F1': model_results['f1'][best_idx],
            'Test_AUC': model_results['auc'][best_idx],
            'Tiempo_Entrenamiento': model_results['train_time'][best_idx],
            'Caract_Positivas': model_results['positive_features'][best_idx],
            'Caract_Negativas': model_results['negative_features'][best_idx]
        })

    # Crear DataFrame de comparación
    comparison_df = pd.DataFrame(model_comparison)

    # Añadir descripciones más claras
    comparison_df['Descripción'] = [
        'Incremental Completo',
        'Incremental Sin Obvias',
        'Fijo Todas', 
        'Fijo Sin Obvias'
    ]

    # Mostrar tabla de comparación para datos de entrenamiento
    print("Comparación de modelos - Métricas en conjunto de ENTRENAMIENTO:")
    display(comparison_df[['Descripción', 'Num_Características', 'Train_Accuracy', 'Train_Precision', 'Train_Recall', 'Train_F1', 'Train_AUC']])
    
    # Mostrar tabla de comparación para datos de prueba
    print("\nComparación de modelos - Métricas en conjunto de PRUEBA:")
    display(comparison_df[['Descripción', 'Num_Características', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC']])
    
    return comparison_df
