# AI write a machine learning cicle like the follwing:
# // PART I: DATA ANALYSIS (Data Curators & Feature Analysts)
# FUNCTION Step_1_Data_Structuring(raw_tick_data):
#     // Avoid standard Time Bars; use Information-Driven Bars to synchronize with market activity [3, 4].
#     bars = Generate_Information_Driven_Bars(raw_tick_data, type="Dollar_Bars") 
#     RETURN bars

# FUNCTION Step_2_Feature_Engineering(bars):
#     // Use Fractional Differentiation to reach stationarity while preserving maximum memory [5-7].
#     d_star = Find_Minimum_D_For_Stationarity(bars)
#     stationary_features = Apply_FracDiff(bars, d_star)
    
#     // Orthogonalize features to mitigate linear substitution effects (multi-collinearity) [8, 9].
#     orthogonal_features = Apply_PCA(stationary_features)
#     RETURN orthogonal_features

# FUNCTION Step_3_Labeling_and_Weighting(bars):
#     // Use path-dependent labels based on dynamic volatility thresholds [10, 11].
#     labels = Apply_Triple_Barrier_Method(bars, pt_sl_limits, vertical_barrier)
    
#     // Account for overlapping outcomes and non-IID data [12, 13].
#     concurrency = Compute_Concurrent_Labels(labels)
#     uniqueness = Compute_Average_Uniqueness(labels, concurrency)
    
#     // Weight samples by uniqueness and absolute return attribution [14, 15].
#     sample_weights = Compute_Sample_Weights(labels, uniqueness)
#     RETURN labels, sample_weights

# For cross validation use PurgedKFold
# For feature engineering use all features used in create_features

# AI!

