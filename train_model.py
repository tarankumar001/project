"""
Motor Model Training Script
Trains a Random Forest classifier to predict motor condition based on temperature and vibration.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import os

# Configuration
CSV_FILENAME = 'motor_data.csv'
MODEL_FILENAME = 'motor_model.pkl'

def load_data(filename):
    """
    Load data from CSV file.
    Returns: DataFrame with the data
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Error: {filename} not found. Please run data_logger.py first to collect data.")
        
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} rows from {filename}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def compute_vibration_rms(df):
    """
    Compute vibration RMS from accelerometer data.
    Formula: sqrt(ax^2 + ay^2 + az^2)
    """
    # Calculate RMS (Root Mean Square) of vibration
    df['vib_rms'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    print("Computed vibration RMS for all samples")
    return df

def prepare_features_and_target(df):
    """
    Prepare features (temperature, vib_rms) and target (label).
    Returns: X (features), y (target)
    """
    # Select features: temperature and vibration RMS
    X = df[['temperature', 'vib_rms']].copy()
    
    # Select target: label
    y = df['label'].copy()
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of classes: {len(y.unique())}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")
    
    return X, y

def train_model(X_train, y_train):
    """
    Train Random Forest Classifier.
    Returns: Trained model
    """
    print("\nTraining Random Forest Classifier...")
    
    # Create Random Forest model
    # n_estimators: number of trees in the forest
    # random_state: for reproducibility
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics.
    """
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Print confusion matrix in a more readable format
    print("\nConfusion Matrix (Formatted):")
    classes = sorted(y_test.unique())
    print(" " * 12, end="")
    for cls in classes:
        print(f"Predicted {cls:>8}", end="")
    print()
    for i, cls in enumerate(classes):
        print(f"Actual {cls:>8}", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>12}", end="")
        print()
    
    # ROC Curve (only if binary classification)
    if len(classes) == 2:
        # For binary classification, use the positive class probability
        y_test_binary = (y_test == classes[1]).astype(int)
        fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba[:, 1])
        auc_score = roc_auc_score(y_test_binary, y_pred_proba[:, 1])
        
        print(f"\nROC AUC Score: {auc_score:.4f}")
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=150)
        print("ROC curve saved as 'roc_curve.png'")
        plt.close()
    else:
        print(f"\nNote: ROC curve is only plotted for binary classification.")
        print(f"Your dataset has {len(classes)} classes. Skipping ROC curve.")
    
    return accuracy, cm

def save_model(model, filename):
    """
    Save trained model to file using joblib.
    """
    try:
        joblib.dump(model, filename)
        print(f"\nModel saved successfully as '{filename}'")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def main():
    """Main function to run the training pipeline."""
    print("Motor Model Training")
    print("=" * 50)
    
    try:
        # Step 1: Load data
        df = load_data(CSV_FILENAME)
        
        # Check if we have enough data
        if len(df) < 10:
            print(f"\nWarning: Only {len(df)} samples found. More data is recommended for training.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return
        
        # Step 2: Compute vibration RMS
        df = compute_vibration_rms(df)
        
        # Step 3: Prepare features and target
        X, y = prepare_features_and_target(df)
        
        # Check if we have labels other than 0
        if len(y.unique()) == 1:
            print("\nWarning: All labels are the same. Model training may not be meaningful.")
            print("Please collect data with different labels (e.g., normal=0, abnormal=1)")
            return
        
        # Step 4: Split data into train and test sets
        # test_size: 20% of data for testing
        # random_state: for reproducibility
        # stratify: ensures both sets have similar class distribution
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y  # Maintain class distribution
        )
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Step 5: Train model
        model = train_model(X_train, y_train)
        
        # Step 6: Evaluate model
        accuracy, cm = evaluate_model(model, X_test, y_test)
        
        # Step 7: Save model
        save_model(model, MODEL_FILENAME)
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("=" * 50)
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPlease run data_logger.py first to collect data.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
