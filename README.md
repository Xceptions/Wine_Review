# Wine_Review
An XGBoost Model on Wine Review

This model was built using luigi to handle the tasks management

To run:
    PYTHONPATH=. luigi --module index TrainModel --input-data ../data_root/raw/wine_dataset.csv --local-scheduler
	
To make the predictions:
    python predictions.py

The model evaluation is saved as "model_evaluation.html" and was built using pweave
