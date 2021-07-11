### Hit Song Predictor Spotify
##### MIT License 

## Contributor: Rutu Nanavati
#### Requirements:
Python 3
joblib==1.0.1
numpy==1.20.3
pandas==1.2.4
python-dateutil==2.8.1
pytz==2021.1
scikit-learn==0.24.2
scipy==1.6.3
six==1.16.0
sklearn==0.0
threadpoolctl==2.1.0


### CSV files
	* `wm_project.csv` : input data provided
	* `test_data.csv`  : this includes rows that had `hits` NA in `wm_project.csv`
	* `predicted_y.csv`: Predicted results from the features in `test_data.csv`


### Notebook for EDA
####  `hit-song-prediction.ipynb`
	* Input files: wm_project.csv
	* hit-song-prediction.ipynb contains EDA and model training.
	* Also contains some insights from the data

### Python Modules
#### `main.py`
	 * Input files : `test_data.csv` or batch of values for which results need to be Predicted.
	 * Output file: `predicted_y.csv` or results on input batch values

#### `predict_hit.py` :  will use track id to predict if the song is hit or not for a particular song.

#### `train.py` :  trains the model
   * Input files : `wm_project.csv`
	 * Output : `models/` models are stored in the model folder
#### `project_functions.py`

### Folders
#### `models/`
	 *  models are stored to be loaded and run to perform prediction

#### Steps to run the model:
  * install requirements
	* Run `python main.py test_data.csv`
		- we can replace `test_data.csv` with any other batch file to obtain prediction
	* The same environment can be used to run the `hit-song-prediction.ipynb`
#### Next steps:
  * Research from initial insights to work on model improvement.
	* Research and Test causation build Probabilistic model that would account for causation.

##### Feel Free to ask any follow-up questions at nanavati.rutu@gmail.com
