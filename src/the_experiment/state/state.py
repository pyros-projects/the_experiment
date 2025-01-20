
from the_experiment.comparison.load_model import check_out_folder
from the_experiment.comparison.model_eval import ModelEvaluator

training_folders = check_out_folder()
MODEL_EVALUATOR:ModelEvaluator = ModelEvaluator(training_folders)