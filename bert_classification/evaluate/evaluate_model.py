from azureml.core import Run
import argparse
import traceback
from util.model_helper import get_model

run = Run.get_context()

exp = run.experiment
ws = run.experiment.workspace
run_id = 'amlcompute'

parser = argparse.ArgumentParser("evaluate")

parser.add_argument(
    "--run_id",
    type=str,
    help="Training run ID",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the Model",
    default="bert_classification_model.bin",
)

parser.add_argument(
    "--allow_run_cancel",
    type=str,
    help="Set this to false to avoid evaluation step from cancelling run after an unsuccessful evaluation",  # NOQA: E501
    default="true",
)

args = parser.parse_args()
if (args.run_id is not None):
    run_id = args.run_id
if (run_id == 'amlcompute'):
    run_id = run.parent.id
model_name = args.model_name
metric_eval = "loss"

allow_run_cancel = args.allow_run_cancel
# Parameterize the matrices on which the models should be compared
# Add golden data set on which all the model performance can be evaluated
try:
    firstRegistration = False
    tag_name = 'experiment_name'

    model = get_model(
        model_name=model_name,
        tag_name=tag_name,
        tag_value=exp.name,
        aml_workspace=ws)

    if (model is not None):
        production_model_loss = 100
        if (metric_eval in model.tags):
            production_model_loss = float(model.tags[metric_eval])
        new_model_loss = float(run.parent.get_metrics().get(metric_eval))
        if (production_model_loss is None or new_model_loss is None):
            print("Unable to find", metric_eval, "metrics, "
                  "exiting evaluation")
            if((allow_run_cancel).lower() == 'true'):
                run.parent.cancel()
        else:
            print(
                "Current Production model loss: {}, "
                "New trained model loss: {}".format(
                    production_model_loss, new_model_loss
                )
            )

        if (new_model_loss < production_model_loss):
            print("New trained model performs better, "
                  "thus it should be registered")
        else:
            print("New trained model metric is worse than or equal to "
                  "production model so skipping model registration.")
            if((allow_run_cancel).lower() == 'true'):
                run.parent.cancel()
    else:
        print("This is the first model, "
              "thus it should be registered")

except Exception:
    traceback.print_exc(limit=None, file=None, chain=True)
    print("Something went wrong trying to evaluate. Exiting.")
    raise
