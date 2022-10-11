import os
import yaml
import mlflow


def download_artifact_yaml(artifact_uri):

    filename = mlflow.artifacts.download_artifacts(artifact_uri)

    with open(filename) as f:
        artifact = yaml.safe_load(f)

    return artifact


def get_artifact_storage(tracking_uri, experiment_name, run_id):

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    artifact_storage = os.path.join(experiment.artifact_location, run_id, "artifacts")

    return artifact_storage
