#Most commong libraries used in tfx
#Make sure you have the latest requirements.txt file installed firs before proceeding
#Common used Libraries
import os
import urllib.request
import tempfile
import shutil
import pprint
from absl import logging
logging.set_verbosity(logging.INFO)  # Set default logging level.
pp = pprint.PrettyPrinter()
#Tensorflow related libraries
import tensorflow as tf
from tfx import v1 as tfx
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
#TFX standard components
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import InfraValidator
from tfx.components import Pusher
#from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components import Tuner

#Orchestration Libraries
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

import taxi_transform
import taxi_constants

#VESIONING CHECK
print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))
print('Tensorflow data Validation version: {}'.format(tfdv.__version__))
print('Tensorflow model analysis version: {}'.format(tfma.__version__))

#Data Confiuration part
# This is the root directory for your TFX pip package installation.
_tfx_root = tfx.__path__[0]

# This is the directory containing the TFX Chicago Taxi Pipeline example.
_taxi_root = os.path.join(_tfx_root, 'examples/chicago_taxi_pipeline')

_data_root = tempfile.mkdtemp(prefix='tfx-data')
DATA_PATH = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/chicago_taxi_pipeline/data/simple/data.csv'
_data_filepath = os.path.join(_data_root, "data.csv")
urllib.request.urlretrieve(DATA_PATH, _data_filepath)

#Pipeline Configuration
PIPELINE_NAME = 'chicago_taxi_pipeline'
# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('serving_model/taxi_simple', PIPELINE_NAME)

#Schema Part

#Module Part 
_constants_file = 'taxi_constants.py'
_transform_file = 'taxi_transform.py'
_train_file = 'taxi_trainer.py'

# context = InteractiveContext()

#TFX Pipeline
def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                    serving_model_dir: str, metadata_path: str) -> tfx.dsl.Pipeline:
    """Implements the penguin pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

  # Import the schema.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"], 
        infer_feature_shape=False)

  # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

  #Transforms input data using preprocessing_fn in the 'module_file'.
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(_transform_file))

  # Uses user-provided Python function that trains a model.
    trainer = tfx.components.Trainer(
        module_file=os.path.abspath(_train_file),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=10000),
        eval_args=tfx.proto.EvalArgs(num_steps=5000))

    # Pushes the model to a filesystem destination.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

        
    eval_config = tfma.EvalConfig(
        model_specs=[
            # This assumes a serving model with signature 'serving_default'. If
            # using estimator based EvalSavedModel, add signature_name: 'eval' and 
            # remove the label_key.
            tfma.ModelSpec(label_key='tips')
        ],
            metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ]
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ])

    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
                'latest_blessed_model_resolver')

    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        pusher,
        model_resolver,
        evaluator,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
        .sqlite_metadata_connection_config(metadata_path),
        components=components)

tfx.orchestration.LocalDagRunner().run(_create_pipeline(pipeline_name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT, 
data_root=_data_root, serving_model_dir=SERVING_MODEL_DIR, metadata_path=METADATA_PATH))
