"""
Resume Parser for ATS System.
Extracts structured information from PDF, DOCX, and DOC resumes with AI-powered skill analysis.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

# PDF/DOCX parsing libraries
try:
    import PyPDF2
    from docx import Document
except ImportError:
    pass

# DOC parsing libraries (older binary format)
try:
    import textract
except ImportError:
    textract = None
try:
    import pypandoc
except ImportError:
    pypandoc = None
try:
    from nt_textfileloader import TextFileLoader
    nt_loader = TextFileLoader()
except ImportError:
    nt_loader = None

# NLP libraries
try:
    import spacy
    from spacy.matcher import Matcher
except ImportError:
    pass

# OpenAI/Azure OpenAI for AI-powered extraction
try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    pass

logger = logging.getLogger(__name__)

from profile_type_utils import determine_primary_profile_type

# Import the new EducationExtractor
try:
    from education_extractor import EducationExtractor, extract_education as extract_education_standalone
except ImportError:
    EducationExtractor = None
    extract_education_standalone = None
    logger.warning("EducationExtractor not available, using fallback extraction")

# Import the new ExperienceExtractor
try:
    from experience_extractor import ExperienceExtractor, extract_experience as extract_experience_standalone
except ImportError:
    ExperienceExtractor = None
    extract_experience_standalone = None
    logger.warning("ExperienceExtractor not available, using fallback extraction")


class ResumeParser:
    """Intelligent resume parser with NLP and AI capabilities."""

        # Comprehensive technical skills database - ONLY these should appear in primary_skills
    TECHNICAL_SKILLS = {
        # === Programming Languages ===
        'python', 'py', 'java','core java','advanced java' 'javascript', 'js', 'ecmascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
        'swift', 'kotlin', 'scala', 'r', 'perl', 'bash', 'shell scripting', 'objective-c', 'dart',
        'lua', 'matlab', 'assembly', 'fortran', 'sas', 'haskell', 'clojure', 'visual basic', 'vb.net', 'abap',
        
        # === Frameworks / Libraries ===
        'django', 'django rest framework', 'django-cors-headers', 'django-allauth', 'django-crispy-forms', 'django-channels', 'django-filter', 'django-storages', 'django-redis', 'django-debug-toolbar', 'django-ckeditor', 'django-rest-auth', 'django-simplejwt', 'django-haystack', 'django-elasticsearch-dsl', 'django-oauth-toolkit', 'django-extensions',
        'flask', 'spring', 'react', 'react.js', 'reactjs', 'react framework', 'angular', 'vue', 'node.js', 'fastapi', 'express', 'express.js',
        'next.js', 'nestjs', 'laravel', 'symfony', 'flutter', 'react native', 'svelte', 'pytorch', 'tensorflow',
        'struts', 'play framework', 'koa', 'meteor', 'ember.js', 'backbone.js', 'codeigniter', 'cakephp', 'yii',
        'nuxt.js', 'gatsby', 'blazor', 'qt', 'tornado', 'pyramid', 'bottle', 'falcon', 'aiohttp', 'hug', 'web2py',
        'streamlit', 'gradio', 'dash', 'panel', 'plotly-dash', 'quart', 'starlette', 'connexion', 'masonite', 'sanic',
        'remix', 'solid.js', 'preact', 'alpine.js', 'marko', 'lit', 'stencil', 'dojo', 'blitz.js',
        'recoil', 'mobx', 'react query', 'tanstack query', 'react hook form', 'formik', 'yup', 'zustand', 'immer', 'rxjs', 'jotai', 'valtio', 'xstate',
        'styled-components', 'emotion', 'stitches', 'vanilla extract', 'react router', 'react-router', 'react helmet', 'react intl', 'i18next', 'luxon',
        'axios', 'lodash', 'moment.js', 'day.js', 'date-fns', 'immutable.js', 'chart.js', 'd3.js', 'highcharts', 'echarts', 'handsontable',
        'three.js', 'pixi.js', 'greenSock (gsap)', 'gsap', 'anime.js', 'react three fiber', 'react spring', 'react table', 'react testing library',
        'graphql', 'apollo client', 'urql', 'swr', 'material ui', 'chakra ui', 'ant design', 'mantine', 'recharts',
        'classnames', 'uuid', 'ramda', 'prop-types', 'react-icons', 'react-toastify',
        'hibernate', 'prisma', 'sequelize', 'typeorm', 'knex.js', 'peewee', 'sqlalchemy', 'mongoose', 'pymongo', 'motor', 'mongoengine', 'bson', 'mangum', 'beanie',
        
        # === .NET Framework ===
        '.net', '.net core', '.net framework', 'asp.net', 'asp.net mvc', 'asp.net core', 'ado.net', 'entity framework', 'linq',
        
        # === Databases / Data Tools ===
        'sql', 'sql server', 'mysql', 'postgresql', 'postgres', 'psql', 'postgres db', 'pg', 'mongodb', 'mongo', 'mongodb atlas', 'mongo db', 'redis', 'nosql', 'oracle', 'sqlite', 'elasticsearch', 'snowflake',
        'firebase', 'dynamodb', 'cassandra', 'neo4j', 'bigquery', 'redshift', 'clickhouse', 'couchdb', 'hbase',
        'influxdb', 'memcached', 'realm', 'timescaledb', 'duckdb', 'cosmos db', 'supabase', 'psycopg2', 'psycopg', 'pg-promise', 'asyncpg',
        'pgvector', 'pgcli', 'pgx', 'pgbouncer', 'drizzle orm', 'alembic', 'tortoise orm', 'gino', 'odmantic', 'ormar', 'prisma client', 'objection.js', 'sqlmodel', 'pony orm', 'dataset',
        'pgadmin', 'dbeaver', 'navicat', 'tableplus', 'data grip', 'datagrip', 'heidisql', 'pg_dump', 'pg_restore', 'aws rds', 'azure database for postgresql', 'gcp cloud sql', 'neon.tech', 'timescale cloud',
        'docker postgres', 'kubernetes postgres operator', 'patroni', 'pgbackrest', 'wal-g', 'prometheus exporter', 'flyway', 'liquibase', 'pgbench', 'pg_stat_statements', 'pg_repack', 'pgbadger', 'pgloader', 'pg_upgrade',
        'mongodb compass', 'atlas', 'robo 3t', 'mongo shell', 'studio 3t', 'nosqlbooster', 'mongosh', 'mongostat', 'mongodump', 'mongorestore', 'mongotop', 'mongos', 'mongoperf', 'mongotools',
        'atlas cli', 'mongosh scripts', 'mlab', 'compose mongodb', 'azure cosmos db (mongo api)', 'aws documentdb', 'gcp firestore (mongo mode)', 'realm sync', 'mongo express', 'kubernetes mongo operator',
        'docker mongo', 'helm charts', 'mongobackup', 'mongobenchmark', 'grafana-mongodb plugin', 'prometheus exporter', 'datadog integration', 'elastic beats mongodb module', 'db-migrate',
        'mongoose', 'pymongo', 'motor', 'mongoengine', 'bson', 'mangum', 'gridfs', 'mtools', 'beanie', 'marshmallow', 'dnspython', 'mongo-hint', 'mongo-connector', 'mongoalchemy', 'mongoid', 'mongojs',
        'mongoose-auto-increment', 'mongoose-paginate', 'mongoose-validator', 'mongoose-schema', 'mongoose-aggregate-paginate',
        
        # === Cloud / DevOps ===
        'aws', 'amazon web services', 'aws cloud', 'azure', 'microsoft azure', 'azure cloud', 'gcp', 'docker', 'docker engine', 'containers', 'containerization', 'docker platform', 'docker compose', 'docker-compose', 'swarm', 'docker swarm', 'kubernetes', 'k8s', 'kube', 'kubernetes cluster', 'kubernetes engine', 'jenkins', 'terraform', 'hashicorp terraform', 'iac terraform', 'ansible', 'prometheus', 'grafana',
        'circleci', 'github actions', 'gh actions', 'github workflows', 'gitlab ci', 'bitbucket pipelines', 'travis ci', 'openstack', 'cloudformation', 'helm', 'istio',
        'argo cd', 'argo workflows', 'argo rollouts', 'vault', 'consul', 'packer', 'airflow', 'prefect', 'luigi', 'dagster', 'data pipeline', 'mlops', 'cloud run', 'lambda',
        'chef', 'puppet', 'saltstack', 'vagrant', 'terraform cloud', 'terraform enterprise', 'terragrunt', 'google deployment manager',
        'aws lambda', 'azure functions', 'azure app service', 'azure kubernetes service', 'aks', 'azure logic apps', 'azure data factory', 'azure synapse analytics', 'azure machine learning', 'azure cognitive services', 'azure devops', 'ado', 'azure active directory', 'azure sql database', 'azure cosmos db', 'azure storage', 'azure api management', 'azure service bus', 'event grid', 'event hubs', 'azure pipelines', 'bicep', 'arm templates', 'azure container instances', 'azure virtual machines', 'azure front door', 'azure application gateway', 'azure load balancer', 'azure monitor', 'azure sentinel', 'azure defender', 'azure databricks',
        'gcp cloud run', 'ecs', 'eks', 'cloudwatch', 'serverless framework', 'sam framework', 'chalice', 'pulumi', 'copilot cli',
        'azure-sdk-for-python', 'azure-sdk-for-js', 'azure-mgmt', 'azure-storage-blob', 'azure-identity', 'azure-keyvault', 'azure-cosmos', 'azure-eventhub', 'azure-functions-core-tools', 'msal', 'msrest', 'msgraph-core', 'adal', 'azureml-core', 'azureml-sdk',
        'azure portal', 'azure cli', 'azure powershell', 'azure devops pipelines', 'azure storage explorer', 'azure resource explorer', 'bicep cli', 'azure data studio', 'log analytics workspace', 'azure advisor', 'azure cost management', 'azure security center', 'network watcher', 'azure policy', 'azure blueprints', 'azure arc', 'azure bastion', 'kudu', 'azure app insights',
        'azure repos', 'azure boards', 'azure artifacts', 'azure test plans', 'azure-devops-python-api', 'pytest-azurepipelines', 'ado extensions marketplace',
        'teamcity', 'nexus', 'elk stack', 'gitpython', 'psutil', 'jsonschema', 'requests-oauthlib', 'dotenv', 'windows terminal', 'git bash', 'nexus repository', 'jenkins ui', 'service hooks',
        'pygithub', 'ghapi', 'octokit', 'actions-toolkit', 'shelljs', 'chalk', 'google-auth', 'gcloud-sdk',
        'github cli', 'github desktop', 'dependabot', 'codeql', 'veracode', 'snyk', 'slack', 'discord', 'notion integrations',
        'terraform aws provider', 'terraform azurerm provider', 'terraform google provider', 'terraform kubernetes provider', 'terraform helm provider', 'terraform cloudflare provider', 'terraform datadog provider', 'terraform github provider', 'terraform vault provider', 'terraform null provider', 'terraform local provider', 'terraform tls provider', 'python-hcl2', 'azure-mgmt-resource', 'google-api-python-client',
        'terraform cli', 'terraform fmt', 'terraform plan', 'terraform apply', 'terraform destroy', 'terraform workspace', 'terraform import', 'terraform output', 'terraform graph', 'tflint', 'tfsec', 'checkov', 'infracost', 'terragrunt cli', 'spacelift', 'env0', 'atlantis',
        'boto3', 'botocore', 'aws-sdk', 's3fs', 'aioboto3', 'awswrangler', 'aws cdk', 'cdk pipelines', 'amplify', 'aws cli', 'aws console', 'aws toolkit', 'aws sam cli',
        'cloudtrail', 'codepipeline', 'codebuild', 'codecommit', 'codeartifact', 'eventbridge', 'appsync', 'ec2', 's3', 'rds', 'dynamodb', 'elasticache', 'vpc', 'route53', 'efs',
        'cloudfront', 'alb', 'nlb', 'elb', 'aws config', 'aws shield', 'waf', 'guardduty', 'security hub', 'inspector', 'aws organizations', 'billing console', 'cost explorer', 'aws budgets', 'trusted advisor',
        'aws secrets manager', 'kms', 'parameter store', 'cloud9', 'codeguru', 'aws amplify cli', 'appsync console', 'quickSight ui', 'aws glue studio', 'athena console',
        'aws data exchange', 'aws data sync', 'snowball', 'outposts', 'localstack', 'eksctl', 'aws copilot', 'aws docker integration', 'aws fargate cli', 'aws lightsail ui', 'aws sso console',
        'docker-py', 'compose-cli', 'docker-sdk', 'docker hub', 'portainer', 'rancher', 'minikube', 'microk8s', 'kind', 'k3s', 'k3d',
        'kubectl', 'k9s', 'lens', 'podman', 'buildx', 'containerd', 'cri-o', 'nerdctl', 'colima', 'harbor', 'tekton', 'linkerd', 'linkerd2', 'knative', 'openshift', 'open shift',
        'gke', 'skaffold', 'flux', 'flux cd', 'kustomize', 'tilt', 'velero', 'kubeseal', 'kubens', 'kubectx', 'stern', 'kubetail',
        'dvc', 'mlflow', 'kubernetes-client', 'pykube', 'kube-api', 'client-go', 'fabric8', 'ansible-k8s', 'terraform-provider-kubernetes', 'helmfile', 'helm sdk', 'operator-sdk',
        'argo', 'argo cd', 'spinnaker', 'nomad', 'mesos', 'tanzu', 'garden', 'crossplane', 'kubeflow', 'kubebuilder', 'prometheus operator', 'grafana tempo', 'grafana loki', 'jaeger', 'open telemetry', 'cert manager',
        'calico', 'flannel', 'cilium', 'weave net', 'kube-router', 'traefik', 'nginx ingress controller', 'haproxy ingress', 'istio gateway', 'kong ingress', 'service mesh interface (smi)',
        'docker desktop', 'docker stats', 'docker inspect', 'docker logs', 'docker exec', 'docker cp', 'docker build', 'docker run', 'docker ps', 'docker prune', 'docker context', 'docker network', 'docker volume', 'docker system prune', 'docker tag', 'docker push',
        'compose up', 'compose down', 'compose logs', 'compose build', 'compose start', 'dive', 'ctop', 'cadvisor', 'datadog', 'new relic', 'elastic apm', 'splunk forwarder',
        'semaphore ci', 'harness', 'octopus deploy', 'vercel cli', 'netlify cli', 'aws glue', 'athena', 'redshift', 'data pipeline', 'quickSight', 'aws batch', 'fargate', 'elastic beanstalk', 'elasticache', 'emr', 'dms', 'snow family', 'sagemaker', 'bedrock', 'comprehend',
        'cloud computing', 'paas', 'iaas', 'saas', 'virtual networks', 'subnets', 'network security groups', 'private endpoints', 'load balancing', 'scaling', 'availability zones', 'resource groups', 'resource locks', 'identity and access management', 'rbac', 'service principals', 'managed identities', 'vnet peering', 'vpn gateway', 'expressroute', 'application insights', 'monitoring and alerting', 'disaster recovery', 'backup and restore', 'infrastructure as code', 'iac', 'immutable infrastructure', 'declarative configuration', 'cicd pipelines', 'devops automation', 'logging and metrics', 'data ingestion', 'data transformation', 'data pipelines', 'integration runtime', 'data lake architecture', 'data warehouse', 'big data analytics', 'machine learning models', 'model deployment', 'containerization', 'microservices architecture', 'serverless computing', 'function triggers', 'durable functions', 'api management', 'web apps', 'app gateways', 'ssl certificates', 'dns zones', 'custom domains', 'cost optimization', 'governance and compliance', 'zero trust security', 'threat protection',
        'terraform modules', 'providers', 'resources', 'data sources', 'variables', 'locals', 'outputs', 'state file', 'remote backend', 'terraform cloud backend', 's3 backend', 'azure blob backend', 'gcs backend', 'workspaces', 'dependency locking', 'module versioning', 'terraform registry', 'terraform plan and apply', 'terraform destroy', 'terraform refresh', 'terraform validate', 'terraform fmt', 'terraform import', 'terraform output', 'terraform taint', 'terraform graph', 'terraform console', 'state management', 'state locking', 'drift detection', 'environment segregation', 'remote execution', 'cloud provisioning', 'multi-cloud deployment', 'aws infrastructure', 'azure infrastructure', 'gcp infrastructure', 'kubernetes provisioning', 'helm release management', 'network configuration', 'vpc setup', 'subnets', 'security groups', 'iam roles', 'key management', 'load balancers', 'auto scaling groups', 'vm instances', 'dns records', 'storage accounts', 'object storage', 'database provisioning', 'monitoring setup', 'log configuration', 'pipeline integration', 'terraform testing', 'policy as code', 'opa integration', 'sentinel policies', 'cost estimation', 'infracost integration', 'gitops', 'ci/cd integration', 'version control', 'automation pipelines', 'terraform best practices', 'reusable modules', 'monorepo structure', 'root module design', 'dynamic blocks', 'count and for_each', 'lifecycle rules', 'sensitive variables', 'secrets management', 'vault integration', 'output sanitization', 'error handling', 'terraform upgrade process',
        
        # === Security / Authentication ===
        'oauth', 'oauth2', 'jwt', 'ssl', 'tls', 'saml', 'openid connect', 'mfa', 'iam', 'cybersecurity',
        'network security', 'firewall', 'penetration testing', 'encryption', 'hashing',
        
        # === AI / ML / Data Science ===
        'machine learning', 'ml', 'applied ml', 'data modeling', 'predictive modeling', 'ai', 'data science', 'analytics', 'nlp', 'computer vision', 'deep learning', 'dl', 'neural networks', 'deep neural networks', 'representation learning', 'pandas',
        'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'huggingface', 'huggingface transformers', 'openai api', 'llm', 'generative ai', 'langchain',
        'autogen', 'rasa', 'spacy', 'transformers', 'text classification', 'sentiment analysis', 'data visualization',
        'tableau', 'power bi', 'microsoft powerbi', 'bi tools', 'big data', 'hadoop', 'spark', 'pyspark', 'databricks', 'xgboost', 'lightgbm', 'keras',
        'power query', 'dax', 'm language', 'azure synapse', 'azure data lake', 'sql server analysis services', 'dataflows', 'fabric data warehouse', 'excel powerpivot',
        'powerbi-api', 'powerbi-rest-client', 'powerbi-embedded-sdk', 'pyodbc',
        'power bi desktop', 'power bi service', 'power bi mobile', 'power bi report builder', 'data gateway', 'fabric workspace', 'azure data factory', 'dax studio', 'tabular editor',
        'data modeling', 'relationships', 'measures', 'calculated columns', 'row level security', 'dashboards', 'reports', 'data refresh', 'directquery', 'import mode', 'workspaces', 'sharing & publishing', 'embedded analytics', 'data transformation', 'dax functions',
        'fastai', 'catboost', 'mlflow', 'scipy', 'joblib', 'pickle', 'nltk', 'gensim', 'polars', 'colab', 'tensorboard', 'sagemaker',
        'weights & biases', 'wandb', 'azure ml', 'google ai platform', 'vertex ai', 'onnx', 'mxnet', 'caffe', 'theano', 'jax',
        'torchvision', 'torchaudio', 'tqdm', 'opencv', 'opencv-python', 'datasets', 'albumentations', 'neptune.ai', 'clearml',
        'torch lightning', 'chainer', 'mindspore', 'deeplearning4j', 'dl4j', 'sonnet', 'paddlepaddle', 'detectron2', 'yolov8', 'mediapipe', 'mmcv', 'openvino', 'openml', 'mljar-supervised', 'pycaret', 'autokeras', 'h2o.ai',
        'optuna', 'ray tune', 'tfx', 'sagemaker pipelines', 'vertex ai pipelines', 'kubeflow', 'dvc', 'zenml', 'mlrun', 'polyaxon', 'valohai', 'flyte',
        'textblob', 'word2vec', 'sentence-transformers', 'faiss', 'llama-index', 'bertviz', 'shap', 'lime', 'yellowbrick', 'imbalanced-learn', 'category-encoders', 'featuretools', 'dask', 'modin', 'vaex',
        'statsmodels', 'sympy', 'pymc', 'arviz', 'prophet', 'fbprophet', 'gluonts', 'tsfresh', 'river', 'scikit-time',
        'imageio', 'pydub', 'speechbrain', 'timm', 'diffusers', 'accelerate', 'bitsandbytes', 'deepspeed', 'peft', 'sentencepiece', 'huggingface-hub', 'torchmetrics', 'pytorch-ignite', 'ignite', 'keras-tuner', 'tensorlayer', 'tensorflow-addons', 'tensorflow-datasets',
        'jupyter', 'deepnote', 'polynote', 'nvidia-smi', 'pytorch profiler', 'tensorboard profiler', 'mlrun', 'aimstack', 'supervisely', 'roboflow', 'label studio', 'voxel51 fiftyone', 'metaflow',
        'supervised learning', 'unsupervised learning', 'semi-supervised learning', 'reinforcement learning', 'transfer learning', 'active learning', 'federated learning', 'online learning', 'batch learning', 'ensemble learning',
        'linear regression', 'logistic regression', 'decision trees', 'random forest', 'svm', 'naive bayes', 'k-means clustering', 'hierarchical clustering', 'dbscan', 'pca', 'lda', 'qda', 'knn', 'gbm',
        'feature engineering', 'feature selection', 'data preprocessing', 'normalization', 'standardization', 'outlier detection', 'missing value imputation', 'dimensionality reduction', 'one-hot encoding', 'scaling', 'label encoding', 'feature importance',
        'cross validation', 'train-test split', 'confusion matrix', 'roc-auc', 'precision recall', 'f1 score', 'mae', 'rmse', 'accuracy', 'r2 score', 'bias-variance tradeoff', 'model explainability', 'shap values', 'lime interpretation',
        'grid search', 'random search', 'bayesian optimization', 'optuna tuning', 'hyperopt', 'ray tune', 'early stopping',
        'cnn', 'rnn', 'lstm', 'gru', 'transformers', 'attention mechanism', 'autoencoders', 'gan', 'vae', 'reinforcement learning agents',
        'tokenization', 'stemming', 'lemmatization', 'word embeddings', 'pos tagging', 'sentiment analysis', 'text classification', 'language modeling', 'seq2seq', 'encoder-decoder', 'bert', 'gpt', 't5', 'llm fine-tuning',
        'image classification', 'object detection', 'segmentation', 'face recognition', 'ocr', 'data augmentation', 'cnn architectures', 'resnet', 'mobilenet', 'efficientnet', 'vision transformers',
        'time series forecasting', 'arima', 'sarima', 'lstm forecasting', 'prophet forecasting', 'seasonal decomposition', 'anomaly detection', 'trend analysis', 'rolling averages', 'autocorrelation', 'stationarity',
        'model deployment', 'api serving', 'model versioning', 'model registry', 'monitoring', 'model drift', 'feature store', 'pipeline orchestration', 'continuous training', 'cicd for ml', 'ml model packaging',
        'data lakes', 'data warehouse', 'data pipelines', 'etl', 'elt', 'big data', 'distributed training', 'cloud ml services', 'gpu acceleration', 'tensorRT optimization', 'batch inference', 'real-time inference',
        'meta learning', 'self-supervised learning', 'zero-shot learning', 'few-shot learning', 'contrastive learning', 'multi-modal learning', 'knowledge distillation', 'causal inference', 'explainable ai', 'ethical ai', 'ai fairness',
        'tensorboard visualization', 'wandb dashboards', 'confusion matrix plotting', 'feature importance plotting', 'learning curves', 'residual plots', 'data drift visualization', 'distribution plots', 'embedding projection',
        'recommendation systems', 'churn prediction', 'fraud detection', 'credit scoring', 'forecasting', 'image captioning', 'speech recognition', 'chatbots', 'document classification', 'automation pipelines',
        'feedforward neural network', 'backpropagation', 'gradient descent', 'stochastic gradient descent', 'activation functions', 'relu', 'sigmoid', 'tanh', 'softmax', 'dropout', 'batch normalization', 'weight initialization',
        'loss functions', 'cross entropy', 'mse loss', 'mae loss', 'hinge loss', 'optimizer', 'adam', 'sgd', 'rmsprop', 'adagrad',
        'alexnet', 'vgg', 'inception', 'unet', 'yolo', 'faster rcnn', 'vit', 'llama', 'clip', 'detr', 'segment anything model', 'stable diffusion', 'dreambooth', 'controlnet',
        'learning rate scheduling', 'regularization', 'gradient clipping', 'mixed precision training', 'distributed training', 'multi gpu training', 'tensor parallelism', 'pipeline parallelism',
        'fine-tuning', 'transfer learning', 'zero-shot learning', 'few-shot learning', 'self-supervised learning', 'contrastive learning', 'meta learning', 'active learning', 'semi-supervised learning', 'reinforcement learning',
        'instance segmentation', 'semantic segmentation', 'pose estimation', 'optical flow', 'image generation', 'super resolution', 'style transfer', 'gesture recognition', '3d vision', 'depth estimation', 'video processing', 'image preprocessing', 'augmentation pipeline', 'image embeddings',
        'text generation', 'translation', 'summarization', 'question answering', 'named entity recognition', 'speech synthesis', 'audio classification', 'emotion detection', 'text embeddings', 'prompt tuning', 'instruction tuning', 'llm fine-tuning',
        'transformer architecture', 'attention', 'self attention', 'multi-head attention', 'positional encoding', 'cross attention', 'diffusion models', 'score-based models', 'energy-based models', 'graph neural networks', 'gnn', 'graph convolution networks',
        'capsule networks', 'neural architecture search', 'neural rendering', 'neural radiance fields', 'nerf', 'implicit representations', 'adversarial training', 'adversarial attacks', 'model robustness', 'explainable ai', 'interpretability', 'grad-cam',
        'onnx export', 'tensorrt optimization', 'quantization', 'pruning', 'model compression', 'knowledge distillation', 'inference optimization', 'model serving', 'api deployment', 'torchscript', 'tf serving', 'mlflow model registry',
        'containerization', 'gpu acceleration', 'cuda', 'cudnn', 'nvcc', 'opencl', 'distributed inference', 'batch inference', 'real-time inference', 'edge ai', 'tinyml', 'mobile deployment', 'tensorflow lite', 'coreml', 'onnx runtime',
        'training curves', 'gradient flow', 'confusion matrix', 'feature maps', 'embedding visualization', 'activation visualization', 'model profiling', 'loss curve analysis',
        'autonomous vehicles', 'medical imaging', 'recommendation systems', 'fraud detection', 'speech recognition', 'document processing', 'video analytics', 'industrial automation', 'ai assistants', 'image restoration',
        
        # === APIs, Architecture, Monitoring ===
        'rest api', 'graphql', 'graphql api', 'restful api', 'restful services', 'soap', 'rpc', 'grpc', 'openapi',
        'swagger', 'swagger ui', 'api testing', 'load testing', 'jmeter', 'new relic', 'datadog', 'sentry',
        'application monitoring', 'performance tuning', 'microservices', 'websockets', 'api gateway',
        'message queues', 'rabbitmq', 'kafka', 'celery', 'redis streams', 'event-driven architecture',
        'service mesh', 'load balancer',
        
        # === CI/CD & Testing ===
        'git', 'github', 'gitlab', 'agile', 'scrum', 'devops', 'pytest', 'jest',
        'mocha', 'cypress', 'postman', 'newman', 'swagger', 'jira', 'confluence', 'maven', 'gradle', 'ant', 'sonarqube',
        'selenium', 'selenium-webdriver', 'playwright', 'puppeteer', 'testng', 'junit', 'mockito', 'karma', 'chai', 'enzyme', 'vitest', 'pytest-django',
        'sinon.js', 'ava', 'tape', 'supertest', 'nightwatch', 'testing library', 'qUnit', 'protractor', 'webdriverio',
        'pytest-docker', 'pytest-ansible', 'pytest-kubernetes', 'pytest-helm', 'pytest-operator', 'pytest-yaml', 'pytest-parallel', 'pytest-mock',
        'ci/cd pipelines', 'continuous integration', 'continuous deployment', 'build pipelines', 'release pipelines', 'yaml templates', 'yaml pipelines', 'stages', 'jobs', 'steps', 'variables', 'environments', 'agents', 'self-hosted agents', 'deployment groups', 'service connections', 'artifact feeds', 'code versioning', 'branch policies', 'merge requests', 'pull requests', 'work items', 'agile boards', 'kanban', 'scrum sprints', 'test management', 'build automation', 'deployment automation', 'approvals and gates', 'integration testing', 'docker build and push', 'kubernetes deploy', 'multi-stage pipelines', 'variable groups', 'secrets management', 'key vault integration', 'notifications and alerts', 'release rollback', 'blue-green deployment', 'canary deployment', 'code coverage', 'quality gates', 'unit testing', 'security scanning', 'artifact retention', 'pipeline caching', 'dependency management', 'governance policies', 'cost optimization', 'workflow automation', 'repository branching', 'git flow', 'version tagging', 'pipeline triggers', 'manual approvals', 'task groups', 'templates reuse', 'yaml reuse', 'cross-platform builds', 'docker-compose integration', 'test results publishing', 'parallel execution', 'scheduled builds', 'infrastructure provisioning', 'monitoring and logging', 'incident management', 'sla tracking', 'service hooks', 'webhooks', 'azure monitor integration', 'azure security compliance', 'enterprise policy enforcement',
        'workflows', 'runners', 'self-hosted runners', 'matrix builds', 'on push triggers', 'on pull_request triggers', 'manual dispatch', 'scheduled workflows', 'cron syntax', 'repository dispatch', 'composite actions', 'reusable workflows', 'workflow templates', 'caching dependencies', 'build artifacts', 'test automation', 'environment protection rules', 'branch protection', 'required reviews', 'pull request checks', 'multi-environment deployment', 'canary releases', 'terraform deployment', 'container registry', 'helm release', 'npm publish', 'pypi publish', 'package versioning', 'semantic versioning', 'tagging', 'release creation', 'github environments', 'job dependencies', 'parallel jobs', 'artifact retention', 'workflow logs', 'monitoring and alerts', 'status checks', 'test result publishing', 'linting', 'static code analysis', 'snyk integration', 'dependabot alerts', 'codeql scanning', 'secret scanning', 'workflow permissions', 'fine-grained tokens', 'oidc authentication', 'aws oidc federation', 'azure oidc integration', 'gcp service accounts', 'cross-cloud deployment', 'slack notifications', 'teams notifications', 'email alerts', 'ci optimization', 'caching strategies', 'container workflows', 'monorepo support', 'matrix strategy', 'build speed optimization', 'test parallelization', 'custom action creation', 'dockerfile actions', 'javascript actions', 'composite actions', 'version pinning', 'marketplace actions', 'open source contribution workflows', 'github pages deploy', 'static site deploy', 'serverless deployment', 'cloud function triggers', 'pull request automation', 'issue automation', 'auto merge', 'auto label', 'release draft', 'changelog generation',
        
        # === Frontend / UI / UX ===
        'html', 'html5', 'css', 'css3', 'bootstrap', 'tailwind css', 'jquery', 'tailwind', 'chakra ui', 'material ui', 'ant design', 'semantic ui', 'foundation', 'bulma', 'daisy ui', 'uikit', 'redux', 'zustand',
        'framer motion', 'figma', 'ux design', 'responsive design', 'pwa', 'webpack', 'vite', 'babel', 'webpack-cli',
        'babel-cli', 'grunt', 'gulp', 'parcel', 'rollup', 'snowpack', 'storybook', 'chromatic', 'bit.dev',
        'sass', 'less', 'postcss', 'styled components', 'emotion', 'gsap', 'anime.js', 'three.js', 'pixi.js',
        'webpack-dev-server', 'browserify', 'swc', 'postcss', 'tailwind cli', 'husky', 'lint-staged', 'commitlint', 'git hooks',
        
        # === Mobile / Cross-Platform ===
        'android', 'ios', 'xcode', 'swiftui', 'jetpack compose', 'ionic', 'capacitor', 'cordova',
        'unity', 'unreal engine', 'electron', 'nw.js', 'expo', 'deno', 'bun',
        
        # === ERP / CRM / Low-Code ===
        'sap', 'sap abap', 'sap hana', 'salesforce', 'salesforce crm', 'salesforce apex', 'salesforce lightning',
        'lwc', 'visualforce', 'force.com', 'heroku', 'tableau crm', 'muleSoft', 'sales cloud', 'service cloud',
        'apex classes', 'soql', 'sosl', 'aura components', 'api sdk', 'salesforce dx', 'trigger handlers',
        'metadata api', 'salesforce cli', 'workbench', 'data loader', 'developer console', 'vs code extension',
        'sandbox', 'trailhead',
        'power apps', 'microsoft powerapps', 'power platform', 'power automate', 'microsoft flow', 'powerflow', 'power bi', 'dataverse', 'power fx', 'model-driven apps',
        'canvas apps', 'power pages', 'power virtual agents', 'sharepoint integration', 'teams apps', 'ai builder', 'connectors', 'power platform connectors',
        'desktop flows', 'business process flows', 'approval workflows', 'teams integration', 'logic apps',
        'dataverse api', 'office365 api', 'microsoft graph api', 'graph api', 'excel connectors', 'sql connectors', 'flow api', 'azure functions', 'custom connector sdk',
        'office365 connectors', 'sharepoint connectors', 'http requests', 'json',
        'power apps studio', 'make.powerapps.com', 'power automate portal', 'flow designer', 'power platform admin center', 'power platform cli', 'solution explorer', 'environment variables', 'microsoft teams',
        'desktop flow recorder', 'monitoring dashboard', 'make.powerautomate.com', 'sharepoint', 'azure portal', 'onedrive', 'power automate desktop', 'microsoft dynamics 365', 'business central', 'navision',
        'triggers', 'actions', 'conditions', 'loops', 'approvals', 'data connections', 'scheduled flows', 'instant flows', 'automated flows', 'desktop automation', 'rpa', 'integration', 'custom connectors', 'security roles', 'governance & compliance',
        
        # === Python Tools & Libraries ===
        'beautifulsoup', 'pydantic', 'dataclasses', 'attrs', 'jupyter', 'notebook', 'jupyterlab', 'virtualenv', 'pip', 'conda', 'black', 'flake8', 'mypy', 'poetry', 'tox',
        'isort', 'pre-commit', 'pytest-cov', 'pytest-xdist', 'gunicorn', 'uvicorn', 'hypercorn', 'celery', 'kombu', 'channels', 'crispy forms', 'jinja2', 'whitenoise', 'drf-yasg', 'manage.py', 'pipenv',
        'pgadmin', 'nginx', 'supervisor', 'requests', 'httpx', 'fabric', 'redis-py', 'pika', 'paramiko', 'click', 'typer', 'rich', 'loguru',
        'unittest', 'doctest', 'factory-boy', 'faker', 'coverage.py', 'sqlite browser', 'heroku cli', 'aws elastic beanstalk cli', 'celery beat', 'redis', 'ngrok',
        'turbogears', 'falconry', 'morepath', 'responder', 'nameko', 'cherrypy', 'drf-nested-routers', 'pillow', 'mysqlclient',
        
        # === JavaScript/Node Tools ===
        'npm', 'yarn', 'pnpm', 'npx', 'eslint', 'prettier', 'vite', 'webpack-cli', 'babel-cli', 'grunt', 'gulp', 'rollup', 'parcel', 'snowpack', 'ts-node', 'nodemon', 'browserify', 'esbuild', 'vercel cli', 'netlify cli',
        'node.js runtime', 'v8 engine', 'chrome devtools', 'firefox devtools', 'cloudflare workers', 'aws lambda (nodejs)', 'azure functions (nodejs)', 'google cloud functions', 'deno runtime',
        'github actions', 'gitlab ci', 'circleci', 'travis ci', 'jenkins', 'docker', 'vercel', 'netlify', 'heroku', 'aws amplify', 'digital ocean apps', 'railway', 'render', 'surge', 'cloudflare pages', 'firebase hosting', 's3 static hosting', 'nginx',
        'jest', 'mocha', 'chai', 'cypress', 'playwright', 'puppeteer', 'selenium-webdriver', 'storybook', 'vitepress', 'astro',
        
        
        # === Other / Emerging Tech ===
        'blockchain', 'solidity', 'smart contracts', 'web3', 'nft', 'metaverse', 'edge computing',
        'quantum computing', 'robotics', 'iot', 'raspberry pi', 'arduino', 'automation',
        
        # === IDE / Development Tools ===
        'visual studio', 'visual studio code', 'vscode', 'eclipse', 'intellij idea', 'netbeans', 'xcode', 'android studio',
        'pycharm', 'anaconda', 'miniconda', 'jupyterhub', 'google colab', 'kaggle', 'streamlit cloud', 'huggingface hub'
    }
    
    DOMAINS = {
  "Information Technology","Software Development","Cloud Computing","Cybersecurity","Data Science","Blockchain",
  "Internet of Things","Banking","Finance","Insurance","FinTech","Healthcare","Pharmaceuticals","Biotechnology",
  "Manufacturing","Automotive","Energy","Construction","Retail","E-commerce","Logistics","Telecommunications",
  "Media & Entertainment","Advertising & Marketing","Education Technology","Public Sector","Real Estate",
  "Hospitality","Travel & Tourism","Agriculture","Legal & Compliance","Human Resources","Environmental & Sustainability"
        # Note: 'education' removed - only 'education technology' or 'edtech' should match (via AI prompt)
        # Generic education/degrees are qualifications, not business domains
    }
    
    # AI-powered comprehensive extraction prompt
    AI_COMPREHENSIVE_EXTRACTION_PROMPT = """
🧠 ROLE / PERSONA

You are an Expert Resume Parser and Metadata Extraction Specialist trained to identify and extract complete, accurate, and ATS-ready professional data from unstructured resumes.

Your behavior and purpose:

Act as a senior technical recruiter assistant who understands resume semantics, structure, and ATS taxonomy.

Analyze resumes systematically to identify factual candidate details.

Ensure the extracted data is accurate, complete, normalized, and JSON-valid.

Never generate commentary, guesses, or summaries beyond required fields.

Your only task is to return validated structured JSON output.

🎯 GOAL

Your goal is to analyze the provided resume text and return a structured JSON containing validated professional metadata, including:

Personal Information

Career Details

Skills

Domain

Education

Certifications

Summary

The extracted JSON must be database-ready and syntactically valid (no markdown or extra text).

⚙️ EXTRACTION GUIDELINES
1. full_name

Identify the candidate’s actual personal name.
Follow these detailed rules and rejection patterns:

Rules for Extraction:

Name is usually on line 1 or 2 (Title Case or ALL CAPS).

Should contain 2–4 alphabetic words only.

Reject anything containing punctuation, digits, or organization names.

Stop searching after the first 3 lines.

Reject:

Section headers (Education, Experience, etc.)

Degrees, Job Titles, or Organization Names.

Examples:
✅ Correct: Daniel Mindlin, VARRE DHANA LAKSHMI DURGA
❌ Incorrect: Education, Infosys, B.Tech in EEE

2. email

Extract the primary and valid email address.

Must be RFC-compliant.

Never omit if present.

3. phone_number

Extract a valid phone number, including country code if available.

4. total_experience

total_experience — Extraction Rules

Primary Calculation Method

Always compute total professional experience using the earliest employment start date and the latest employment end date (or “Present”).

Experience = (latest date – earliest date).

Ignore employment gaps.

Ignore job overlaps (do not double count).

Explicit Mentions Are Secondary

If explicit text (e.g., “5 years experience”) conflicts with computed experience, always use the computed value.

Role Coverage

Include all roles listed under Experience / Work History, regardless of job titles.

Exclude: education, certifications, projects without dates.

Handling Missing Dates

If end date is “Present”, “Current”, or missing → assume today’s date.

If start date is missing but other roles have valid dates → use earliest available date.

Overlapping Periods

Count overlapping employment only once.

Overlap should not inflate total experience.

Internships & Part-time Roles

Count them ONLY if they appear inside the main experience section with valid dates.

Do NOT count student/academic roles.

Date Format Flexibility

Recognize formats like:
“Aug 2015”, “August 2015”, “08/2015”, “2015-08”, “2015”, etc.

Output Format

Return total experience as a float rounded to one decimal (e.g., 10.1).

If the number is whole, return an integer (e.g., 10).

5. current_company

Extract the current or most recent employer name.

6. current_designation

Extract the most recent job title.

7. technical_skills

Capture all technical skills (programming languages, frameworks, databases, cloud, DevOps, tools).
Use your internal technical skill dictionary for normalization and matching, also use the sample skills from the sample skills {{TECHNICAL_SKILLS}} if it is present.

8. secondary_skills

Extract non-technical or complementary skills (communication, leadership, project management, mentoring).

9. all_skills

Combine technical and secondary skills into one unified list.

10. domain

Extract professional domains or industries (not education fields).
Use the sample list as reference :
 {{DOMAINS}}

Rules:

If technical keywords (Python, Java, AWS, etc.) appear → include "Information Technology".

Include multiple relevant business domains (e.g., "Banking", "Finance").

Ignore educational degrees — only professional domains count.

11. education

Education Section Detection
Consider any of the following headings as education:
“Education”, “Academic Details”, “Academics”, “Qualification / Qualifications”,
“Educational Background”, “Academic Summary”, “Scholastic Profile”,
“Educational Credentials”, “Education Summary”, “Education & Training”
(extract only actual degrees; ignore trainings/certifications).

Identify All Degrees and Academic Qualifications
Detect structured or unstructured formats, including tables with columns like:
“Qualification”, “Course”, “Board/University”, “Year”, “Percentage/CGPA”.

Degree Ranking (Choose Only the Highest Level)
When multiple items appear, return only the highest by this order:

PhD / Doctorate

Master's Degrees (M.S, M.Sc, M.Tech, MBA, MCA, MA, etc.)

Bachelor's Degrees (B.Tech, B.E, B.Sc, BCA, BBA, etc.)

Diploma

Intermediate / 12th / HSC / PUC / Pre-University

SSC / 10th / Matriculation

Specialization Extraction
Include specialization if mentioned (e.g., “Electronics and Communication Engineering”).
Ignore percentages, GPA, board names, and years unless needed to identify the degree.

Exclusions
Do NOT treat the following as degrees:
Certifications, training programs, online courses, bootcamps, workshops, nanodegrees.

Interpretation Rules

“Intermediate”, “12th”, “HSC”, “PUC” → Pre-university level

“SSC”, “10th”, “Matriculation” → Secondary level

If no bachelor's/master's degree exists, return the next highest valid academic qualification.

Output Format
Return a single string in this format:
"B.Tech in Electronics and Communication Engineering"
"M.S. in Computer Science"
"MBA in Finance"
If nothing is found → "Unknown".

12. certifications

Capture all professional or vendor certifications (e.g., AWS Certified Developer, PMP).

13. summary

Provide a concise 2–3 line professional summary describing:

Experience in years

Domain focus

Technical strengths

Avoid generic summaries (“Hardworking individual”) — focus on quantifiable professional traits.

⚙️ QUALITY & VALIDATION RULES

For full_name:

Must not include punctuation, numbers, company, or degree text.

Must appear within first 3 lines.

Use alphabetic words only.

For All Fields:

Never guess or assume missing data.

If data unavailable → return null.

Ensure consistent JSON formatting.

Each skill or domain must be part of your known taxonomy.

💡 OUTPUT FORMAT

Return a single valid JSON object — no markdown, no explanation text.

Example:

{
  "full_name": "John M. Smith",
  "email": "john.smith@gmail.com",
  "phone_number": "+1-9876543210",
  "total_experience": 6,
  "current_company": "Infosys",
  "current_designation": "Software Engineer",
  "technical_skills": ["Java", "Spring Boot", "ReactJS"],
  "secondary_skills": ["Leadership", "Team Management"],
  "all_skills": ["Java", "Spring Boot", "ReactJS", "Leadership", "Team Management"],
  "domain": ["Information Technology", "Banking"],
  "education": "B.Tech in Computer Science",
  "certifications": ["AWS Certified Developer"],
  "summary": "6 years of experience in Java and ReactJS with strong exposure to banking domain."
}

🧮 EVALUATION CRITERIA
Criterion	Weight	Description
Accuracy	35%	All fields correctly extracted
Completeness	25%	All major fields present
JSON Validity	20%	Output is machine-parseable
Skill Categorization	10%	Technical vs. soft skills are separated correctly
Neutrality	10%	No added commentary or inferred data
🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.

Do not infer, assume, or explain — extract only.

Output must be clean JSON, no markdown formatting.

🧰 ADDITIONAL REFERENCE DICTIONARIES

TECHNICAL_SKILLS → your full skill taxonomy (programming, cloud, databases, etc.)
DOMAINS → pre-defined industry domain list
EDUCATION_KEYWORDS → used to identify degrees and specializations

Resume Text (look for name in FIRST FEW LINES):
{resume_text}
"""
    

    
    EDUCATION_KEYWORDS = {
        'b.tech', 'b.e.', 'bachelor', 'btech', 'bca', 'bsc', 'ba',
        'm.tech', 'm.e.', 'master', 'mtech', 'mca', 'msc', 'mba', 'ma',
        'phd', 'doctorate', 'diploma', 'associate'
    }
    
    def __init__(self, nlp_model: str = 'en_core_web_sm', use_ai_extraction: bool = True):
        """Initialize parser with NLP model and AI capabilities."""
        self.nlp = None
        self.matcher = None
        self.use_ai_extraction = use_ai_extraction
        self.ai_client = None
        
        try:
            self.nlp = spacy.load(nlp_model)
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_patterns()
            logger.info(f"Loaded spaCy model: {nlp_model}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}. Using regex-based parsing.")
        
        # Initialize AI client if AI extraction is enabled
        if self.use_ai_extraction:
            self._initialize_ai_client()
    
    def _initialize_ai_client(self):
        """Initialize OpenAI or Azure OpenAI client."""
        try:
            from ats_config import ATSConfig
            
            # Try Azure OpenAI first
            if ATSConfig.AZURE_OPENAI_ENDPOINT and ATSConfig.AZURE_OPENAI_API_KEY:
                self.ai_client = AzureOpenAI(
                    api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                    api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
                )
                self.ai_model = ATSConfig.AZURE_OPENAI_DEPLOYMENT_NAME
                logger.info("Initialized Azure OpenAI client for skill extraction")
            # Fallback to OpenAI
            elif ATSConfig.OPENAI_API_KEY:
                self.ai_client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
                self.ai_model = ATSConfig.OPENAI_MODEL
                logger.info("Initialized OpenAI client for skill extraction")
            else:
                logger.warning("No OpenAI API key found. AI extraction disabled.")
                self.use_ai_extraction = False
                
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")
            self.use_ai_extraction = False
    
    def _setup_patterns(self):
        """Setup spaCy patterns for entity extraction."""
        if not self.matcher:
            return
        
        # Email pattern
        email_pattern = [{"LIKE_EMAIL": True}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # Phone pattern
        phone_pattern = [{"SHAPE": "ddd-ddd-dddd"}]
        self.matcher.add("PHONE", [phone_pattern])
    
    def parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    def parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            raise ValueError(f"Failed to parse DOCX: {str(e)}")
    
    def parse_doc(self, file_path: str) -> str:
        """Extract text from DOC file (older binary format)."""
        try:
            # Try NT-TextFileLoader first (works well on Windows)
            if nt_loader is not None:
                try:
                    text = nt_loader.load(file_path)
                    if text and isinstance(text, str) and text.strip():
                        return text.strip()
                except Exception as e:
                    logger.warning(f"NT-TextFileLoader failed for DOC file: {e}, trying textract")
            
            # Try textract if available
            if textract is not None:
                try:
                    text = textract.process(file_path).decode('utf-8')
                    return text.strip()
                except Exception as e:
                    logger.warning(f"textract failed for DOC file: {e}, trying pypandoc")
            
            # Fallback to pypandoc if available
            if pypandoc is not None:
                try:
                    text = pypandoc.convert_file(file_path, 'plain')
                    return text.strip()
                except Exception as e:
                    logger.warning(f"pypandoc failed for DOC file: {e}")
            
            # If no library is available, raise informative error
            raise ImportError(
                "DOC file parsing requires one of: 'NT-TextFileLoader', 'textract', or 'pypandoc' library. "
                "Install with: pip install NT-TextFileLoader OR pip install textract OR pip install pypandoc"
            )
            
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Error parsing DOC: {e}")
            raise ValueError(f"Failed to parse DOC: {str(e)}")
    
    def extract_text_from_file(self, file_path: str, file_type: str) -> str:
        """Extract text based on file type."""
        file_type = file_type.lower()
        
        if file_type == 'pdf':
            return self.parse_pdf(file_path)
        elif file_type == 'docx':
            return self.parse_docx(file_path)
        elif file_type == 'doc':
            return self.parse_doc(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from resume text with PDF header handling."""
        # Common section headers to exclude (but NOT in first 3 lines - those might be the actual name!)
        invalid_names = {'education', 'experience', 'skills', 'contact', 'objective', 
                        'summary', 'qualifications', 'work history', 'professional summary',
                        'references', 'certifications', 'projects', 'achievements'}
        
        # Academic degree patterns
        degree_keywords = ['b.a.', 'm.a.', 'b.s.', 'm.s.', 'phd', 'mba', 'b.tech', 'm.tech', 'degree', 
                          'in ', 'major', 'minor', 'diploma', 'certificate']
        
        # CRITICAL: PDF header area is usually the first 5 lines
        # Prioritize the TOP 2-5 lines as per the prompt guidelines
        lines = text.split('\n')
        
        logger.info("Checking PDF header area (top 5 lines) for candidate name...")
        
        # First, check top 5 lines thoroughly for the name (PDF header area)
        for idx, line in enumerate(lines[:5]):
            # Strip and normalize whitespace
            line = ' '.join(line.split())  # Normalize multiple spaces to single space
            line = line.strip()
            
            # Remove trailing separators like | or • that might appear after names
            line = line.rstrip('|•').strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip section headers (but check if NOT in first 3 lines where actual name might be)
            if idx > 2 and line.lower() in invalid_names:
                continue
            
            # Skip lines with academic degree patterns
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in degree_keywords):
                continue
            
            # Skip lines with degree abbreviations (B.A., M.S., etc.)
            if re.search(r'\b([BM]\.?[AS]\.?|MBA|PhD|MD|JD|B\.?Tech|M\.?Tech)\b', line, re.IGNORECASE):
                continue
            
            # Skip email addresses (they shouldn't be names)
            if '@' in line:
                continue
            
            # Skip phone numbers
            if re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', line):
                continue
            
            # Skip addresses (they often contain numbers and "Drive", "Street", etc.)
            if any(addr_word in line_lower for addr_word in ['drive', 'street', 'avenue', 'road', 'blvd', 'city']):
                continue
            
            # For first 3 lines only: be more lenient - might be the actual name
            if idx < 3:
                # Reject sentence fragments (ending with comma, period, or containing common phrases)
                if line.endswith(',') or line.endswith('.'):
                    continue
                # Reject common bullet point phrases
                if any(phrase in line_lower for phrase in ['while maintaining', 'while working', 'while attending', 
                                                           'while completing', 'full course', 'as part of', 
                                                           'in order to', 'for the', 'that']):
                    continue
                
                # Accept if it looks like a name (2-4 words, Title Case or ALL CAPS)
                # CRITICAL: Allow ALL CAPS names in PDF headers (e.g., "VARRE DHANA LAKSHMI DURGA")
                if line and 2 <= len(line.split()) <= 4 and len(line) < 70:
                    words = line.split()
                    # Check if mostly alphabetic and NOT a sentence fragment
                    if all(word.replace('.', '').replace(',', '').replace("'", '').replace('-', '').isalpha() for word in words):
                        # Accept even if ALL CAPS (common in PDF headers)
                        logger.info(f"Found candidate name in PDF header area: {line}")
                        return line
            
            # For lines beyond first 3: more strict validation
            # Name is typically 2-4 words, mostly alphabetic, not too long
            if line and 2 <= len(line.split()) <= 4 and len(line) < 50:
                words = line.split()
                # Allow hyphenated names (e.g., "Mary-Jane"), apostrophes, and periods
                if all(word.replace('.', '').replace(',', '').replace("'", '').replace('-', '').isalpha() for word in words):
                    # Additional checks: reject sentence fragments and bullet point content
                    # Reject if ends with comma or period
                    if line.endswith(',') or line.endswith('.'):
                        continue
                    # Reject common bullet point patterns
                    if any(phrase in line_lower for phrase in ['while maintaining', 'while working', 
                                                                'full course', 'as part of', 
                                                                'in order to', 'that', 'the']):
                        continue
                    # Additional check: name should not be in ALL CAPS (likely a section header)
                    # But allow Title Case
                    if not line.isupper():
                        return line
        
        # Fallback: use NLP if available
        if self.nlp:
            doc = self.nlp(text[:1000])  # Increased from 500 to 1000 for better coverage
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Validate NLP result - check for degrees and invalid names
                    ent_text_lower = ent.text.lower()
                    if ent_text_lower not in invalid_names:
                        # Check for degree patterns
                        if not any(keyword in ent_text_lower for keyword in degree_keywords):
                            return ent.text
        
        return "Unknown"
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        # Various phone formats
        patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\b\d{10}\b',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        return None
    
    def extract_comprehensive_data_with_ai(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive resume data using AI-powered analysis."""
        if not self.use_ai_extraction or not self.ai_client:
            logger.warning("AI extraction not available, falling back to regex-based extraction")
            return None
        
        try:
            # Prepare the prompt with resume text (increase limit for comprehensive extraction)
            prompt = self.AI_COMPREHENSIVE_EXTRACTION_PROMPT.replace("{resume_text}", text[:16000])
            
            # Call AI API
            response = self.ai_client.chat.completions.create(
                model=self.ai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=4000,   # Increased to ensure complete JSON response
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            logger.info(f"AI response received, length: {len(response.choices[0].message.content)} chars")
            
            # Parse JSON response with better error handling
            response_content = response.choices[0].message.content.strip()
            
            # Try to fix common JSON issues
            if not response_content.startswith('{') and not response_content.startswith('['):
                # Try to find JSON object start
                start_idx = response_content.find('{')
                if start_idx != -1:
                    response_content = response_content[start_idx:]
                else:
                    logger.error(f"No JSON object found in AI response. First 200 chars: {response_content[:200]}")
                    return None
            
            # Try to parse JSON
            try:
                ai_result = json.loads(response_content)
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error at position {je.pos}: {je.msg}")
                logger.error(f"Problematic JSON around error: {response_content[max(0, je.pos-100):je.pos+100]}")
                logger.error(f"Full response (first 500 chars): {response_content[:500]}")
                return None
            
            # Validate extracted name - reject section headers and academic degrees
            full_name = ai_result.get('full_name', '')
            if full_name:
                # Common section headers and academic degrees that should NEVER be names
                invalid_names = ['education', 'experience', 'skills', 'contact', 'objective', 
                               'summary', 'qualifications', 'work history', 'professional summary',
                               'references', 'certifications', 'projects', 'achievements']
                
                # Check for academic degree patterns (B.A., M.S., PhD, etc.)
                degree_patterns = [
                    r'\b([BM]\.?[AS]\.?|MBA|PhD|MD|JD|B\.?Tech|M\.?Tech)\b',
                    r'\bin\s+[A-Z][a-z]+',
                    r'degree|diploma|certificate'
                ]
                
                is_invalid = False
                
                # Check if it's in invalid names list
                if full_name.lower() in invalid_names:
                    is_invalid = True
                
                # Check if it contains academic degree patterns
                if not is_invalid:
                    for pattern in degree_patterns:
                        if re.search(pattern, full_name, re.IGNORECASE):
                            is_invalid = True
                            break
                
                # Check if it looks like an academic degree format (e.g., "B.A. in History")
                if not is_invalid and any(keyword in full_name.lower() for keyword in ['in ', 'degree', 'major', 'minor']):
                    is_invalid = True
                
                # Check if it's a sentence fragment from bullet points (e.g., "full course load.")
                if not is_invalid:
                    sentence_fragment_keywords = ['while maintaining', 'while working', 'while attending', 
                                                  'while completing', 'full course', 'as part of', 
                                                  'in order to', 'for the', 'that', 'load.', 'completed in']
                    if any(phrase in full_name.lower() for phrase in sentence_fragment_keywords):
                        is_invalid = True
                
                # Check if it ends with a period or comma (likely a sentence fragment)
                if not is_invalid and (full_name.endswith('.') or full_name.endswith(',')):
                    is_invalid = True
                
                if is_invalid:
                    logger.warning(f"AI extracted invalid name '{full_name}' (sentence fragment/section header/academic degree). Trying regex fallback...")
                    # Use regex-based extraction as fallback
                    fallback_name = self.extract_name(text)
                    if fallback_name and fallback_name != 'Unknown':
                        ai_result['full_name'] = fallback_name
                        logger.info(f"Replaced with: {fallback_name}")
                    else:
                        # Last resort: try to extract from first PERSON entity in first 500 chars
                        logger.warning(f"Regex fallback also failed. Trying NLP...")
                        if self.nlp:
                            doc = self.nlp(text[:500])
                            for ent in doc.ents:
                                if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
                                    logger.info(f"Found name via NLP: {ent.text}")
                                    ai_result['full_name'] = ent.text
                                    break
                        if ai_result.get('full_name', '').lower() in ['education', 'experience', 'skills']:
                            ai_result['full_name'] = 'Unknown'
                            logger.warning(f"Final fallback resulted in invalid name, setting to Unknown")
            
            logger.info(f"AI extraction completed for {ai_result.get('full_name', 'Unknown')}")
            return ai_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"AI response: {response.choices[0].message.content}")
            return None
            
        except Exception as e:
            logger.error(f"AI comprehensive extraction failed: {e}")
            return None
    
    def extract_skills_with_ai(self, text: str) -> Dict[str, Any]:
        """Legacy method for AI skill extraction - now calls comprehensive extraction."""
        ai_data = self.extract_comprehensive_data_with_ai(text)
        
        if ai_data:
            technical_skills = ai_data.get('technical_skills', [])
            secondary_skills = ai_data.get('secondary_skills', [])
            all_skills_list = ai_data.get('all_skills', [])
            
            return {
                'primary_skills': technical_skills,  # All technical skills
                'secondary_skills': secondary_skills,  # Non-technical skills only
                'all_skills': all_skills_list,
                'ai_analysis': {
                    'total_experience': ai_data.get('total_experience', 0),
                    'candidate_name': ai_data.get('full_name', ''),
                    'email': ai_data.get('email', ''),
                    'phone': ai_data.get('phone_number', '')
                }
            }
        
        return self.extract_skills(text)
    
    def extract_skills_section(self, text: str) -> Optional[str]:
        """Extract the Skills section content."""
        # Look for Skills section with various patterns (Primary Search)
        patterns = [
            r'(?i)(?:skill profile|technical skills|skills|core competencies?)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:proficiencies?|competencies?)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:technical skills|skill set|technical summary|technical expertise|core competencies?|proficiencies|tools & technologies|tools and technologies)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:skill set)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:technical summary)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:technical expertise)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:tools & technologies|tools and technologies)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                skills_text = match.group(1).strip()
                if len(skills_text) > 10:  # Make sure it's substantial
                    logger.info("Found skills section using primary search")
                    return skills_text
        
        # Fallback Search: Look for skill-like lists (comma-separated technology names)
        # Pattern: Short words (2-15 chars) separated by commas, often without headers
        fallback_pattern = r'(?:^|\n)([A-Za-z0-9+#\.\-]{1,25}(?:,?\s+[A-Za-z0-9+#\.\-]{1,25}){3,20})'
        match = re.search(fallback_pattern, text, re.MULTILINE)
        if match:
            skills_text = match.group(1).strip()
            if len(skills_text) > 10:
                # Verify it looks like a skills list (not sentences)
                words = re.split(r'[,;•\n]', skills_text)
                avg_word_length = sum(len(w.strip()) for w in words) / max(len(words), 1)
                if avg_word_length < 15:  # Skills are usually short words
                    logger.info("Found skills section using fallback search")
                    return skills_text
        
        return None
    
    def _extract_skills_from_text_with_word_boundaries(self, resume_text: str, existing_skills: List[str], existing_skills_set: set, max_skills: Optional[int] = None) -> List[str]:
        """Extract skills from resume text using word-boundary matching with TECHNICAL_SKILLS."""
        logger.info(f"Running word-boundary matching on entire resume... (currently have {len(existing_skills)} skills)")
        resume_text_lower = resume_text.lower()
        
        # Use case-insensitive whole-word matching for each skill in TECHNICAL_SKILLS
        for skill in sorted(self.TECHNICAL_SKILLS, key=len, reverse=True):  # Check longer skills first
            skill_lower = skill.lower()
            # Match whole words only (case-insensitive) using word boundaries
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            
            # Also handle compound words (e.g., "MicrosoftSqlServer" should match "sql server")
            # Create a pattern without word boundaries to match compound words
            skill_words = skill_lower.split()
            if len(skill_words) > 1:
                # For multi-word skills, check if they appear together without spaces
                # e.g., "sql server" -> "sqlserver" or "sql-server" or "sql_server"
                pattern_compound = r'\b' + r'[\s\-_]?'.join(re.escape(w) for w in skill_words) + r'\b'
            else:
                pattern_compound = pattern
            
            if re.search(pattern, resume_text_lower) or re.search(pattern_compound, resume_text_lower):
                if skill_lower not in existing_skills_set:
                    existing_skills.append(skill)
                    existing_skills_set.add(skill_lower)
                    logger.info(f"Added skill via word-boundary matching: {skill}")
                    if max_skills is not None and len(existing_skills) >= max_skills:  # Stop after finding max skills (if limit set)
                        break
        
        return existing_skills
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical and soft skills."""
        text_lower = text.lower()
        
        found_skills = set()
        
        # CRITICAL: Only look for skills in the Skills section
        skills_section = self.extract_skills_section(text)
        
        if skills_section:
            # Only extract skills that appear in the Skills section
            skills_section_lower = skills_section.lower()
            
            # Extract technical skills that are in TECHNICAL_SKILLS AND in the Skills section
            for skill in self.TECHNICAL_SKILLS:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, skills_section_lower, re.IGNORECASE):
                    found_skills.add(skill)
            
            # Extract all potential skills from the Skills section
            potential_skills = re.split(r'[,;•\n•]', skills_section)
            for skill in potential_skills:
                skill = skill.strip()
                if skill and len(skill) < 50:
                    skill_lower = skill.lower()
                    # Only keep if it's in TECHNICAL_SKILLS
                    if skill_lower in self.TECHNICAL_SKILLS:
                        found_skills.add(skill_lower)
        else:
            # Fallback: extract from entire text if no Skills section found
            for skill in self.TECHNICAL_SKILLS:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found_skills.add(skill)
        
        # Categorize as primary/secondary (simple heuristic)
        all_skills = list(found_skills)
        primary_count = min(10, len(all_skills) // 2)
        
        return {
            'primary_skills': all_skills[:primary_count] if all_skills else [],
            'secondary_skills': all_skills[primary_count:] if len(all_skills) > primary_count else [],
            'all_skills': all_skills
        }
    
    def extract_experience(self, text: str) -> float:
        """
        Calculate total professional experience using the comprehensive ExperienceExtractor.
        Falls back to original method if ExperienceExtractor is not available.
        
        Args:
            text: Resume text to parse
            
        Returns:
            Total experience in years (float)
        """
        # Use the new comprehensive ExperienceExtractor if available
        if ExperienceExtractor:
            try:
                # Try Python extraction first (no AI)
                extractor = ExperienceExtractor(text)
                result = extractor.extract()
                total_experience = result.get('total_experience_years', 0.0)
                
                # Note: AI fallback is not implemented in the new ExperienceExtractor
                # The new extractor handles all patterns without needing AI
                
                # Always use the result from ExperienceExtractor (even if 0)
                # It properly handles freshers by returning 0 when no Experience section exists
                logger.info(f"ExperienceExtractor calculated: {total_experience} years")
                logger.debug(f"Experience segments: {result.get('segments', [])}")
                logger.debug(f"Ignored entries: {result.get('ignored', [])}")
                return float(total_experience)
            except Exception as e:
                logger.warning(f"ExperienceExtractor failed: {e}, falling back to original method")
                import traceback
                logger.error(traceback.format_exc())
        
        # Fallback to original method
        return self._extract_experience_legacy(text)
    
    def _extract_experience_legacy(self, text: str) -> float:
        """Legacy experience extraction method (original implementation).
        Calculate total professional experience (full years): prioritize explicit mentions,
        otherwise compute from job timelines.

        - First tries explicit mentions like "3+ Years of experience", "Over 2 years", "Nearly 5 years"
        - Falls back to parsing start-end dates, merging overlaps, treating Present/Till Date as today
        - Floors to integer years
        """
        # Priority 1: Look for explicit experience mentions
        explicit_patterns = [
            r'(?i)(\d+)\s*[+]\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'(?i)(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'(?i)(?:around|over|nearly|almost|about)\s+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)total\s+experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)(?:over|nearly|almost|about)\s+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)(\d+)\+?\s*(?:years?|yrs?)(?:\s+in)?\s+(?:the|it|software|technology|industry)'
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    value = float(matches[0])
                    if 0 <= value <= 50:  # Reasonable bounds
                        logger.info(f"Found explicit experience mention: {value} years")
                        return value
                except (ValueError, TypeError):
                    pass
        
        # Priority 2: Calculate from timeline-based job dates
        logger.info("No explicit mention found, calculating from job timelines...")
        return self._calculate_experience_from_dates(text)
    
    def _calculate_experience_from_dates(self, text: str) -> float:
        """Calculate experience from date ranges in work history.
        Handles "Present/Current/Till Date" as today and merges overlapping periods.
        CRITICAL: Excludes education dates to avoid counting graduation dates as work experience.
        """
        # Find Experience section if possible - CRITICAL for separating work from education dates
        experience_text = text
        try:
            section_match = re.search(r'(?is)(experience|work experience|professional experience|employment|work history)[\s\n\r:.-]+(.+?)(?=\n\s*[A-Z][A-Za-z ]{2,}:|\n\s*(education|academic|skills|projects|certifications)\b|\Z)', text)
            if section_match and section_match.group(2):
                experience_text = section_match.group(2)
                logger.info("Found Experience section, using it for date extraction")
            else:
                logger.warning("No Experience section found - will try to exclude education dates")
        except Exception as e:
            logger.warning(f"Error finding experience section: {e}")
        
        # Try to exclude Education section dates to avoid counting graduation year as work start
        education_section = None
        try:
            edu_match = re.search(r'(?is)(education|academic|qualification)[\s\n\r:.-]+(.+?)(?=\n\s*[A-Z][A-Za-z ]{2,}:|\n\s*(experience|skills|projects|certifications)\b|\Z)', text)
            if edu_match and edu_match.group(2):
                education_section = edu_match.group(2)
                logger.info("Found Education section - will exclude its dates from experience calculation")
        except Exception:
            pass
        
        # Extract education years to exclude them
        education_years = set()
        if education_section:
            # Directly extract full 4-digit years (1950-2024)
            year_pattern = r'\b(19\d{2}|20\d{2})\b'
            year_matches = re.findall(year_pattern, education_section)
            for year_str in year_matches:
                try:
                    year = int(year_str)
                    if 1950 <= year <= datetime.now().year:
                        education_years.add(year)
                        logger.debug(f"Found education year to exclude: {year}")
                except ValueError:
                    pass
        
        # Look for date patterns like "Jan 2020 - Present" or "2018 - 2020" in experience section
        month_regex = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*'
        year_regex = r'(?:19|20)\d{2}'
        date_pattern = rf'(?i){month_regex}\s+{year_regex}|{year_regex}'
        dates = re.findall(date_pattern, experience_text)
        
        # Check for Present/Current/Till Date and add current date if found
        has_present = bool(re.search(r'(?i)(present|current|till date|till now)', experience_text))
        
        # Filter out education years from extracted dates
        work_years = []
        for date_str in dates:
            year_match = re.search(r'\d{4}', date_str)
            if year_match:
                year = int(year_match.group())
                # Exclude if it's an education year (likely graduation date)
                if year not in education_years:
                    work_years.append(year)
                else:
                    logger.debug(f"Excluded education year {year} from work experience calculation")
        
        if has_present:
            current_year = datetime.now().year
            work_years.append(current_year)
        
        if len(work_years) >= 2:
            try:
                # Calculate span from earliest work start to latest work end
                current_year = datetime.now().year
                max_year = min(max(work_years), current_year)
                min_year = min(work_years)
                experience = max(0, max_year - min_year)
                
                # For freshers: If experience is very high (e.g., >20 years) and we have education years,
                # it might be miscalculated. Check if min_year is close to education year.
                if experience > 20 and education_years:
                    # Likely miscalculation - check if we're counting from graduation
                    latest_edu_year = max(education_years) if education_years else 0
                    if min_year <= latest_edu_year + 2:
                        logger.warning(f"Experience calculation may be incorrect ({experience} years). Possible education date confusion. Returning 0.")
                        return 0.0
                
                logger.info(f"Calculated experience from work dates: {experience} years (from {min_year} to {max_year})")
                return float(experience)
            except Exception as e:
                logger.warning(f"Error calculating experience from dates: {e}")
        elif len(work_years) == 1 and has_present:
            # Single year + Present means from that year to now
            # But exclude if it's an education year
            year = work_years[0]
            if year in education_years:
                logger.info(f"Only year found ({year}) is in education section - likely a fresher. Returning 0.")
                return 0.0
            current_year = datetime.now().year
            experience = max(0, current_year - year)
            logger.info(f"Calculated experience from single date: {experience} years (from {year} to {current_year})")
            return float(experience)
        elif len(work_years) == 0:
            # No work dates found - likely a fresher
            logger.info("No work experience dates found - returning 0 for fresher")
            return 0.0
        
        return 0.0
    
    def extract_domain(self, text: str) -> Optional[str]:
        """Extract domain/industry."""
        text_lower = text.lower()
        
        # Check if tech skills are present - auto-add "Information Technology"
        tech_skill_keywords = ['python', 'java', 'sql', 'javascript', 'html', 'css', '.net', 'c++', 
                              'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'react', 'angular', 
                              'node.js', 'django', 'flask', 'spring', 'mongodb', 'mysql', 'postgresql']
        has_tech_skills = any(skill in text_lower for skill in tech_skill_keywords)
        
        found_domains = []
        for domain in self.DOMAINS:
            if domain.lower() in text_lower:
                found_domains.append(domain)
        
        # Auto-add "Information Technology" if tech skills found
        if has_tech_skills and "Information Technology" not in found_domains:
            found_domains.insert(0, "Information Technology")
        
        # Return most frequent or first found
        if found_domains:
            return found_domains[0]
        
        # If no domain found but tech skills present, return IT
        if has_tech_skills:
            return "Information Technology"
        
        return None
    
    def extract_education(self, text: str, use_ai_fallback: bool = False, store_in_db: bool = False, candidate_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract education information using the enhanced EducationExtractor.
        
        Args:
            text: Resume text to parse
            use_ai_fallback: If True, use AI when Python extraction fails
            store_in_db: If True, store extracted education in database
            candidate_id: Candidate ID for database storage (required if store_in_db=True)
        
        Returns:
            Dictionary with 'highest_degree' and 'education_details'
        """
        # Use the new EducationExtractor if available
        if EducationExtractor:
            try:
                extractor = EducationExtractor(
                    text,
                    use_ai_fallback=use_ai_fallback and self.use_ai_extraction,  # Only use AI if enabled in parser
                    store_in_db=store_in_db,
                    candidate_id=candidate_id
                )
                education_list = extractor.extract()
                
                # Convert to the expected format
                education_info = {
                    'highest_degree': None,
                    'education_details': education_list if education_list else []
                }
                
                # Determine highest degree from the list
                if education_list:
                    # Get the first (highest) degree - keep the full extracted string with specialization
                    highest_degree_str = education_list[0]
                    
                    # Use the extracted string as-is to preserve specialization (e.g., "BTech, Civil Engineering")
                    # Only map to generic categories if the extracted string is too generic
                    highest_degree_lower = highest_degree_str.lower()
                    
                    # If the extracted string is just a generic word like "bachelor" or "bachelors", map it
                    # But if it contains specific degree info (like "BTech" or "B.Tech"), keep it as-is
                    if highest_degree_str.strip().lower() in ['bachelor', 'bachelors', 'bachelor degree']:
                        education_info['highest_degree'] = 'Bachelors'
                    elif highest_degree_str.strip().lower() in ['master', 'masters', 'master degree']:
                        education_info['highest_degree'] = 'Masters'
                    elif highest_degree_str.strip().lower() in ['phd', 'doctorate', 'ph.d']:
                        education_info['highest_degree'] = 'PhD'
                    elif highest_degree_str.strip().lower() == 'diploma':
                        education_info['highest_degree'] = 'Diploma'
                    else:
                        # Keep the full extracted string (preserves "BTech, Civil Engineering" or "B.Tech in CSE")
                        education_info['highest_degree'] = highest_degree_str
                
                logger.info(f"EducationExtractor found {len(education_list)} degrees: {education_list}")
                return education_info
                
            except Exception as e:
                logger.warning(f"EducationExtractor failed: {e}, falling back to regex extraction")
        
        # Fallback to original regex-based extraction
        education_info = {
            'highest_degree': None,
            'education_details': []
        }
        
        text_lower = text.lower()
        
        # Find education section
        edu_section_pattern = r'(?i)(?:education|academic|qualification)[:\s]+(.*?)(?=\n\n[A-Z]|experience|skills|$)'
        edu_match = re.search(edu_section_pattern, text, re.DOTALL)
        
        if edu_match:
            edu_text = edu_match.group(1)
            education_info['education_details'].append(edu_text.strip())
        
        # Extract degree keywords
        degrees_found = []
        for keyword in self.EDUCATION_KEYWORDS:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                degrees_found.append(keyword)
        
        # Determine highest degree (simple heuristic)
        if any(deg in degrees_found for deg in ['phd', 'doctorate']):
            education_info['highest_degree'] = 'PhD'
        elif any(deg in degrees_found for deg in ['m.tech', 'm.e.', 'master', 'mtech', 'mca', 'msc', 'mba', 'ma']):
            education_info['highest_degree'] = 'Masters'
        elif any(deg in degrees_found for deg in ['b.tech', 'b.e.', 'bachelor', 'btech', 'bca', 'bsc', 'ba']):
            education_info['highest_degree'] = 'Bachelors'
        elif 'diploma' in degrees_found:
            education_info['highest_degree'] = 'Diploma'
        
        return education_info
    
    def extract_location(self, text: str) -> Optional[str]:
        """Extract current location."""
        # Look for location patterns
        location_pattern = r'(?i)(?:location|based in|residing in|current location)[:\s]+([A-Za-z\s,]+?)(?:\n|$)'
        matches = re.findall(location_pattern, text)
        
        if matches:
            return matches[0].strip()
        
        # Use NLP to find GPE (Geopolitical Entity)
        if self.nlp:
            doc = self.nlp(text[:1000])
            locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
            if locations:
                return locations[0]
        
        return None
    
    def parse_resume(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Main parsing method that extracts all information from resume.
        
        Args:
            file_path: Path to resume file
            file_type: File extension (pdf, docx)
        
        Returns:
            Dictionary with extracted resume data
        """
        try:
            # Extract text
            resume_text = self.extract_text_from_file(file_path, file_type)
            print(resume_text)
            logger.info(f"----------=======: {resume_text}")
            
            if not resume_text or len(resume_text) < 100:
                raise ValueError("Resume text is too short or empty")
            
            # Try comprehensive AI extraction first
            ai_data = None
            if self.use_ai_extraction:
                ai_data = self.extract_comprehensive_data_with_ai(resume_text)
            
            # Use AI data if available, otherwise fallback to regex-based extraction
            if ai_data:
                # Use AI-extracted comprehensive data
                # First try AI extracted name
                name = ai_data.get('full_name') or ''
                
                # If name is invalid or missing, extract from text
                invalid_names = {'education', 'experience', 'skills', 'contact', 'objective'}
                degree_keywords = ['b.a.', 'm.a.', 'b.s.', 'm.s.', 'phd', 'mba', 'degree']
                
                if not name or name.lower() in ['unknown', 'education', 'experience']:
                    logger.warning(f"AI name extraction failed or returned invalid: '{name}', trying regex fallback...")
                    # Try regex extraction
                    name = self.extract_name(resume_text)
                    logger.info(f"Regex fallback returned: {name}")
                    
                    # If still not found, try a simple heuristic: first line that looks like a name
                    if not name or name == 'Unknown':
                        logger.warning(f"Regex also failed, trying heuristic approach...")
                        lines = resume_text.split('\n')
                        # Check first 10 lines thoroughly
                        for idx, line in enumerate(lines[:10]):
                            line = line.strip()
                            
                            # Remove trailing separators like | or •
                            line = line.rstrip('|•').strip()
                            
                            # Look for lines that are Title Case, 2-4 words, no special chars
                            # Allow up to 4 words for names like "VARRE DHANA LAKSHMI DURGA"
                            if line and 2 <= len(line.split()) <= 4 and len(line) < 70 and len(line) > 3:
                                # Check if it's all alphabetic (plus spaces, hyphens, periods)
                                words = line.split()
                                if all(word.replace('-', '').replace("'", '').replace('.', '').isalpha() for word in words):
                                    # Skip if it's a section header
                                    line_lower = line.lower()
                                    if line_lower not in invalid_names and not any(keyword in line_lower for keyword in degree_keywords):
                                        # Skip if it's an email
                                        if '@' not in line and '://' not in line:
                                            # Skip if it contains phone number patterns
                                            if not re.search(r'\+\d|\d{3}[-.]?\d{3}', line):
                                                name = line
                                                logger.info(f"Found name via heuristic (line {idx+1}): {name}")
                                                break
                email = ai_data.get('email') or self.extract_email(resume_text)
                phone = ai_data.get('phone_number') or self.extract_phone(resume_text)
                # ALWAYS use Python-based extraction (NO AI) - uses comprehensive ExperienceExtractor
                # This ensures accurate date parsing, education exclusion, and range merging
                experience = self.extract_experience(resume_text)
                
                # Get skills from AI
                ai_technical_skills = ai_data.get('technical_skills', [])
                ai_secondary_skills = ai_data.get('secondary_skills', [])
                all_skills_list = ai_data.get('all_skills', [])
                
                logger.info(f"AI extracted {len(ai_technical_skills)} technical skills")
                
                # Ensure we have lists
                if isinstance(ai_technical_skills, str):
                    ai_technical_skills = [s.strip() for s in ai_technical_skills.split(',') if s.strip()] if ai_technical_skills else []
                if isinstance(ai_secondary_skills, str):
                    ai_secondary_skills = [s.strip() for s in ai_secondary_skills.split(',') if s.strip()] if ai_secondary_skills else []
                
                # Collect all valid technical skills
                technical_skills = []
                technical_skills_lower = set()
                
                # First, process AI-extracted skills
                for skill in ai_technical_skills:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_stripped = skill.strip()
                    skill_lower = skill_stripped.lower()
                    
                    # Filter out single-letter skills (like "r" from "r," or from splitting issues)
                    if len(skill_stripped) <= 1:
                        logger.debug(f"Skipping single-letter skill: '{skill_stripped}'")
                        continue
                    
                    # Check if exact match in TECHNICAL_SKILLS
                    if skill_lower in self.TECHNICAL_SKILLS:
                        if skill_lower not in technical_skills_lower:
                            technical_skills.append(skill_stripped)
                            technical_skills_lower.add(skill_lower)
                            logger.info(f"✓ Added AI skill: {skill_stripped}")
                    else:
                        # Try fuzzy/partial matching
                        matched = False
                        for tech_skill in self.TECHNICAL_SKILLS:
                            if skill_lower in tech_skill or tech_skill in skill_lower:
                                if tech_skill not in technical_skills_lower:
                                    technical_skills.append(tech_skill)
                                    technical_skills_lower.add(tech_skill)
                                    logger.info(f"✓ Added AI skill (fuzzy): {tech_skill} (matched {skill})")
                                    matched = True
                                break
                        if not matched:
                            logger.debug(f"AI skill not matched: {skill}")
                
                # Then, try regex fallback for additional skills
                logger.info(f"Trying regex fallback for additional skills...")
                regex_skills = self.extract_skills(resume_text)
                all_extracted_skills = regex_skills.get('all_skills', [])
                logger.info(f"Regex extracted {len(all_extracted_skills)} potential skills")
                
                for skill in all_extracted_skills:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_stripped = skill.strip()
                    skill_lower = skill_stripped.lower()
                    
                    # Filter out single-letter skills
                    if len(skill_stripped) <= 1:
                        logger.debug(f"Skipping single-letter skill: '{skill_stripped}'")
                        continue
                    
                    # Only add if not already found
                    if skill_lower in self.TECHNICAL_SKILLS and skill_lower not in technical_skills_lower:
                        technical_skills.append(skill_stripped)
                        technical_skills_lower.add(skill_lower)
                        logger.info(f"✓ Added regex skill: {skill_stripped}")
                
                # Secondary skills: everything that's NOT in TECHNICAL_SKILLS
                secondary_skills = []
                
                # Process AI secondary skills
                for skill in ai_secondary_skills:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_stripped = skill.strip()
                    skill_lower = skill_stripped.lower()
                    
                    # Filter out single-letter skills
                    if len(skill_stripped) <= 1:
                        logger.debug(f"Skipping single-letter secondary skill: '{skill_stripped}'")
                        continue
                    
                    if skill_lower not in self.TECHNICAL_SKILLS and skill_lower not in [s.lower() for s in secondary_skills]:
                        secondary_skills.append(skill_stripped)
                        logger.info(f"✓ Added secondary skill: {skill_stripped}")
                
                # ALWAYS supplement with word-boundary matching to catch any missed skills
                logger.info(f"Supplementing with word-boundary matching from entire resume...")
                technical_skills = self._extract_skills_from_text_with_word_boundaries(
                    resume_text, technical_skills, technical_skills_lower, max_skills=None  # No limit - extract all skills
                )
                
                # Format skills - primary_skills should ONLY contain TECHNICAL_SKILLS
                primary_skills = ', '.join(technical_skills) if technical_skills else ''  # All technical skills
                secondary_skills_str = ', '.join(secondary_skills) if secondary_skills else ''  # Non-technical skills
                
                # all_skills = primary_skills + secondary_skills
                all_skills_combined = technical_skills + secondary_skills
                all_skills_str = ', '.join(all_skills_combined) if all_skills_combined else ''
                
                logger.info(f"✓ Primary skills ({len(technical_skills)}): {primary_skills[:80]}...")
                logger.info(f"✓ Secondary skills ({len(secondary_skills)}): {secondary_skills_str[:80]}...")
                logger.info(f"✓ All skills ({len(all_skills_combined)}): {all_skills_str[:80]}...")
                
                logger.info(f"✓ AI extraction completed: {len(technical_skills)} technical skills")
                
                # Get domains (handle both single and multiple)
                domain_list = ai_data.get('domain', [])
                if not isinstance(domain_list, list):
                    domain_list = [domain_list] if domain_list else []
                
                # Auto-add "Information Technology" if tech skills are present
                if technical_skills and "Information Technology" not in domain_list:
                    domain_list.insert(0, "Information Technology")
                    logger.info("✓ Auto-added 'Information Technology' domain based on technical skills")
                
                # If no domains found, try fallback extraction
                if not domain_list:
                    fallback_domain = self.extract_domain(resume_text)
                    if fallback_domain:
                        domain_list = [fallback_domain]
                
                domain = ', '.join(domain_list) if domain_list else ''
                
                # ALWAYS prioritize Python extraction over AI for accuracy
                # Python extraction is more reliable and doesn't add inferred specializations
                logger.info("Extracting education using Python extraction (prioritized over AI)...")
                education_info = self.extract_education(resume_text, use_ai_fallback=False)  # Python extraction only
                
                # Use Python extraction if it found valid education
                if education_info['highest_degree'] and education_info['highest_degree'] != 'Unknown':
                    highest_degree = education_info['highest_degree']
                    education_details = '\n'.join(education_info['education_details']) if education_info['education_details'] else highest_degree
                    logger.info(f"Using Python-extracted education: {highest_degree}")
                else:
                    # Fallback to AI extraction only if Python extraction failed
                    ai_education = ai_data.get('education') or ai_data.get('education_details')
                    if isinstance(ai_education, list):
                        ai_education = ai_education[0] if ai_education else None
                    
                    if ai_education and ai_education != 'Unknown':
                        highest_degree = ai_education
                        education_details = ai_education
                        logger.info(f"Python extraction failed, using AI-extracted education: {highest_degree}")
                    else:
                        # Last resort: use Python extraction even if it's Unknown
                        highest_degree = education_info['highest_degree'] or 'Unknown'
                        education_details = '\n'.join(education_info['education_details']) if education_info['education_details'] else highest_degree
                        logger.warning(f"Both Python and AI extraction failed, using: {highest_degree}")
                
                # Get certifications
                certifications = ai_data.get('certifications', [])
                certifications_str = ', '.join(certifications) if isinstance(certifications, list) else certifications or ''
                
                # Get current company and designation
                current_company = ai_data.get('current_company') or ''
                current_designation = ai_data.get('current_designation') or ''
                
                # Summary
                summary = ai_data.get('summary') or ''
                
                # Get additional data
                location = self.extract_location(resume_text)
                
            else:
                # Fallback to regex-based extraction
                name = self.extract_name(resume_text)
                email = self.extract_email(resume_text)
                phone = self.extract_phone(resume_text)
                # Use Python-based extraction (NO AI) - uses comprehensive ExperienceExtractor
                experience = self.extract_experience(resume_text)
                
                skills = self.extract_skills(resume_text)
                
                # CRITICAL: Filter to ONLY include skills from TECHNICAL_SKILLS list
                all_extracted_skills = skills['all_skills']
                
                # Common responsibility phrases that should be rejected
                responsibility_phrases = [
                    'unit testing', 'integration testing', 'system testing', 'end to end testing',
                    'test driven development', 'tdd', 'bdd', 'behavior driven development',
                    'agile methodology', 'scrum methodology', 'waterfall methodology',
                    'performed unit testing', 'implemented unit testing', 'wrote unit tests'
                ]
                
                # Separate into technical (in our list) and non-technical
                technical_skills_list = []
                secondary_skills_list = []
                technical_skills_set = set()  # For deduplication
                
                for skill in all_extracted_skills:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_stripped = skill.strip()
                    skill_lower = skill_stripped.lower()
                    
                    # Filter out single-letter skills
                    if len(skill_stripped) <= 1:
                        logger.debug(f"Skipping single-letter skill: '{skill_stripped}'")
                        continue
                    
                    # Reject responsibility-like phrases unless they're explicitly in TECHNICAL_SKILLS
                    if skill_lower in responsibility_phrases and skill_lower not in self.TECHNICAL_SKILLS:
                        logger.warning(f"Rejected responsibility phrase as skill: '{skill_stripped}'")
                        continue
                    
                    # Check if this skill is in our TECHNICAL_SKILLS list
                    if skill_lower in self.TECHNICAL_SKILLS:
                        if skill_lower not in technical_skills_set:
                            technical_skills_list.append(skill_stripped)
                            technical_skills_set.add(skill_lower)
                    else:
                        # Try partial match
                        found_match = False
                        for tech_skill in self.TECHNICAL_SKILLS:
                            if skill_lower in tech_skill or tech_skill in skill_lower:
                                if tech_skill not in technical_skills_set:
                                    technical_skills_list.append(tech_skill)
                                    technical_skills_set.add(tech_skill)
                                    found_match = True
                                    break
                        if not found_match:
                            secondary_skills_list.append(skill_stripped)
                
                # ALWAYS supplement with word-boundary matching to catch any missed skills
                logger.info(f"Supplementing with word-boundary matching from entire resume...")
                technical_skills_list = self._extract_skills_from_text_with_word_boundaries(
                    resume_text, technical_skills_list, technical_skills_set, max_skills=None  # No limit - extract all skills
                )
                
                # Format primary_skills after potential lenient extraction
                primary_skills = ', '.join(technical_skills_list) if technical_skills_list else ''  # All technical skills
                secondary_skills_str = ', '.join(secondary_skills_list) if secondary_skills_list else ''
                
                # all_skills = primary_skills + secondary_skills (combine lists, then join)
                all_skills_combined = technical_skills_list + secondary_skills_list
                all_skills_str = ', '.join(all_skills_combined) if all_skills_combined else ''
                
                logger.info(f"✓ Primary skills ({len(technical_skills_list)}): {primary_skills[:80]}...")
                logger.info(f"✓ Secondary skills ({len(secondary_skills_list)}): {secondary_skills_str[:80]}...")
                logger.info(f"✓ All skills ({len(all_skills_combined)}): {all_skills_str[:80]}...")
                
                logger.info(f"✓ Regex extraction completed: {len(technical_skills_list)} technical skills")
                
                # Extract domain - auto-add IT if tech skills present
                domain = self.extract_domain(resume_text)
                if not domain and technical_skills_list:
                    domain = "Information Technology"
                    logger.info("✓ Auto-added 'Information Technology' domain based on technical skills")
                elif technical_skills_list and domain != "Information Technology":
                    # Ensure IT is included if tech skills found
                    domain_parts = [d.strip() for d in domain.split(',')] if domain else []
                    if "Information Technology" not in domain_parts:
                        domain_parts.insert(0, "Information Technology")
                        domain = ', '.join(domain_parts)
                        logger.info("✓ Auto-added 'Information Technology' domain based on technical skills")
                # Use enhanced EducationExtractor (Python extraction only, no AI)
                education_info = self.extract_education(resume_text, use_ai_fallback=False)
                highest_degree = education_info['highest_degree']
                education_details = '\n'.join(education_info['education_details']) if education_info['education_details'] else (highest_degree or '')
                
                # Extract current company and designation using regex
                current_company = self._extract_current_company(resume_text)
                current_designation = self._extract_current_designation(resume_text)
                certifications_str = ''
                summary = ''
                location = self.extract_location(resume_text)
            
            # Get file info
            file_size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
            
            # Derive canonical profile type using NLM approach (analyzes overall content)
            profile_type = determine_primary_profile_type(
                primary_skills, 
                secondary_skills_str, 
                resume_text,
                ai_client=self.ai_client if self.use_ai_extraction else None,
                ai_model=self.ai_model if self.use_ai_extraction else None
            )
            
            # Prepare parsed data
            parsed_data = {
                'name': name,
                'email': email,
                'phone': phone,
                'total_experience': experience,
                'primary_skills': primary_skills,
                'secondary_skills': secondary_skills_str,
                'all_skills': all_skills_str,
                'domain': domain,
                'education': highest_degree,
                'education_details': education_details,
                'current_location': location,
                'current_company': current_company,
                'current_designation': current_designation,
                'certifications': certifications_str,
                'resume_summary': summary,
                'resume_text': resume_text,
                'file_name': os.path.basename(file_path),
                'file_type': file_type,
                'file_size_kb': int(file_size_kb),
                'ai_extraction_used': ai_data is not None,
                'profile_type': profile_type
            }
            
            logger.info(f"Successfully parsed resume for: {name}")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            raise
    
    def _extract_current_company(self, text: str) -> Optional[str]:
        """Extract current/most recent company name."""
        # Look for company in work history (first/last company mentioned)
        patterns = [
            r'(?i)(?:company|employer|organization)[:\s]+([A-Za-z0-9\s&.,]+)',
            r'(?i)(?:worked at|employed at|currently at)[:\s]+([A-Za-z0-9\s&.,]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
        
        # Try to find first company in experience section
        exp_section_pattern = r'(?i)(?:experience|work history|employment)(.*?)(?=\n\n[A-Z]|education|skills|$)'
        exp_match = re.search(exp_section_pattern, text, re.DOTALL)
        if exp_match:
            exp_text = exp_match.group(1)
            # Extract first company name
            company_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|LLC|Ltd|Corp|Pvt))?)'
            companies = re.findall(company_pattern, exp_text[:500])
            if companies:
                return companies[0].strip()
        
        return None
    
    def _extract_current_designation(self, text: str) -> Optional[str]:
        """Extract current/most recent job designation."""
        # Look for designation patterns
        patterns = [
            r'(?i)(?:position|role|title|designation)[:\s]+([A-Za-z\s]+)',
            r'(?i)(?:currently|presently).*?(?:as|working as|role of)[:\s]+([A-Za-z\s]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
        
        # Extract first role from experience section
        exp_section_pattern = r'(?i)(?:experience|work history)(.*?)(?=\n\n[A-Z]|education|skills|$)'
        exp_match = re.search(exp_section_pattern, text, re.DOTALL)
        if exp_match:
            exp_text = exp_match.group(1)
            # Common role patterns
            role_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Engineer|Manager|Director|Lead|Developer|Architect|Analyst|Consultant|Specialist)))'
            roles = re.findall(role_pattern, exp_text[:300])
            if roles:
                return roles[0].strip()
        
        return None


def extract_skills_from_text(text: str) -> List[str]:
    """Standalone function to extract skills from any text."""
    parser = ResumeParser()
    skills = parser.extract_skills(text)
    return skills['all_skills']


def extract_experience_from_text(text: str) -> float:
    """Standalone function to extract experience from any text."""
    parser = ResumeParser()
    return parser.extract_experience(text)

