# MLOps Assignment Labs

This repository includes code of all the labs completed in MLOps course.

## Labs

- [Docker Labs](./Docker_Labs/)
  - [Lab 1](./Docker_Labs/Lab1/) - Basic Docker containerization with a machine learning model. Trains a Support Vector Classifier on the Wine dataset and saves the model.

- [Data Labs](./Data_Labs/)
  - [LLM Data Pipeline](./Data_Labs/LLM_Data_Pipeline/) - Streaming language modeling data pipeline using Hugging Face Datasets and Transformers. Demonstrates memory-efficient processing of large text corpora for LM training.

- [API Labs](./API_Labs/)
  - [FastAPI Labs](./API_Labs/FastAPI_Labs/) - FastAPI-based machine learning inference API. Trains a Support Vector Classifier on the Wine dataset and serves predictions through a `/predict` endpoint.

- [Experiment Tracking Labs](./Experiment_Tracking_Labs/)
  - [W&B Lab](./Experiment_Tracking_Labs/W&B/) - Experiment tracking workflow using TensorFlow/Keras and Weights & Biases. Trains a CNN on CIFAR-10, logs metrics and artifacts, and supports environment-based configuration.

- [GCP Labs](./GCP_Labs/)
  - [Compute Engine Labs](./GCP_Labs/Compute_Engine_Labs/)
    - [Lab 1](./GCP_Labs/Compute_Engine_Labs/Lab1/) - FastAPI Wine Classification API deployment on Google Cloud Platform's Compute Engine. Demonstrates deploying ML models as scalable REST APIs on cloud infrastructure.

- [GitHub Labs](./Github_Labs/)
  - [Lab 2](./Github_Labs/Lab2/) - Automated Model Retraining With GitHub Actions. Demonstrates an end-to-end MLOps loop where a model is tested, retrained, evaluated, versioned, and committed automatically through GitHub Actions.
