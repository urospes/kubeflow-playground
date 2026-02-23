    # # MAIN
    # dataset_paths = f"s3://{'/'.join(train_dataset.path.split('/')[:-1])}/"
    # print(dataset_paths)
    # _ = TrainerClient().train(
    #     runtime=TrainerClient().get_runtime("torch-distributed-no-istio"),
    #     initializer=Initializer(
    #         dataset=S3DatasetInitializer(storage_uri=dataset_paths)
    #     ),
    #     trainer=CustomTrainer(
    #         func=train_wrapper_func,
    #         func_args={
    #             "model_save_path": kfp_model.path,
    #             "learning_rate": learning_rate,
    #             "n_epochs": n_epochs,
    #         },
    #         num_nodes=1,
    #         resources_per_node={
    #             "cpu": 1,
    #             "memory": "2Gi",
    #         },
    #         packages_to_install=["pandas==2.3.3"],
    #     ),
    #     options=[Name(name="training-maternity-health")],


from model_registry import ModelRegistry

registry = ModelRegistry(
    server_address="http://model-registry-service.kubeflow-user-example-com.svc.cluster.local",
    port=8080,
    author="uros pesic",
    is_secure=False
)
print(registry)
registry.register_model()
