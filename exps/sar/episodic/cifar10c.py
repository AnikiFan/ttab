class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        seed=[2024],
        main_file=[
            "run_exp.py",
            ],
        job_name=[
            "sar_cifar10c_episodic_oracle_model_selection",
        ],
        base_data_name=[
            "cifar10",
        ],
        data_names=[
            "cifar10_c_deterministic-gaussian_noise-5",
            "cifar10_c_deterministic-shot_noise-5",
            "cifar10_c_deterministic-impulse_noise-5",
            "cifar10_c_deterministic-defocus_blur-5",
            "cifar10_c_deterministic-glass_blur-5",
            "cifar10_c_deterministic-motion_blur-5",
            "cifar10_c_deterministic-zoom_blur-5",
            "cifar10_c_deterministic-snow-5",
            "cifar10_c_deterministic-frost-5",
            "cifar10_c_deterministic-fog-5",
            "cifar10_c_deterministic-brightness-5",
            "cifar10_c_deterministic-contrast-5",
            "cifar10_c_deterministic-elastic_transform-5",
            "cifar10_c_deterministic-pixelate-5",
            "cifar10_c_deterministic-jpeg_compression-5",
        ],
        model_name=[
            "resnet26",
        ],
        model_adaptation_method=[
            "sar",
        ],
        model_selection_method=[
            "oracle_model_selection", 
        ],
        offline_pre_adapt=[
            "false",
        ],
        data_wise=["batch_wise"],
        batch_size=[64],
        episodic=[
            "true",
        ],
        inter_domain=["HomogeneousNoMixture"],
        non_iid_ness=[0.1],
        non_iid_pattern=["class_wise_over_domain"],
        python_path=["/root/miniconda3/bin/python"],
        data_path=["./datasets/"],
        ckpt_path=[
            "./pretrained_ckpts/classification/resnet26_with_head/cifar10/rn26_bn.pth",
        ],
        lr=[
            [1e-3], 
            [5e-4], 
            [1e-4],
        ],
        n_train_steps=[50],
        intra_domain_shuffle=["true"],
        record_preadapted_perf=["true"],
        device=[
            "cuda:0",
        ],
        gradient_checkpoint=["false"],
    )
