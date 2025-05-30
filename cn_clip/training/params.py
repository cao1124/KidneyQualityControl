import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B-32", "ViT-B-16", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    elif model_name in ["ViT-L-14", "ViT-L-14-336"]:
        return {"lr": 4.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    train_data = "/mnt/disk1/yangyixin/code/Chinese-CLIP-master/data/datasets/upbelly_all/lmdb/train"
    val_data = "/mnt/disk1/yangyixin/code/Chinese-CLIP-master/data/datasets/upbelly_all/lmdb/valid"
    resume="/mnt/disk1/yangyixin/code/Chinese-CLIP-master/data/pretrained_weights/clip_cn_vit-b-16.pt" # or specify your customed ckpt path to resume
    reset_data_offset=True
    reset_optimizer=True
    # reset_optimizer=""
    # output options
    output_base_dir="/mnt/disk1/yangyixin/code/Chinese-CLIP-master/exps/upbelly_all/"
    name= 'upbelly_all_bs96' # organcls_50w_bs64
    save_step_frequency=999999999 # disable it
    save_epoch_frequency=5
    log_interval=50
    report_training_batch_acc=True
    # report_training_batch_acc=""
    # training hyper-params
    context_length=52
    warmup=100
    batch_size=96
    valid_batch_size=96
    train_workers=4
    val_workers=4
    accum_freq=1
    lr=2e-5
    wd=0.001
    max_epochs=50 # or you can alternatively specify --max-steps
    valid_step_interval=9999999999999
    valid_epoch_interval=5
    vision_model="ViT-B-16"
    text_model="RoBERTa-wwm-ext-base-chinese"
    use_augment="--use-augment"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        # required=True,
        default=train_data,
        help="Path to the LMDB directory with training data split",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=val_data,
        help="Path to the LMDB directory with validation data split, default to None which disables validation",
    )
    parser.add_argument(
        "--num-workers", type=int, default=train_workers, help="The number of workers for training dataloader."
    )
    parser.add_argument(
        "--valid-num-workers", type=int, default=val_workers, help="The number of workers for validation dataloader (if making validation)."
    )
    parser.add_argument(
        "--logsdir",
        type=str,
        default=output_base_dir,
        help="Where to store logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=name,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=log_interval, help="How often to log loss info."
    )
    parser.add_argument(
        "--report-training-batch-acc", default=report_training_batch_acc, action="store_true", help="Whether to report training batch accuracy."
    )
    parser.add_argument(
        "--batch-size", type=int, default=batch_size, help="Batch size for training per GPU."
    )
    parser.add_argument(
        "--valid-batch-size", type=int, default=valid_batch_size, help="Batch size for validation per GPU."
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Number of steps to train for (in higher priority to --max_epochs)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=max_epochs, help="Number of full epochs to train for (only works if --max_steps is None)."
    )
    parser.add_argument(
        "--valid-step-interval", type=int, default=valid_step_interval, help="The step interval for validation (default to None which disables validation between steps)."
    )
    parser.add_argument(
        "--valid-epoch-interval", type=int, default=valid_epoch_interval, help="The epoch interval for validation (default to 1, set None to disable validation between epochs)."
    )
    parser.add_argument(
        "--context-length", type=int, default=context_length, help="The maximum length of input text (include [CLS] & [SEP] tokens). Default to 52."
    )
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=wd, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=warmup, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync."
    )
    parser.add_argument("--use-augment",
        default=False,
        action="store_true",
        help="Whether to use image augment."
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-epoch-frequency", type=int, default=save_epoch_frequency, help="How often to save checkpoints by epochs."
    )
    parser.add_argument(
        "--save-step-frequency", type=int, default=save_step_frequency, help="How often to save checkpoints by steps."
    )
    parser.add_argument(
        "--resume",
        default=resume,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        default=reset_optimizer,
        help="If resumed from a checkpoint, whether to reset the optimizer states.",
    )
    parser.add_argument(
        "--reset-data-offset",
        action="store_true",
        default=reset_data_offset,
        help="If resumed from a checkpoint, whether to reset the dataset offset to the beginning.",
    )    
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision."
    )
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--mask-ratio",
        default=0,
        type=float,
        help="Random mask ratio of patches during finetuning. Default to zero which does not mask any patches.",
    )
    parser.add_argument(
        "--clip-weight-path",
        default=None,
        type=str,
        help="The path of openai pretrained weight, used to initialize the image encoder, should be set to None if you do not use pretrained CLIP",
    )    
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=False,
        help="Freeze the weight of vision encoder.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )    
    parser.add_argument(
        "--bert-weight-path",
        default=None,
        type=str,
        help="The path of bert pretrained weight, used to initialize the text encoder, should be set to None if you do not use pretrained BERT",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--use-flash-attention",
        default=False,
        action="store_true",
        help="Enable flash attention."
    )
    parser.add_argument(
        "--accum-freq",
        type=int,
        default=accum_freq,
        help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    # arguments for distributed training
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123, 
        help="Random seed."
    )
    # arguments for distillation
    parser.add_argument(
        "--distillation",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--teacher-model-name",
        type=str,
        default=None,
        help="The name of teacher model."
    )
    parser.add_argument(
        "--kd_loss_weight",
        type=float,
        default=0.5,
        help="Weight of KD loss."
    )
    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.vision_model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
