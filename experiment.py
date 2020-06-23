from argparse import ArgumentParser
from training_module import TrainingModule, DistilledTrainingModule
import torch as th
import pytorch_lightning as pl
import json
import os

def parse_arguments():
    parser = ArgumentParser()
    # program arguments
    parser.add_argument('--data_dir', type=str, default='~/datasets/imagenette2')
    parser.add_argument('--train_data', type=str, default='train')
    parser.add_argument('--val_data', type=str, default='val')
    parser.add_argument('--train_teacher', type=bool, default=False)
    parser.add_argument('--distill', type=bool, default=False)
    parser.add_argument('--student_model', type=str, default='simple')
    parser.add_argument('--teacher_model', type=str, default='densenet121')

    # model specific arguments
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=10)

    # trainer arguments
    parser.add_argument('--default_root_dir', type=str, default='logs')
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--gpus', type=int, default=(1 if th.cuda.is_available() else 0))
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)

    # debug parameters
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    #parser.add_argument('--overfit_batches', type=int, default=1)

    args = parser.parse_args()
    if os.path.isfile('commandline_args.txt'):
        with open('commandline_args.txt', 'r') as f:
            args.__dict__ = json.load(f)

    return args



def main(args):

    training_module = None

    # adapt the models for new image_size before changing image_size
    assert (args.image_size == 224)

    if args.train_teacher:
        training_module = TrainingModule(model_name=args.teacher_model, hparams=args, pre_trained=True)
    elif args.distill:
        training_module = DistilledTrainingModule(student_model_name=args.student_model,
                                                  teacher_model_name=args.teacher_model,
                                                  hparams=args)
    else:
        training_module = TrainingModule(model_name=args.student_model, hparams=args)

    trainer = pl.Trainer.from_argparse_args(args)
    log_name = f'{args.teacher_model if args.train_teacher else args.student_model}_' \
               f'{args.batch_size}_' \
               f'{args.learning_rate}_' \
               f'{("mse" if args.distill else "ce")}_' \
               f'{args.max_epochs}'
    trainer.logger = pl.loggers.TensorBoardLogger('logs/', name=log_name, version=0)

    trainer.fit(training_module)


if __name__ == '__main__':
    args = parse_arguments()
    with open('commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(args)
    main(args)
