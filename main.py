import argparse
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='test', choices=['preprocess', 'train', 'test', 'predict'])
parser.add_argument('--gpu', type=int, default=0, choices=[i for i in range(8)])
parser.add_argument('--config', type=str, default='config.yaml')

args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
config['gpu'] = args.gpu

if args.task == 'preprocess':
    from src.preprocess import preprocess
    preprocess(config)
elif args.task == 'train':
    from src.train import train_language_model
    train_language_model(config)
elif args.task == 'test':
    from src.test import test_language_model
    test_language_model(config)
elif args.task == 'predict':
    from src.predict import predict
    predict(config)
else:
    raise ValueError('argument --task error')