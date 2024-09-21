import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="command for test on the dataset")

    parser.add_argument('--dir', type=str,default=r'/datasets',help='directory to store dataset')
    parser.add_argument('--dataset',type=str,default='CUHKPEDES')

    parser.add_argument('--width',type=int,default=128)
    parser.add_argument('--height',type=int,default=384)

    parser.add_argument('--num_classes', type=int, default=11003)
    parser.add_argument('--pretrained', action='store_true',
                        help='whether or not to restore the pretrained visual model')
    parser.add_argument('--droprate', default=0, type=float, help='drop rate')


    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--embedding_type', type=str,
                        default='BERT')

    parser.add_argument('--model_path', type=str,
                        default=r"./log/Experiment",
                        help='directory to load checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
                        default="./log/Experiment",
                        help='directory to store checkpoint')
    parser.add_argument('--log_test_dir', type=str,
                        default="./log/Experiment",
                        help='directory to store test')


    parser.add_argument('--feature_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epoches', type=int, default=3)
    parser.add_argument('--gpus', type=str, default='0')

    args = parser.parse_args()
    return args


