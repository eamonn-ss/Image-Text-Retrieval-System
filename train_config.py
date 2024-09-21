import argparse

def parse_args():
    parser = argparse.ArgumentParser(description= 'command for train on Dataset')
    parser.add_argument('--name',default='Experiment',type=str,help='Output the model name')
    parser.add_argument('--dir',type=str,default='/datasets',help='directory to store dataset')
    parser.add_argument('--embedding_type',default='BERT',type=str,help='embedding type')
    parser.add_argument('--dataset',type=str,default="CUHKPEDES")
    parser.add_argument('--checkpoint_dir',type=str,default='./log',help='directory to store checkpoint')
    parser.add_argument('--log_dir',type=str,default='./log',help='directory to store log')
    parser.add_argument('--width',type=int,default=128)
    parser.add_argument('--height',type=int,default=384)
    parser.add_argument('--num_classes',type=int,default=11003)
    parser.add_argument('--feature_size',type=int,default=1024)
    parser.add_argument('--pretrained',action='store_false',help='wheather or not restore the pretrained visual model')
    parser.add_argument('--max_length',default=128,type=int,help='max_length')
    parser.add_argument('--droprate',default=0,type=float,help='drop rate')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--num_epoches',type=int,default=80)
    parser.add_argument('resume',action='store_true',help='whether or not to restore the pretrained whole model')
    parser.add_argument('--CMPM', default=True)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--wd', type=float, default=0.00004)
    parser.add_argument('--adam_lr', type=float, default=0.003, help='the learning rate of adam')
    parser.add_argument('--adam_alpha', type=float, default=0.9)
    parser.add_argument('--adam_beta', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--lr_decay_type', type=str, default='MultiStepLR',
                        help='One of "MultiStepLR" or "StepLR" or "ReduceLROnPlateau"')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--epoches_decay', type=str, default='50', help='#epoches when learning rate decays')
    parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='6')
    parser.add_argument('--use_pretrained', action='store_true', help='Whether or not to restore the pretrained model')
    parser.add_argument('--gamma',type=float,default=2.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--use_self_attention', type=bool, default=True, help='Whether to use bilinear pooling')
    args = parser.parse_args()
    return args