import argparse
import os
import sys
import yaml
from lib_dataset import _single_datasets_,_multi_datasets_

FEWSHOT_FAIR_PROTECTED_KEYS = {
    'embedding_hidden',
    'fs_way',
    'fs_train_way',
    'fs_val_way',
    'fs_test_way',
    'fs_shot',
    'fs_query',
    'fs_train_episodes',
    'fs_val_episodes',
    'fs_test_episodes',
    'fs_class_split',
    'fs_metric',
    'fs_temperature',
    'lr',
    'wd',
    'dropout',
    'epochs',
    'patience',
    'fs_eval_interval',
    'fs_patience_episodes',
}

FEWSHOT_FAIR_DEFAULTS = {
    'embedding_hidden': 128,
    'lr': 1.0e-2,
    'wd': 0.0,
    'dropout': 0.2,
    'fs_metric': 'cosine',
    'fs_temperature': 10.0,
    'fs_train_episodes': 300,
    'fs_val_episodes': 100,
    'fs_test_episodes': 600,
    'num_seeds': 5,
}

def _collect_cli_keys(argv):
    cli_keys = set()
    for token in argv[1:]:
        if token.startswith('--'):
            key = token[2:].split('=', 1)[0].replace('-', '_')
            cli_keys.add(key)
    return cli_keys

def _apply_fewshot_fair_defaults(args):
    if args.task_type == 'fewshot_node_cls' and args.fs_fair_config:
        cli_keys = getattr(args, '_cli_keys', set())
        for key, value in FEWSHOT_FAIR_DEFAULTS.items():
            if key not in cli_keys:
                setattr(args, key, value)

def update_from_dict(obj, updates, protected_keys=None):
    protected_keys = protected_keys or set()
    cli_keys = getattr(obj, '_cli_keys', set())
    for key, value in updates.items():
        # set higher priority from command line as we explore some factors
        if key in protected_keys:
            continue
        if key in cli_keys:
            continue
        if key in ['init'] and getattr(obj, 'init', None) is not None:
            continue
        setattr(obj, key, value)

# recommend hyperparameters here
def method_config(args):

    _apply_fewshot_fair_defaults(args)
    if args.is_default:
        config_name = 'default'
    else:
        config_name = args.dname
    try:
        # conf_dt = json.load(open(f"{os.path.join('./', 'lib_configs', args.method.lower(), config_name)}.json")) 
        if args.task_type == 'fewshot_node_cls':
            task_prefix = 'node_yamls'
        else:
            task_prefix=args.task_type.split('_')[0]+'_yamls'
        yaml_path = os.path.join(os.path.dirname(__file__), 'lib_yamls', task_prefix, 'config_'+args.method.lower()+'.yaml')
        config_all = yaml.safe_load(open(yaml_path))
        if config_name in config_all:
            conf_dt = config_all[config_name]
        else:
            conf_dt = config_all['default']
        protected_keys = FEWSHOT_FAIR_PROTECTED_KEYS if args.task_type == 'fewshot_node_cls' and args.fs_fair_config else set()
        update_from_dict(args, conf_dt, protected_keys=protected_keys)
        args._method_config_path = yaml_path
        args._method_config_name = config_name if config_name in config_all else 'default'
        args._method_config_applied = True
        args._fs_fair_protected_keys = sorted(protected_keys)
    except:
        args._method_config_path = None
        args._method_config_name = None
        args._method_config_applied = False
        if args.method not in ['ZEN', 'RawFeatureProto']:
            print('No config file found or error in json format, please use method_config(args)')

    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_task_args(args):
    
    if args.task_type == 'node_cls':
        if args.dname not in _single_datasets_:
            raise ValueError('The dataset is not suitable for node classification')
        args.add_self_loop=True 
        if args.use_bench_prop:
            args.train_prop,args.valid_prop = 0.5,0.25
        args.early_stop = False
    elif args.task_type == 'fewshot_node_cls':
        if args.dname not in _single_datasets_:
            raise ValueError('The dataset is not suitable for few-shot node classification')
        args.add_self_loop=True
        args.embedding_mode = True
        args.early_stop = True
    elif args.task_type == 'hg_cls':
        if args.dname not in _multi_datasets_:
            raise ValueError('The datasets is not suitable for hypergraph classification')
        args.add_self_loop=False
        if args.use_bench_prop:
            args.train_prop,args.valid_prop = 0.8,0.1
        args.early_stop = True
        if args.method in ['EHNN','TMPHN']:
            raise ValueError(f'{args.method} is not supoorted for hypergraph classification task') 
    else:
        if args.dname not in _single_datasets_:
            raise ValueError('The dataset is not suitable for edge prediction')

        if args.method in ['HyperND']:
            args.add_self_loop=True 
        elif args.method in ['DPHGNN','LEGCN','PhenomNN','HJRL','TFHNN','HNHN','AllSetformer'] and args.dname in ['pokec']:
            args.add_self_loop=True
        elif args.method in ['TMPHN']:
            if args.dname in ['pokec']:
                args.add_self_loop=True
            else:
                args.add_self_loop=False
            args.device='cpu' 
        else:
            args.add_self_loop=False
        if args.use_bench_prop:
            args.train_prop,args.valid_prop = 0.6,0.2
        args.early_stop = True
    
    return args

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    '''
    Semi-supervised setting: Train/Valid/Test: 50/25/25
    
    '''
    parser.add_argument('--use_bench_prop', default=True)
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.2)

    parser.add_argument('--dname', default='cora',choices=['cora','citeseer','pubmed',
                                                            'coauthor_cora','coauthor_dblp',
                                                            '20newsW100', 'ModelNet40', 'zoo','NTU2012', 'Mushroom',
                                                            'yelp','walmart-trips-100','house-committees-100',
                                                            'actor','amazon','pokec','twitch',
                                                            'german','bail','credit',
                                                            'amazon_review','magpm','trivago','ogbn_mag',
                                                            "RHG_3", "RHG_10", "RHG_table", "RHG_pyramid",
                                                            "IMDB_dir_form", "IMDB_dir_genre",
                                                            "IMDB_wri_form", "IMDB_wri_genre",
                                                            "stream_player","twitter_friend"])
    
    parser.add_argument('--task_type',default='edge_pred',choices=['node_cls','edge_pred','hg_cls','fewshot_node_cls'])
    parser.add_argument('--is_default',default=False)
    parser.add_argument('--use_processed', default=True)
    parser.add_argument('--method', default='HGNN') 
    
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_seeds', type=int, default=2)
    parser.add_argument('--epochs', default=5, type=int) 
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float) # []
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--clip_grad',default=False,type=bool)
    parser.add_argument('--clip_thresh',default=5.0,type=float)
    parser.add_argument('--num_splits',type=int,default=10)
    parser.add_argument('--mem_verbose',default=True)
    parser.add_argument('--mem_display_step',default=100)
    parser.add_argument('--display_step', type=int, default=20)
    parser.add_argument('--eval_verbose',default=True)
    
    parser.add_argument('--embedding_mode',default=False,type=bool) 
    parser.add_argument('--embedding_hidden',default=128,type=int) 
    
    parser.add_argument('--normtype', default='all_one') # ['all_one','deg_half_sym']
    parser.add_argument('--aggregate', default=None, choices=['add','sum','mean'])
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--exclude_self', action='store_true')
    
    parser.add_argument('--edge_split_mode',default='ind',choices=['ind','trand'])
    parser.add_argument('--edge_pred_protocol',default='legacy',choices=['legacy','observed'])
    parser.add_argument('--edge_save_dir', action='store_true',default=f'./lib_edge_splits/') 
    parser.add_argument('--edge_batch_size', action='store_true',default=512) 
    parser.add_argument('--e_embed_hidden',default=64) 
    parser.add_argument('--e_embed_layer',default=2)
    parser.add_argument('--e_embed_dropout',default=0.2) 
    parser.add_argument('--e_embed_norm',default='ln') 
    parser.add_argument('--aggr_mode',default='max',choices=['max','mean','maxmin'])
    parser.add_argument('--ns_method',default='mixed',choices=['mns','sns','cns','mixed']) 
    parser.add_argument('--edge_aggr',default='group',choices=['group','single'])
    
    parser.add_argument('--hg_batch_size',default=256) # batch_size
    parser.add_argument('--pooling',default='mean')
    parser.add_argument('--g_embed_hidden',default=128) 
    parser.add_argument('--g_embed_layer',default=2) 
    parser.add_argument('--g_embed_dropout',default=0.2) 
    parser.add_argument('--g_embed_norm',default='ln') 
    parser.add_argument('--use_weighted_loss',default=False)
    parser.add_argument('--early_stop',default=True) 
    parser.add_argument('--patience',default=100,type=int)

    parser.add_argument('--fs_way',default=5,type=int)
    parser.add_argument('--fs_train_way',default=None,type=int)
    parser.add_argument('--fs_val_way',default=None,type=int)
    parser.add_argument('--fs_test_way',default=None,type=int)
    parser.add_argument('--fs_shot',default=1,type=int)
    parser.add_argument('--fs_query',default=15,type=int)
    parser.add_argument('--fs_train_episodes',default=1000,type=int)
    parser.add_argument('--fs_val_episodes',default=200,type=int)
    parser.add_argument('--fs_test_episodes',default=600,type=int)
    parser.add_argument('--fs_metric',default='cosine',choices=['cosine','euclidean'])
    parser.add_argument('--fs_temperature',default=10.0,type=float)
    parser.add_argument('--fs_class_split',default=None,type=str)
    parser.add_argument('--fs_val_class_ratio',default=0.2,type=float)
    parser.add_argument('--fs_test_class_ratio',default=0.2,type=float)
    parser.add_argument('--fs_fixed_class_order',default=False,nargs='?',const=True,type=str2bool)
    parser.add_argument('--fs_hypergcn_no_last_relu',default=False,nargs='?',const=True,type=str2bool)
    parser.add_argument('--fs_eval_interval',default=20,type=int)
    parser.add_argument('--fs_patience_episodes',default=100,type=int)
    parser.add_argument('--fs_fair_config',default=True,nargs='?',const=True,type=str2bool)
    parser.add_argument('--fs_save_episode_bank',default=True,nargs='?',const=True,type=str2bool)
    parser.add_argument('--fs_reuse_episode_bank',default=True,nargs='?',const=True,type=str2bool)
    parser.add_argument('--fs_episode_bank_dir',default='result/fewshot_episode_banks',type=str)
    parser.add_argument('--fs_log_dir',default='logs',type=str)

    parser.add_argument('--zen_hyperparams',default='0.3333333333,0.3333333333,0.3333333334',type=str)
    parser.add_argument('--zen_projection_seed',default=0,type=int)
    parser.add_argument('--zen_mode',default='no_projection',
                        choices=['no_projection','random_projection','trainable_adapter','raw_feature_proto'])

    parser.add_argument('--is_perturbed',default=False) 
    parser.add_argument('--is_poison',default=True) 
    parser.add_argument('--pert_mode',default='spar_label',choices=['spar_feat','noise_feat',
                                                                    'drop_incidence','add_incidence',
                                                                    'spar_label','flip_label'])
    parser.add_argument('--pert_p',default=0.0) 

    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='0.6', type=str)
    
    parser.set_defaults(add_self_loop=False)
    parser.set_defaults(exclude_self=False)

    args = parser.parse_args()
    args._cli_keys = _collect_cli_keys(sys.argv)

    return args
