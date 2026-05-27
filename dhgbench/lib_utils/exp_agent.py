import csv
import json
import os
import copy
import numpy as np
import time
#from parameter_parser import parameter_parser
import torch
from collections import defaultdict
from lib_utils.utils import fix_seed,result_printer,mean_std_metrics
from lib_utils.train_agent import Trainer
from lib_utils.eval_agent import Evaluator
from lib_models.HNN import HCHA,HyperGCN,HNHN,SetGNN,UniGNN,UniGCNII,LEGCN,HyperND,EquivSetGNN,\
                            PlainUnigencoder,HJRL,SheafHyperGNN,EHNN,TMPHN,PhenomNN,PhenomNNS,DPHGNN,TFHNN,PlainMLP,HyperGT,CEGCN,CEGAT,ZENEncoder,RawFeatureProto

from lib_dataset.data_perturbation import perturbation
from lib_dataset.fewshot import build_all_episode_banks,build_fewshot_splits
from lib_dataset.edge_loaders import generate_edge_loaders,generate_split_hyperedges,generate_ind_split_hyperedges,\
                                    generate_observed_split_hyperedges,generate_observed_ind_split_hyperedges,\
                                    build_observed_support_data
from lib_dataset.hg_loaders import generate_split_hypergraphs,generate_hg_loaders
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.aggregator import EdgePredictor,MeanAggregator,MaxminAggregator,MaxAggregator,HyperGPredictor
from lib_utils.fewshot_train_agent import evaluate_fewshot_node_cls,infer_fewshot_embedding_dim,train_fewshot_node_cls
from lib_utils.metrics import aggr_metrics,avg_result_printer_edge

class ExpAgent:
    
    def __init__(self,args,**kwargs):
        """
        Overall pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
        self.trainer=Trainer(args)
        self.evaluator=Evaluator(args)
        self.train_times = []

    def edge_pred_train_eval(self,data):
        if self.args.edge_pred_protocol == 'observed' and self.args.edge_split_mode == 'ind':
            raise ValueError(
                "The observed edge prediction protocol currently supports only "
                "--edge_split_mode=trand. The observed ind split is disabled because "
                "its support graph contains the train positives."
            )
        
        metrics_dict = {'train':defaultdict(list),'val':defaultdict(list),'test':defaultdict(list)}
        
        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            if self.args.edge_pred_protocol == 'observed':
                dir_path = f"{self.args.edge_save_dir}{self.args.edge_split_mode}_{self.args.edge_pred_protocol}/{self.args.dname}/"
            else:
                dir_path = f"{self.args.edge_save_dir}{self.args.edge_split_mode}/{self.args.dname}/"
            
            if self.args.edge_split_mode == 'ind':

                file_path = dir_path+f"split_{seed}.pt"
                if not os.path.exists(file_path):
                    os.makedirs(dir_path, exist_ok=True)
                    if self.args.edge_pred_protocol == 'observed':
                        generate_observed_ind_split_hyperedges(data,self.args,seed)
                    else:
                        generate_ind_split_hyperedges(data,self.args,seed)

            elif self.args.edge_split_mode == 'trand':
                
                file_path = dir_path+f"split_{seed}.pt"
                if not os.path.exists(file_path):
                    os.makedirs(dir_path, exist_ok=True)
                    if self.args.edge_pred_protocol == 'observed':
                        generate_observed_split_hyperedges(data,self.args,seed)
                    else:
                        generate_split_hyperedges(data,self.args,seed)

            else:

                raise NotImplementedError
                
            data_dict = torch.load(file_path, weights_only=False)
            if self.args.edge_pred_protocol == 'observed':
                data_for_edge_pred = build_observed_support_data(data,data_dict,self.args)
                data_for_edge_pred = algo_preprocessing(data_for_edge_pred,self.args)
            else:
                data_for_edge_pred = data
            batch_loaders = generate_edge_loaders(data_dict,self.args)
            
            self.args.embedding_mode = True 
            encoder = parse_model(self.args,data_for_edge_pred)
            
            if self.args.aggr_mode=='maxmin':
                aggregator = MaxminAggregator(self.args) 
            elif self.args.aggr_mode=='mean':
                aggregator = MeanAggregator(self.args) 
            elif self.args.aggr_mode=='max':
                aggregator = MaxAggregator(self.args) 
            
            model = EdgePredictor(encoder,aggregator,self.args)
            if self.args.method == 'TMPHN':
                model.aggregator = model.aggregator.to(self.args.device)
            else:
                model = model.to(self.args.device)
            
            model = self.trainer.training(model,data_for_edge_pred,self.args,seed_split=batch_loaders,task_type='edge_pred')

            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,data_for_edge_pred,seed_split=batch_loaders,task_type='edge_pred',verbose=True)
                metrics_dict = aggr_metrics(metrics_dict,result) 
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,data_for_edge_pred,seed_split=batch_loaders,task_type='edge_pred',verbose=True)
                metrics_dict = aggr_metrics(metrics_dict,result) 

        print(f'---------------------------------[Final]--------------------------------------')
        avg_result_printer_edge(metrics_dict)
        print(f'------------------------------------------------------------------------------')

    def node_cls_train_eval(self,data):
        
        metrics_dict=defaultdict(list)

        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            masks=data.generate_random_split(train_ratio=self.args.train_prop,val_ratio=self.args.valid_prop,seed=seed)

            if self.args.is_perturbed:
                if self.args.pert_mode in ['spar_label','flip_label']:
                    if self.args.pert_mode == 'spar_label':
                        masks = perturbation(data,mode=self.args.pert_mode,p=self.args.pert_p,masks=masks)
                    elif self.args.pert_mode == 'flip_label':
                        data = perturbation(data,mode=self.args.pert_mode,p=self.args.pert_p,masks=masks)
                    else:
                        raise ValueError('Unimplemented perturbation mode for label robustness')

            model = parse_model(self.args,data)
            if self.args.method == 'TMPHN':
                pass
            else:
                model = model.to(self.args.device)

            self.trainer.training(model,data,self.args,seed_split=masks,task_type='node_cls')
            
            self.train_times.append(self.trainer.train_time)

            # Evasion Attack
            if self.args.is_perturbed and not self.args.is_poison:
                test_data = data.evasion_data
            else:
                test_data = data

            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,test_data,seed_split=masks,task_type='node_cls',verbose=True)
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,test_data,seed_split=masks,task_type='node_cls',verbose=False)
            
            for m in result:
                metrics_dict[m].append(result[m])
            
        print(f'---------------------------------[Final]--------------------------------------')
        self.test_dict = defaultdict(list) 
        for m in metrics_dict:
            result_printer(metrics_dict[m],m)
            metrics_mean, metrics_std = mean_std_metrics(metrics_dict[m])
            self.test_dict[m].extend([metrics_mean[-1],metrics_std[-1]])
        print(f'Avg Training Time: {np.mean(self.train_times):2f}')
        print(f'------------------------------------------------------------------------------')

    def hg_cls_train_eval(self,data):
        
        metrics_dict=defaultdict(list)
        
        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            train_set,val_set,test_set = generate_split_hypergraphs(data,self.args.train_prop,self.args.valid_prop,seed)
            batch_loaders = generate_hg_loaders(train_set,val_set,test_set,self.args)
            
            self.args.embedding_mode = True 
            encoder = parse_model(self.args,data)

            model = HyperGPredictor(encoder,data.num_classes,self.args)
            if self.args.method == 'TMPHN':
                model.classifer = model.aggregator.to(self.args.device)
            else:
                model = model.to(self.args.device)
                     
            model = self.trainer.training(model,data,self.args,seed_split=batch_loaders,task_type='hg_cls')
            
            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,data,seed_split=batch_loaders,task_type='hg_cls',verbose=True)
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,data,seed_split=batch_loaders,task_type='hg_cls',verbose=False)

            for m in result:
                metrics_dict[m].append(result[m])

        print(f'---------------------------------[Final]--------------------------------------')
        for m in metrics_dict:
            result_printer(metrics_dict[m],m)
        print(f'------------------------------------------------------------------------------')

    def fewshot_node_cls_train_eval(self,data):

        seed_records = []
        train_way = self.args.fs_train_way or self.args.fs_way
        val_way = self.args.fs_val_way or self.args.fs_way
        test_way = self.args.fs_test_way or self.args.fs_way
        self._print_fewshot_fair_config(train_way, val_way, test_way)

        for seed in range(self.args.num_seeds):

            fix_seed(seed)
            self.args.embedding_mode = True
            split_dict = build_fewshot_splits(self.args, data, seed)
            episode_banks = build_all_episode_banks(self.args, data, split_dict, seed)

            print(f'--------------------------[Few-shot Seed {seed}]-----------------------------')
            print(f'original classes: {split_dict["original_classes"]}')
            print(f'train_classes: {split_dict["train_classes"]}')
            print(f'val_classes: {split_dict["val_classes"]}')
            print(f'test_classes: {split_dict["test_classes"]}')
            print(f'unused_classes: {split_dict["unused_classes"]}')
            print(f'dropped_classes: {split_dict["dropped_classes"]}')
            print(
                'disjoint check: '
                f'train&val={sorted(set(split_dict["train_classes"]) & set(split_dict["val_classes"]))}, '
                f'train&test={sorted(set(split_dict["train_classes"]) & set(split_dict["test_classes"]))}, '
                f'val&test={sorted(set(split_dict["val_classes"]) & set(split_dict["test_classes"]))}'
            )
            print(f'fs_class_split: {self.args.fs_class_split}')
            print(f'fs_way: {self.args.fs_way}')
            print(
                f'split counts: train={len(split_dict["train_classes"])}, val={len(split_dict["val_classes"])}, '
                f'test={len(split_dict["test_classes"])}, unused={len(split_dict["unused_classes"])}'
            )
            print(f'train_way / val_way / test_way: {train_way} / {val_way} / {test_way}')
            print(f'shot / query: {self.args.fs_shot} / {self.args.fs_query}')
            print(f'embedding_hidden: {self.args.embedding_hidden}')
            print(f'split hash: {split_dict["split_hash"]}')
            for split_name in ['train', 'val', 'test']:
                print(f'{split_name} episode bank path: {episode_banks["meta"]["paths"][split_name]}')
                print(f'{split_name} episode bank hash: {episode_banks["meta"]["episode_hashes"][split_name]}')
            print(f'episode bank reused: {episode_banks["meta"]["reused"]}')

            model = parse_model(self.args,data)
            if self.args.method == 'TMPHN':
                pass
            else:
                model = model.to(self.args.device)
            self._print_model_audit(model)

            model, train_info = train_fewshot_node_cls(model,data,episode_banks,self.args)
            emb_dim = infer_fewshot_embedding_dim(model, data, self.args)
            train_acc, train_std, train_ci95 = evaluate_fewshot_node_cls(
                model,
                data,
                episode_banks["train"],
                self.args,
            )
            final_val_acc, final_val_std, final_val_ci95 = evaluate_fewshot_node_cls(
                model,
                data,
                episode_banks["val"],
                self.args,
            )

            test_acc, test_std, test_ci95 = evaluate_fewshot_node_cls(
                model,
                data,
                episode_banks["test"],
                self.args,
            )
            notes = self._fewshot_notes(model, train_info)
            seed_record = {
                'model': self.args.method,
                'seed': seed,
                'train_classes': split_dict['train_classes'],
                'val_classes': split_dict['val_classes'],
                'test_classes': split_dict['test_classes'],
                'unused_classes': split_dict['unused_classes'],
                'dropped_classes': split_dict['dropped_classes'],
                'split_hash': split_dict['split_hash'],
                'train_episode_hash': episode_banks['meta']['episode_hashes']['train'],
                'val_episode_hash': episode_banks['meta']['episode_hashes']['val'],
                'test_episode_hash': episode_banks['meta']['episode_hashes']['test'],
                'train_bank_path': episode_banks['meta']['paths']['train'],
                'val_bank_path': episode_banks['meta']['paths']['val'],
                'test_bank_path': episode_banks['meta']['paths']['test'],
                'train_way': train_way,
                'val_way': val_way,
                'test_way': test_way,
                'shot': self.args.fs_shot,
                'query': self.args.fs_query,
                'embedding_dim': emb_dim,
                'best_val_acc': train_info['best_val_acc'],
                'final_val_acc': final_val_acc,
                'test_acc_mean': test_acc,
                'test_acc_std': test_std,
                'test_ci95': test_ci95,
                'train_acc': train_acc,
                'train_acc_std': train_std,
                'train_ci95': train_ci95,
                'trainable_param_count': train_info['trainable_param_count'],
                'total_param_count': train_info['total_param_count'],
                'training_time': train_info['training_time'],
                'notes': notes,
            }
            seed_records.append(seed_record)
            print(
                f'[Seed {seed}] final best val acc: {train_info["best_val_acc"]:.4f}; '
                f'test episodic accuracy: '
                f'{test_acc:.4f} +- {test_std:.4f} (95% CI {test_ci95:.4f})'
            )
            print(
                f'params: total={train_info["total_param_count"]}, '
                f'trainable={train_info["trainable_param_count"]}, embedding_dim={emb_dim}'
            )
            print(f'notes: {notes}')
            print(f'------------------------------------------------------------------------------')

        summary = self._summarize_fewshot_records(seed_records)
        print(f'---------------------------------[Final]--------------------------------------')
        self._print_fewshot_summary_table([summary])
        self._write_fewshot_result_files(seed_records, summary)
        print(f'------------------------------------------------------------------------------')

    def _print_fewshot_fair_config(self, train_way, val_way, test_way):
        print('-----------------------[Few-shot Fair Config]-------------------------------')
        print(f'fs_fair_config={self.args.fs_fair_config}')
        print(f'final lr={self.args.lr}')
        print(f'final wd={self.args.wd}')
        print(f'final dropout={self.args.dropout}')
        print(f'final embedding_hidden={self.args.embedding_hidden}')
        print(f'final train_way / val_way / test_way={train_way} / {val_way} / {test_way}')
        print(f'fs_train_episodes / fs_val_episodes / fs_test_episodes={self.args.fs_train_episodes} / {self.args.fs_val_episodes} / {self.args.fs_test_episodes}')
        print(f'fs_eval_interval={self.args.fs_eval_interval}, fs_patience_episodes={self.args.fs_patience_episodes}')
        print(f'fs_metric={self.args.fs_metric}, fs_temperature={self.args.fs_temperature}')
        print(f'method-specific architecture config applied={getattr(self.args, "_method_config_applied", False)}')
        if getattr(self.args, "_method_config_applied", False):
            print(f'method config path={self.args._method_config_path}')
            print(f'method config section={self.args._method_config_name}')
        print(f'------------------------------------------------------------------------------')

    def _fewshot_notes(self, model, train_info):
        notes = [train_info['notes']]
        if self.args.method == 'ZEN':
            notes.append(f'ZEN {self.args.zen_mode}')
            notes.append(f'projection_trainable={any(param.requires_grad for param in model.parameters())}')
        elif self.args.method == 'RawFeatureProto':
            notes.append('raw feature prototype')
        elif self.args.method == 'AllDeepSets':
            notes.append(f'AllDeepSets PMA=False aggregate={self.args.aggregate}')
        elif self.args.method == 'HyperGCN':
            notes.append(f'HyperGCN last_relu={not self.args.fs_hypergcn_no_last_relu}')
        return '; '.join(notes)

    def _print_model_audit(self, model):
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f'model audit: total_params={total_params}, trainable_params={trainable_params}')
        if self.args.method == 'ZEN':
            projection_used = getattr(model, 'projection', torch.empty(0)).numel() > 0
            adapter = getattr(model, 'adapter', None)
            print(f'zen_mode={self.args.zen_mode}')
            print(f'ZEN has trainable parameters={trainable_params > 0}')
            print(f'ZEN projection used={projection_used}')
            print(f'ZEN projection trainable={adapter is not None and any(param.requires_grad for param in adapter.parameters())}')
        elif self.args.method == 'RawFeatureProto':
            print('RawFeatureProto baseline: model(data) returns data.x and ProtoHead does episode-local classification.')

    def _summarize_fewshot_records(self, records):
        test_values = np.array([record['test_acc_mean'] for record in records], dtype=np.float64)
        val_values = np.array([record['final_val_acc'] for record in records], dtype=np.float64)
        train_values = np.array([record['train_acc'] for record in records], dtype=np.float64)
        test_mean = float(test_values.mean())
        test_std = float(test_values.std())
        return {
            'model': self.args.method,
            'params': records[0]['total_param_count'],
            'trainable': records[0]['trainable_param_count'],
            'emb_dim': records[0]['embedding_dim'],
            'train_acc': float(train_values.mean()),
            'val_acc': float(val_values.mean()),
            'test_acc_mean': test_mean,
            'test_acc_std': test_std,
            'ci95': float(1.96 * test_std / np.sqrt(len(test_values))),
            'notes': records[0]['notes'],
            'num_seeds': len(records),
            'training_time': float(np.sum([record['training_time'] for record in records])),
        }

    def _result_dir(self):
        return os.path.join(self.args.fs_log_dir, self.args.dname, 'bench')

    def _write_fewshot_result_files(self, seed_records, summary):
        result_dir = self._result_dir()
        os.makedirs(result_dir, exist_ok=True)
        log_path = os.path.join(result_dir, 'result.log')
        csv_path = os.path.join(result_dir, 'result.csv')
        json_path = os.path.join(result_dir, 'result.json')

        existing_payload = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    existing_payload = json.load(f)
            except json.JSONDecodeError:
                existing_payload = []
        payload_entry = {
            'summary': summary,
            'seeds': seed_records,
            'created_at_unix': time.time(),
        }
        existing_payload.append(payload_entry)
        with open(json_path, 'w') as f:
            json.dump(existing_payload, f, indent=2)

        fieldnames = ['model','params','trainable','emb_dim','train_acc','val_acc','test_acc_mean','test_acc_std','ci95','notes','num_seeds','training_time']
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: summary[key] for key in fieldnames})

        summaries = [entry['summary'] for entry in existing_payload]
        with open(log_path, 'a') as f:
            f.write('\n[DHG-Bench fewshot_node_cls result]\n')
            f.write(json.dumps(summary, indent=2) + '\n')
            f.write(self._fewshot_summary_table_text(summaries) + '\n')
        print(f'few-shot result log saved to: {log_path}')
        print(f'few-shot result csv saved to: {csv_path}')
        print(f'few-shot result json saved to: {json_path}')

    def _print_fewshot_summary_table(self, summaries):
        print(self._fewshot_summary_table_text(summaries))

    def _fewshot_summary_table_text(self, summaries):
        headers = ['model','params','trainable','emb_dim','train_acc','val_acc','test_acc_mean','test_acc_std','ci95','notes']
        lines = [' | '.join(headers)]
        for row in summaries:
            lines.append(
                f"{row['model']} | {row['params']} | {row['trainable']} | {row['emb_dim']} | "
                f"{row['train_acc']:.4f} | {row['val_acc']:.4f} | {row['test_acc_mean']:.4f} | "
                f"{row['test_acc_std']:.4f} | {row['ci95']:.4f} | {row['notes']}"
            )
        return '\n'.join(lines)
        
    def running(self,task_type,data):
        
        if task_type == 'node_cls':
            self.node_cls_train_eval(data)
        elif task_type == 'edge_pred':
            self.edge_pred_train_eval(data)
        elif task_type == 'hg_cls':
            self.hg_cls_train_eval(data)
        elif task_type == 'fewshot_node_cls':
            self.fewshot_node_cls_train_eval(data)
        else:
            raise NotImplementedError

def parse_model(args, data):
    
    if args.embedding_mode:
        num_targets=args.embedding_hidden
    else:
        num_targets=data.num_classes
    
    # --------- Hypergraph Semi-supervised Models --------------------
    
    if args.method == 'AllSetformer':
        if args.LearnMask:
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method == 'AllDeepSets':
        local_args = copy.copy(args)
        local_args.PMA = False
        if getattr(local_args, 'aggregate', None) is None:
            local_args.aggregate = 'mean'
            args.aggregate = 'mean'
        if local_args.LearnMask:
            model = SetGNN(data.num_features, num_targets, local_args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, local_args)
        for layer in model.V2EConvs:
            assert layer.attention is False
        for layer in model.E2VConvs:
            assert layer.attention is False
        if getattr(args, 'task_type', None) == 'fewshot_node_cls':
            print('AllDeepSets check:')
            print('  PMA=False')
            print('  attention=False')
            print(f'  aggregate={local_args.aggregate}')
            print(f'  MLP_hidden={local_args.MLP_hidden}')
            print(f'  decoder_hidden={local_args.decoder_hidden}')
    elif args.method in ['HGNN','HCHA']:
        model = HCHA(data.num_features, num_targets, args)
    elif args.method == 'HNHN':
        model = HNHN(data.num_features, num_targets, args)
    elif args.method in ['UniGIN']:
        model = UniGNN(data.num_features, num_targets, args)
    elif args.method == 'UniGCNII':
        model = UniGCNII(data.num_features, num_targets, args)
    elif args.method == 'HyperGCN':
        model = HyperGCN(data.num_features, num_targets, args)
    elif args.method == 'LEGCN':
        model = LEGCN(data.num_features, num_targets, args)
    elif args.method == 'HJRL':
        model = HJRL(data.num_features, num_targets, args)
    elif args.method == 'HyperND':
        model = HyperND(data.num_features, num_targets, args)
    elif args.method == 'EDHNN':
        model = EquivSetGNN(data.num_features, num_targets, args)
    elif args.method == 'SheafHyperGNN':
        model = SheafHyperGNN(data.num_features,num_targets,args)
    elif args.method == 'EHNN':
        model = EHNN(data.num_features,num_targets,args,data.ehnn_cache)
    elif args.method == 'TMPHN':
        model = TMPHN(data.num_features,num_targets,data.x,data.neig_dict,args)
    elif args.method == 'PhenomNNS':
        model = PhenomNNS(data.num_features,num_targets,args)
    elif args.method == 'PhenomNN':
        model = PhenomNN(data.num_features,num_targets,args)
    elif args.method == 'DPHGNN':
        model = DPHGNN(data.num_features,num_targets,args)
    elif args.method == 'PlainUnigencoder':
        model = PlainUnigencoder(data.num_features, num_targets, args)
    elif args.method == 'TFHNN':
        model = TFHNN(data.num_features,num_targets,args)
    elif args.method == 'MLP':
        model = PlainMLP(data.num_features,num_targets,args)
    elif args.method == 'HyperGT':
        model = HyperGT(data.num_features,num_targets,args)
    elif args.method == 'CEGCN':
        model = CEGCN(data.num_features,num_targets,args)
    elif args.method == 'CEGAT':
        model = CEGAT(data.num_features,num_targets,args)
    elif args.method == 'ZEN':
        model = ZENEncoder(data.num_features,num_targets,args)
    elif args.method == 'RawFeatureProto':
        model = RawFeatureProto(data.num_features,num_targets,args)
    else:
        raise ValueError('Unimplemented model')

    return model
