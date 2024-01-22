import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import FuyuProcessor, FuyuForCausalLM
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm.auto import tqdm
import argparse
from PIL import Image
import os
from peft import LoraConfig
import warnings
import numpy as np
import wandb
import copy
from collections import deque
from transformers import ViltProcessor, ViltForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import random
import torch
import heapq
from utils import TopAccuracyTextsNoDuplicates, extract_text_after_colon, evaluation, evaluation_full,got_example, create_balanced_subset
from utils import got_example_input, got_example,evaluation_roberta_soft,evaluation_roberta, TopAccuracyTextsScore
from dataset_utils import load_all_dataset
import sys
import datetime
import random
from dataset_utils import dataset_dicts
from utils import remove_text_after_key, evaluation_soft, evaluate_openai,reward_openai
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target_model_name',
        type= str,
        default= 'gpt2'
    )
    parser.add_argument(
        '--agent_model_name',
        type= str,
        default= 'gpt2'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default= 'sst2'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default= 4
    )
    parser.add_argument(
        '--queue_size',
        type=int,
        default= 5
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default= 1e-6
    )
    parser.add_argument(
        '--using_lora',
        type=bool,
        default= True
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default= 1000
    )
    parser.add_argument(
        '--start_prompt',
        type=str,
        default= 'To determine whether she is Harin or not, note the following instruction : '
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default= 'Default'
    )
    parser.add_argument(
        '--train_on_gpu',
        type=bool,
        default= True
    )
    parser.add_argument(
        '--language_feedback',
        type=bool,
        default= True
    )
    parser.add_argument(
        '--topk',
        type=int,
        default= 5
    )
    parser.add_argument(
        '--init_kl_coef',
        type=float,
        default= 0.4
    )
    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = ['negative','positive']
    )
    parser.add_argument(
        '--dataset_size',
        type = int,
        default = -1
    )
    parser.add_argument(
        '--test_batch_size',
        type = int,
        default = 1
    )
    parser.add_argument(
        '--meta_question',
        type = str,
        default = "I gave a friend an instruction and ten inputs. The friend read the instruction and wrote an output for every one of the inputs. Plase write instruction to help my friends. Here are the input-output pairs:",
    )
    parser.add_argument(
        '--test_term',
        type = int,
        default = 20,
    )
    parser.add_argument(
        '--example',
        type = int,
        default = 5,
    )
    parser.add_argument(
        '--max_length',
        type = int,
        default = 100,
    )
    parser.add_argument(
        '--softmax_reward',
        type = bool,
        default = False,
    )
    parser.add_argument(
        '--use_fewshot',
        type = bool,
        default = False,
    )
    parser.add_argument(
        '--debug_mode',
        type = bool,
        default = False,
    )
    parser.add_argument(
        '--cache_dir',
        type= str,
        default= '/mnt/sdb/mc/'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type= int,
        default= 1
    )
    parser.add_argument(
        '--use_score_scaling',
        type= bool,
        default= False
    )
    parser.add_argument(
        '--use_score_norm',
        type= bool,
        default= False
    )
    parser.add_argument(
        '--side',
        type= str,
        default= 'Last'
    )
    parser.add_argument(
        '--cliprange',
        type= float,
        default= 0.2
    )
    parser.add_argument(
        '--ratio_threshold',
        type= float,
        default= 10.0
    )
    parser.add_argument(
        '--project',
        type= str,
        default= 'spo_v1'
    )
    parser.add_argument(
        '--train_batch_size',
        type= int,
        default= 1,
    )
    parser.add_argument(
        '--cut_enter',
        type= bool,
        default= False,
    )
    parser.add_argument(
        '--acc_reward',
        type= bool,
        default= False,
    )
    parser.add_argument(
        '--acc_cutoff',
        type= float,
        default= 0.1,
    )
    parser.add_argument(
        '--fewshot',
        type= bool,
        default= False,
    )
    parser.add_argument(
        '--cold_test',
        type= bool,
        default= False,
    )
    parser.add_argument(
        '--dynamic_shot',
        type= bool,
        default= False,
    )
    args = parser.parse_args()
    return args




def main():
    
    #start and setting
    args = parse_args()
    wandb.init(project=args.project,name = args.run_name)
    
    #device setting
    device = 'cuda:0'
    
    #file setting
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = args.agent_model_name
    if '/' in agent_name:
        agent_name = agent_name.replace('/','_')
    target_name = args.target_model_name
    if '/' in target_name:
        target_name = target_name.replace('/','_')
    filename = f"data/{agent_name}_{target_name}_{current_time}_{args.dataset_name}.txt"
    original_stdout = sys.stdout
    
    
    #dataset load
    dataset = load_all_dataset(args.dataset_name)
    train_dataset = dataset[0]
    test_dataset = dataset[2]
    
    
    dataset_dict = dataset_dicts(args.dataset_name)
    if args.target_model_name == 'roberta-large':
        verbalizer = ['\u0120'+dataset_dict[i] for i in range(len(dataset_dict))]
    else:
        verbalizer = [dataset_dict[i] for i in range(len(dataset_dict))]
    args.verbalizer = verbalizer
    print(verbalizer)
    num_labels = len(verbalizer)
    train_dataset = create_balanced_subset(train_dataset,16*num_labels)
    validation_dataset=  create_balanced_subset(train_dataset,16*num_labels)
    if len(test_dataset) > 5000:
        test_dataset = create_balanced_subset(test_dataset,1000)
    print('test_dataset_size :', len(test_dataset))
    
    
    
    print('agent : ',args.agent_model_name)
    print('dataset : ',args.dataset_name)
    print(args)
    
    #cold test
    if args.cold_test:
        prompts = [
            'Is this review positive?',
            'Is this review means good?'
        ]
        print('cold test!')
        accuracy = evaluate_openai(prompts,test_datasets)
        print('accuracy : ',accuracy)
        
    
    #ppo setting
    bs = args.batch_size
    config = PPOConfig(
        model_name = args.agent_model_name,
        learning_rate = args.learning_rate,
        batch_size = bs,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        use_score_scaling = args.use_score_scaling,
        use_score_norm = args.use_score_norm,
        log_with='wandb',
        cliprange = args.cliprange,
        ratio_threshold = args.ratio_threshold,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    
    #load agent model
    train_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        peft_config=lora_config,
        cache_dir = args.cache_dir)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        peft_config=lora_config,
        cache_dir = args.cache_dir)
    ppo_tokenizer = AutoTokenizer.from_pretrained(args.agent_model_name,cache_dir=args.cache_dir)
    ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
    ppo_trainer = PPOTrainer(config, train_model, ref_model, ppo_tokenizer)
    #criterion = torch.nn.CrossEntropyLoss()
    
    
    #generation_kwargs setting
    generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": ppo_tokenizer.eos_token_id,
    "max_new_tokens":args.max_length,
    "min_length": -1,
    }
    
    
    #queue setting
    queue = TopAccuracyTextsNoDuplicates(max_size=args.queue_size)
    real_queue = TopAccuracyTextsScore(max_size=args.queue_size)
    
    #input setting
    need_input = ['sst2','qnli','yelp_polarity','customer_review']


    #setting dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    #make fewshot example
    if args.dataset_name in need_input:
        examples = got_example_input(validation_dataset,dataset_dict,shot=args.example)
    else:
        examples = got_example(validation_dataset,dataset_dict,shot=args.example)
    example = ''
    for e in examples:
        example += e + '\n'
    
    #make query text (meta question + fewshot example)
    query_text = [
        {"role": "user", "content": args.meta_question + '\n' + example},
        {"role": "assistant", "content" : "Instruction : "}
    ]
    
    
    #start training
    for ep in tqdm(range(args.max_epochs)):
        mean = 0
        total = 0
        for batch in train_dataloader:
            
            #setting inputs
            if 'text' in batch.keys():
                inputs = batch['text']
            else:
                inputs = batch['sentence']
            labels = batch['label']
            
            #if dynamic_shot=True, change fewshot example
            if args.dynamic_shot:
                if args.dataset_name in need_input:
                    examples = got_example_input(validation_dataset,dataset_dict,shot=args.example)
                else:
                    examples = got_example(validation_dataset,dataset_dict,shot=args.example)
                example = ''
                for e in examples:
                    example += e + '\n'
                
                query_text = [
                    {"role": "user", "content": args.meta_question + '\n' + example},
                    {"role": "assistant", "content" : "Instruction : "}
                ]


            #Tokenize
            query_tensors = ppo_tokenizer.apply_chat_template(query_text, return_tensors='pt').view(-1).to(device)
            
            #generate (한번에 여러 개의 prompt를 생성)
            response_tensors = ppo_trainer.generate(query_tensors.view(-1),**generation_kwargs,return_prompt=False, num_return_sequences=bs)
            
            #decode
            used_prompt = [ppo_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
            
            #개행 문자 이후는 제거하는 옵션. 주로 꺼둠
            if args.cut_enter:
                used_prompt = [remove_text_after_key(r, '\n') for r in used_prompt]
                
            #input 길이가 너무 짧은게 계속 나오면 끝냄
            check = 0
            for prompt in used_prompt:
                if len(prompt) < 5:
                    check +=1
            if check ==4 :
                print('broken!')
                return
            
            #Evaluation
            with torch.no_grad():
                rewards, accuracys = reward_openai(
                    used_prompt,
                    inputs,
                    labels,
                    examples=examples
                )
                    
                #reward에 정확도도 더하는 옵션. 수렴이 빨라지지만 골로 갈 확률도 증가
                if args.acc_reward:
                    rewards = [rewards[i] + accuracys[i] for i in range(len(rewards))]
                else:
                    rewards = [rewards[i] for i in range(len(rewards))]
                    
                
                print('query : ',query_text)
                print('used prompt : \n')
                
                
                for i in range(len(used_prompt)):
                    print('reward : ', rewards[i].item(),'acc :', accuracys[i], ' prompt : ', used_prompt[i], '\n')
                        
                    #정확도가 acc_cutoff 이상이면 queue에 추가
                    if accuracys[i] >= args.acc_cutoff:
                        queue.add(rewards[i].item(),used_prompt[i],ep)
            batch['query'] = query_text
            batch['response'] = used_prompt
            
            #ppo training
            stats = ppo_trainer.step([query_tensors.view(-1) for i in range(bs)],[response for response in response_tensors],[torch.tensor(reward) for reward in rewards])
            ppo_trainer.log_stats(stats,batch,rewards)
            
            
            #rewards logging
            rewards = torch.stack(rewards)
            mean_reward = torch.mean(rewards)
            std_reward = torch.std(rewards)
            max_reward = torch.max(rewards)
            print('mean_reward:',mean_reward)
            print('max_reward:',max_reward)
            wandb.log({'mean_reward':mean_reward})
            wandb.log({'max_reward':max_reward})
            wandb.log({'mean_accuracy':np.mean(np.array(accuracys))})
            
        #Test
        if ep % args.test_term == 0:
            #queue에서 높은 reward를 받은 queue들을 가져옴
            a = queue.get_top_texts()
            print('start test! \n test size : ',len(a))   
            new_acc = evaluate_openai(
                [a[i][1] for i in range(len(a))],
                test_dataset,
                examples=examples,
            )
                
            #test 결과를 queue에 추가
            for i in range(len(a)):
                real_queue.add(new_acc[i],a[i][1],a[i][2],a[i][0])
                
            #queue logging
            print('real queue updated!\n')
            li = real_queue.get_top_texts()
            li_new = []
            for num in range(len(li)):
                l = li[num]
                print('acc : ',l[0],'\nmaded_epoch :', l[2], '\ntext : ',l[1],'\nscore :' ,l[3],'\n' )
                li_new.append(l[0])
            li_new_np = np.array(li_new)
            if len(li_new_np)==0:
                li_new_np = np.array([0])
            wandb.log({'real_mean' : np.mean(li_new_np), 'real_std' : np.std(li_new_np), 'real_max' : np.max(li_new_np)})
if __name__ == '__main__':
    main()