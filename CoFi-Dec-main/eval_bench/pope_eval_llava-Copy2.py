#成功的

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
# from ZoomEye.ZoomEye.eval.demo import run_demo

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
from PIL import Image
from torchvision.transforms import v2

from pope_loader import POPEDataSet

# import kornia
from degf_utils.degf_sample import evolve_degf_sampling
from degf_utils.vcd_add_noise import add_diffusion_noise
from degf_utils.image_variation import get_image_variation_pipeline, apply_image_variation
from degf_utils.image_similarity import get_clip_similarity
from degf_utils.image_generation import get_image_generation_pipeline, generate_image_stable_diffusion
# from huggingface_hub import login

# login()
evolve_degf_sampling()
torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    
    parser.add_argument("--neg_image", type=str, default=None, 
                        help="Path to the negative image generated for diffusion method")
    
    
    parser.add_argument("--model_path", type=str, default="/mnt/server8_hard1/donguk/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--data_path", type=str, default="/mnt/server18_hard0/jhjang/LVLM/crg/data/coco/val2014")
    parser.add_argument("--pope_path", type=str, default="/mnt/server8_hard1/donguk/rips2024/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/pope")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_diffusion", type=str2bool, default=False)
    
    parser.add_argument("--degf_alpha_pos", type=float, default=3)
    parser.add_argument("--degf_alpha_neg", type=float, default=1)
    parser.add_argument("--degf_beta", type=float, default=0.1)
    
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--experiment_index", type=int, default=0)



    parser.add_argument("--data_ratio", type=float, default=1.0, help="Ratio of data to use (0-1)")
    
    
    parser.add_argument("--output_image_dir", type=str, default="/root/autodl-tmp/DeGF/outputs", help="Base directory for saving output images")


    args = parser.parse_args()
    return args


def print_acc(pred_list, label_list, logger):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    logger.info('TP\tFP\tTN\tFN\t')
    logger.info('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP) / float(TP + FN)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, precision, recall, f1, yes_ratio


def print_current_round_stats(pred, label, round_num, total_rounds, logger):
    """
    打印单轮统计结果
    """
    # 计算单轮指标
    acc, precision, recall, f1, yes_ratio = print_acc([pred], [label], logger)
    acc = round(acc*100,2)
    precision = round(precision*100,2)
    recall = round(recall*100,2)
    f1 = round(f1*100,2)
    yes_ratio = round(yes_ratio*100,2)
    
    logger.info(f"\n=== Round {round_num}/{total_rounds} Statistics ===")
    logger.info(f"Single round - acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}")



def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
        break
    
    return pred_list





def main():
    args = parse_args()
    
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()
    
    # Setup an experiment folder:
    if dist.get_rank() == 0:
        # 创建基础输出目录
        os.makedirs(args.output_image_dir, exist_ok=True)

        # 根据实验参数创建子文件夹
        model_string_name = args.model_path.split("/")[-1]
        experiment_subdir = f"{model_string_name}_ritual{args.use_ritual}_vcd{args.use_vcd}_m3id{args.use_m3id}_diffusion{args.use_diffusion}_exp{args.experiment_index}"
        experiment_output_dir = os.path.join(args.output_image_dir, experiment_subdir)
        os.makedirs(experiment_output_dir, exist_ok=True)

        # 创建不同类型图像的子文件夹
        image_output_dirs = {
            'raw_images': os.path.join(experiment_output_dir, 'raw_images'),
            'neg_images': os.path.join(experiment_output_dir, 'neg_images'),
            'zoom_eye_outputs': os.path.join(experiment_output_dir, 'zoom_eye_outputs')
        }
        for dir_path in image_output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        logger = create_logger(experiment_output_dir)
        logger.info(f"Experiment directory created at {experiment_output_dir}")
    else:
        logger = create_logger(None)

    
        
        
        
        
       

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing Model')

    #### for ritual
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)



    pope_dataset = POPEDataSet(
    pope_path=args.pope_path, 
    data_path=args.data_path,
    trans=image_processor,
    model=args.model_base
    )

    # 添加采样代码
    total_len = len(pope_dataset)
    subset_size = int(total_len * args.data_ratio)
    indices = torch.randperm(total_len)[:subset_size]
    subset_dataset = torch.utils.data.Subset(pope_dataset, indices)

    pope_loader = torch.utils.data.DataLoader(
        subset_dataset,  # 这里改为 subset_dataset
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )


    # ==============================================
    #               Augmentations
    # ==============================================

    aug_dict = {
    'horizontal flip':v2.RandomHorizontalFlip(p=1),
    'vertical flip':v2.RandomVerticalFlip(p=1),
    'rotation':v2.RandomRotation(degrees=180),
    'color jitter':v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
    'gaussian blur':v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
    'crop':v2.RandomResizedCrop(size=336),
    }
    
    # For statistics
    pos_aug_counter = {k:0 for k in aug_dict}
    pos_aug_counter.update({None: 0})

    # ========================================
    #            Start Generation
    # ========================================
    logger.info("Start eval...")
    pred_list, label_list = [], []
    pred_list2 = []
    js_list_first = []
    if args.use_diffusion:
        # sd_pipe = get_image_variation_pipeline()
        pipe = get_image_generation_pipeline()



    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"][0]
        qs = data["query"][0]
        label = data["label"]
        image_path = data["image_path"]
        label_list = label_list + list(label)

        # 在这里插入新的 ZoomEye 执行代码
        
        output_dir = os.path.join("/root/autodl-tmp/DeGF/outputs", 
                          f"llava-v1.5-7b_ritual{args.use_ritual}_vcd{args.use_vcd}_m3id{args.use_m3id}_diffusion{args.use_diffusion}_exp{args.experiment_index}", "zoom_eye_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_image = f"zoomeye_output_{batch_id}.jpg"
        output_path = os.path.join(output_dir, output_image)

        try:
            escaped_question = qs.replace('"', '\\"')

            # 打印完整的命令，便于调试
            print("ZoomEye Command:")
            zoom_eye_cmd = [
                "conda", "run", "-n", "zoom_eye",
                "python", "-c",
                f'''import sys
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 确保输出目录存在
output_dir = "{output_dir}"
os.makedirs(output_dir, exist_ok=True)

sys.path.append("{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")
from ZoomEye.ZoomEye.eval.demo import run_demo

# 打印详细的输入信息
print("Input Image Path:", "{image_path[0]}")
print("Output Image Path:", "{output_path}")
print("Question:", "{escaped_question}")

result = run_demo(
    model_path="{args.model_path}",
    input_image="{image_path[0]}",
    question="{escaped_question}",
    output_image_path="{output_path}"
)

# 记录结果到文件
with open("{output_dir}/zoomeye_result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))

# 检查输出文件是否存在
import os
print("Output image exists:", os.path.exists("{output_path}"))
'''
            ]

            print("Full ZoomEye Command:", " ".join(zoom_eye_cmd))
    
            # 详细的进程执行和错误捕获
            try:
                result = subprocess.run(
                    zoom_eye_cmd,
                    capture_output=True,  
                    text=True,
                    check=True,  
                    timeout=120  
                )

                # 打印完整的输出
                print("ZoomEye Command STDOUT:")
                print(result.stdout)
                print("ZoomEye Command STDERR:")
                print(result.stderr)

                # 检查输出文件是否存在
                if os.path.exists(output_path):
                    print(f"Output image found at: {output_path}")
                else:
                    print(f"No output image at: {output_path}")

                # 尝试读取结果文件
                result_file = os.path.join(output_dir, "zoomeye_result.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        zoom_eye_result = json.load(f)
                    print("ZoomEye Result:", zoom_eye_result)

                zoom_eye_answer = result.stdout.strip()
                logger.info(f"ZoomEye Answer: {zoom_eye_answer}")

            except subprocess.CalledProcessError as e:
                # 详细记录执行错误
                print("Command failed with return code:", e.returncode)
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)
                zoom_eye_answer = None

            except subprocess.TimeoutExpired:
                print("ZoomEye command timed out")
                zoom_eye_answer = None

            except Exception as e:
                print(f"Unexpected error running ZoomEye: {e}")
                zoom_eye_answer = None

        except Exception as e:
            print(f"Error preparing ZoomEye command: {e}")
            zoom_eye_answer = None




            result = subprocess.run(
                zoom_eye_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            zoom_eye_answer = result.stdout.strip()
            logger.info(f"ZoomEye Answer: {zoom_eye_answer}")
            
            # 使用答案生成负面图像
            if args.use_diffusion and zoom_eye_answer:
                neg_image = generate_image_stable_diffusion(pipe, zoom_eye_answer)
                neg_image_path = os.path.join(output_dir, f"neg_image_{batch_id}.jpg")
                neg_image.save(neg_image_path)
                neg_image_tensor = image_processor.preprocess(neg_image, return_tensor='pt')['pixel_values'][0]
                image_neg = torch.tensor(neg_image_tensor)
                
                if dist.get_rank() == 0 and image_output_dirs is not None:
                    output_path = os.path.join(image_output_dirs['zoom_eye_outputs'], f"output_{batch_id}.jpg")
                    neg_image.save(os.path.join(image_output_dirs['neg_images'], f'zoom_eye_neg_image_{batch_id}.jpg'))
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running ZoomEye: {str(e)}")
            logger.error(f"stderr: {e.stderr}")
            zoom_eye_answer = None
            image_neg = None

        # 后续的原有代码继续保持不变
        image_pos = None
        image_neg = None

        

        if args.use_ritual:
            # ==============================================
            #              Image Transforms
            # ==============================================
            raw_image = Image.open(image_path[0])
            pos_aug = random.choice(list(aug_dict.keys()))

            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0] 
                image_pos = torch.tensor(image_pos)
                
            pos_aug_counter[pos_aug] += 1
            logger.info(f"RITUAL Transformation: {pos_aug}")
            
            if dist.get_rank() == 0 and image_output_dirs is not None:
                raw_image.save(os.path.join(image_output_dirs['raw_images'], f'raw_image_{batch_id}.png'))
                if image_pos is not None:
                    Image.fromarray((image_pos.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
                        os.path.join(image_output_dirs['raw_images'], f'raw_image_pos_{batch_id}.png')
                    )
            
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)
            
        elif args.use_diffusion:
            raw_image = Image.open(image_path[0])
            # pos_aug = random.choice(list(aug_dict.keys()))

            # if pos_aug is not None:
            #     raw_image_pos = aug_dict[pos_aug](raw_image)
            #     image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0] 
            #     image_pos = torch.tensor(image_pos)
                
            # pos_aug_counter[pos_aug] += 1
            # logger.info(f"RITUAL Transformation: {pos_aug}")
            
            
            conv_out = Conversation(
                system="A chat between a curious human and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
                roles=("USER", "ASSISTANT"),
                version="v1",
                messages=[],
                offset=0,
                sep_style=SeparatorStyle.TWO,
                sep=" ",
                sep2="</s>",
            )
            
            # qs_desc = qs + " If yes, describe all relevant details of this object only. If no, describe all existing objects in the image."
            qs_desc = qs + " Briefly describe relevant details."
            qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs_desc # for opera? setting
            # qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs + " Please answer this question with one word." # for VCD setting
            conv_out.append_message(conv_out.roles[0], qu_out)
            conv_out.append_message(conv_out.roles[1], None)
            prompt_out = conv_out.get_prompt()
            
            input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2


#             with torch.inference_mode():
#                 with torch.no_grad():
#                     output_ids, js_list = model.generate(
#                         input_ids,
#                         images=image.unsqueeze(0).half().cuda(),
#                         images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
#                         images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
#                         neg_image=(neg_image_tensor.unsqueeze(0).half().cuda() if image_neg is not None else None),  # 添加neg_image参数
#                         do_sample=False,
#                         temperature=args.temperature,
#                         top_p=args.top_p,
#                         top_k=args.top_k,
#                         max_new_tokens=3,
#                         use_cache=True,
#                         use_ritual=args.use_ritual,
#                         use_vcd=args.use_vcd,
#                         use_m3id=args.use_m3id,
#                         use_diffusion=args.use_diffusion,
#                         degf_alpha_pos=args.degf_alpha_pos,
#                         degf_alpha_neg=args.degf_alpha_neg,
#                         degf_beta=args.degf_beta,
#                     )
                    
                    
#                     print("Debug: Accessing logits")
#                     print(f"next_token_logits shape: {next_token_logits.shape}")
#                     print(f"next_token_logits first 10 values:\n{next_token_logits[0, :10].cpu().numpy()}")

#                     if 'next_token_logits_neg' in locals():
#                         print(f"next_token_logits_neg shape: {next_token_logits_neg.shape}")
#                         print(f"next_token_logits_neg first 10 values:\n{next_token_logits_neg[0, :10].cpu().numpy()}")

#                     if 'next_token_logits_neg_2' in locals():
#                         print(f"next_token_logits_neg_2 shape: {next_token_logits_neg_2.shape}")
#                         print(f"next_token_logits_neg_2 first 10 values:\n{next_token_logits_neg_2[0, :10].cpu().numpy()}")
                        
                        
            with torch.inference_mode():
                with torch.no_grad():
                    # 准备打印 logits 的变量
                    global_next_token_logits = None

                    def capture_logits_hook(module, input, output):
                        nonlocal global_next_token_logits
                        global_next_token_logits = output.logits[:, -1, :]

                    # 注册一个前向钩子来捕获 logits
                    hook = model.register_forward_hook(capture_logits_hook)

                    try:
                        output_ids, js_list = model.generate(
                            input_ids,
                            images=image.unsqueeze(0).half().cuda(),
                            images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                            images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                            do_sample=False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            max_new_tokens=3,
                            use_cache=True,
                            use_ritual=args.use_ritual,
                            use_vcd=args.use_vcd,
                            use_m3id=args.use_m3id,
                            use_diffusion=args.use_diffusion,
                            degf_alpha_pos=args.degf_alpha_pos,
                            degf_alpha_neg=args.degf_alpha_neg,
                            degf_beta=args.degf_beta,
                        )
                    finally:
                        # 确保钩子被移除
                        hook.remove()

                    # 打印 logits 信息
                    if global_next_token_logits is not None:
                        print("Debug: Accessing logits")
                        print(f"next_token_logits shape: {global_next_token_logits.shape}")
                        print(f"next_token_logits first 10 values:\n{global_next_token_logits[0, :10].cpu().numpy()}")
                        
                    
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            description = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            description = description.strip()
            if description.endswith(stop_str):
                description = description[:-len(stop_str)]
            pred_list2 = recorder(description, pred_list2)
            logger.info(f"V: {image_path}")
            logger.info(f"Q: {qs_desc}")
            logger.info(f"D: {description}")
            
            raw_image = Image.open(image_path[0])
            raw_image.save(os.path.join(output_dir, f'raw_image_{batch_id}.png'))
            raw_image_neg = generate_image_stable_diffusion(pipe, description)
            raw_image_neg.save(os.path.join(output_dir, f'neg_image_{batch_id}.png'))
            image_neg = image_processor.preprocess(raw_image_neg, return_tensor='pt')['pixel_values'][0]
            image_neg = torch.tensor(image_neg)
            
            if dist.get_rank() == 0 and image_output_dirs is not None:
                raw_image.save(os.path.join(image_output_dirs['raw_images'], f'raw_image_{batch_id}.png'))
                raw_image_neg.save(os.path.join(image_output_dirs['neg_images'], f'neg_image_{batch_id}.png'))
        
        
        # ==============================================
        #              Text prompt setting
        # ==============================================
        conv_out = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs # for opera? setting
        # qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs + " Please answer this question with one word." # for VCD setting
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

#         with torch.inference_mode():
#             with torch.no_grad():
#                 output_ids, js_list = model.generate(
#                     input_ids,
#                     images=image.unsqueeze(0).half().cuda(),
#                     images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
#                     images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
#                     do_sample=False, # If False, greedy decoding
#                     temperature=args.temperature,# args.temperature
#                     top_p=args.top_p,
#                     top_k=args.top_k,
#                     max_new_tokens=3, # args.max_new_tokens
#                     use_cache=True,
#                     use_ritual=args.use_ritual,
#                     use_vcd=args.use_vcd,
#                     use_m3id=args.use_m3id,
#                     use_diffusion=args.use_diffusion,
#                     degf_alpha_pos=args.degf_alpha_pos,
#                     degf_alpha_neg=args.degf_alpha_neg,
#                     degf_beta=args.degf_beta,
#                 )
#                 print("Debug: Accessing logits")
#                 print(f"next_token_logits shape: {next_token_logits.shape}")
#                 print(f"next_token_logits first 10 values:\n{next_token_logits[0, :10].cpu().numpy()}")

#                 if 'next_token_logits_neg' in locals():
#                     print(f"next_token_logits_neg shape: {next_token_logits_neg.shape}")
#                     print(f"next_token_logits_neg first 10 values:\n{next_token_logits_neg[0, :10].cpu().numpy()}")

#                 if 'next_token_logits_neg_2' in locals():
#                     print(f"next_token_logits_neg_2 shape: {next_token_logits_neg_2.shape}")
#                     print(f"next_token_logits_neg_2 first 10 values:\n{next_token_logits_neg_2[0, :10].cpu().numpy()}")
                    
    
    
        with torch.inference_mode():
            with torch.no_grad():
                # 准备打印 logits 的变量
                global_next_token_logits = None

                def capture_logits_hook(module, input, output):
                    nonlocal global_next_token_logits
                    global_next_token_logits = output.logits[:, -1, :]

                # 注册一个前向钩子来捕获 logits
                hook = model.register_forward_hook(capture_logits_hook)

                try:
                    output_ids, js_list = model.generate(
                        input_ids,
                        images=image.unsqueeze(0).half().cuda(),
                        images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                        images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                        do_sample=False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=3,
                        use_cache=True,
                        use_ritual=args.use_ritual,
                        use_vcd=args.use_vcd,
                        use_m3id=args.use_m3id,
                        use_diffusion=args.use_diffusion,
                        degf_alpha_pos=args.degf_alpha_pos,
                        degf_alpha_neg=args.degf_alpha_neg,
                        degf_beta=args.degf_beta,
                    )
                finally:
                    # 确保钩子被移除
                    hook.remove()

                # 打印 logits 信息
                if global_next_token_logits is not None:
                    print("Debug: Accessing logits")
                    print(f"next_token_logits shape: {global_next_token_logits.shape}")
                    print(f"next_token_logits first 10 values:\n{global_next_token_logits[0, :10].cpu().numpy()}")
                    
                    
                    
                    
                    
                
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        pred_list = recorder(outputs, pred_list)
        # js_list_first.append(js_list[0])
        logger.info(f"[VQA for ritual]")
        logger.info(f"V: {image_path}")
        logger.info(f"Q: {qs}")
        logger.info(f"A: {outputs}")
        
        
        

        if args.use_diffusion:
            # print each token in a list
            output_list = [tokenizer.decode([token]) for token in output_ids[:, input_token_len:].tolist()[0]]
            printout = ""
            for token, js in zip(output_list, js_list):
                printout += f"{token}({js}) "
            logger.info(printout)
        if label == 1: logger.info(f"GT: Yes")
        elif label == 0: logger.info(f"GT: No")

        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        
        logger.info(f"="*50)
        # np.save(f"{experiment_dir}/pred_list.npy", pred_list)
        
        np.save(f"{experiment_output_dir}/pred_list.npy", pred_list)
        
        np.save(f"{experiment_output_dir}/pred_list2.npy", pred_list2)
        np.save(f"{experiment_output_dir}/js_list.npy", js_list_first)
        np.save(f"{experiment_output_dir}/label_list.npy", label_list)

    if len(pred_list) != 0:
        logger.info(vars(args))
        # logger.info("Prompt for Aug:", prompt_aug)
        # logger.info("Prompt for ritual:", prompt_out)
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        if args.use_ritual:
            logger.info(f"RITUAL Transformation: {pos_aug_counter}")

        np.save(f"{experiment_output_dir}/pred_list.npy", pred_list)
        np.save(f"{experiment_output_dir}/js_list.npy", js_list_first)

if __name__ == "__main__":
    main()




