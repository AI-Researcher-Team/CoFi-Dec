#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced CHAIR Dataset Evaluation Script
针对大规模视觉语言模型的增强评估脚本
"""

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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')

# 导入必要的模块
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
from PIL import Image
from torchvision.transforms import v2

from chair_loader import CHAIRDataset

from degf_utils.degf_sample import evolve_degf_sampling
from degf_utils.vcd_add_noise import add_diffusion_noise
from degf_utils.image_variation import get_image_variation_pipeline, apply_image_variation
from degf_utils.image_similarity import get_clip_similarity
from degf_utils.image_generation import get_image_generation_pipeline, generate_image_stable_diffusion

# 设置多进程共享策略
evolve_degf_sampling()
torch.multiprocessing.set_sharing_strategy('file_system')

# 忽略警告
import warnings
warnings.filterwarnings(action='ignore')

def str2bool(v):
    """
    将字符串转换为布尔值
    
    Args:
        v (str/bool): 输入的值
    
    Returns:
        bool: 转换后的布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def recorder(out, pred_list):
    """
    根据输出文本记录预测结果
    
    Args:
        out (str): 模型输出文本
        pred_list (list): 预测列表
    
    Returns:
        list: 更新后的预测列表
    """
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

def print_acc(pred_list, label_list, logger):
    """
    计算并打印性能指标
    
    Args:
        pred_list (list): 预测标签列表
        label_list (list): 真实标签列表
        logger (logging.Logger): 日志记录器
    
    Returns:
        tuple: 准确率、精确率、召回率、F1分数和正样本比例
    """
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list) if len(pred_list) > 0 else 0

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

    logger.info('Confusion Matrix: TP\tFP\tTN\tFN')
    logger.info('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    # 计算各项指标
    precision = float(TP) / float(TP + FP) if TP + FP > 0 else 0
    recall = float(TP) / float(TP + FN) if TP + FN > 0 else 0
    f1 = 2*precision*recall / (precision + recall) if precision + recall > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return acc, precision, recall, f1, yes_ratio

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="Enhanced CHAIR Evaluation on LVLMs.")
    
    # 模型和推理参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--model_base", type=str, default="llava", help="基础模型类型")
    parser.add_argument("--conv_mode", type=str, default="llava_v1", help="对话模式")
    
    # 采样参数
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p采样")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k采样")
    
    # 数据集和路径参数
    parser.add_argument("--data_path", type=str, required=True, help="图像数据路径")
    parser.add_argument("--anno_path", type=str, required=True, help="标注文件路径")
    parser.add_argument("--log_path", type=str, default="./logs", help="日志路径")
    parser.add_argument("--out_path", type=str, default="./results", help="输出结果路径")
    
    # 实验控制参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=2, help="工作进程数")
    
    # 数据采样和输出参数
    parser.add_argument("--data_ratio", type=float, default=1.0, help="使用数据集的比例 (0-1)")
    parser.add_argument("--output_image_dir", type=str, default="./outputs", help="输出图像基础目录")
    
    # 增强方法选择
    parser.add_argument("--use_ritual", type=str2bool, default=False, help="使用RITUAL图像增强")
    parser.add_argument("--use_vcd", type=str2bool, default=False, help="使用VCD方法")
    parser.add_argument("--noise_step", type=int, default=500, help="VCD噪声步骤")
    parser.add_argument("--use_m3id", type=str2bool, default=False, help="使用M3ID方法")
    parser.add_argument("--use_diffusion", type=str2bool, default=False, help="使用扩散生成方法")
    
    # 高级方法超参数
    parser.add_argument("--degf_alpha_pos", type=float, default=3, help="DeGF正样本alpha")
    parser.add_argument("--degf_alpha_neg", type=float, default=1, help="DeGF负样本alpha")
    parser.add_argument("--degf_beta", type=float, default=0.1, help="DeGF beta参数")
    
    # 生成参数
    parser.add_argument("--num_eval_samples", type=int, default=500, help="评估样本数")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="生成新Token的最大数量")
    parser.add_argument("--experiment_index", type=int, default=0, help="当前实验运行索引")

    args = parser.parse_known_args()[0]
    return args

def main():
    """
    主执行函数，包含模型和数据加载、实验设置等核心逻辑
    """
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 设置分布式计算
        dist_util.setup_dist(args)
        device = dist_util.device()

        # 创建实验输出目录
        if dist.get_rank() == 0:
            # 创建基础输出目录
            os.makedirs(args.output_image_dir, exist_ok=True)

            # 根据实验参数创建子文件夹
            model_string_name = os.path.basename(args.model_path.rstrip('/'))
            experiment_subdir = (
                f"{model_string_name}_"
                f"ritual{args.use_ritual}_"
                f"vcd{args.use_vcd}_"
                f"m3id{args.use_m3id}_"
                f"diffusion{args.use_diffusion}_"
                f"exp{args.experiment_index}"
            )
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

            # 设置日志
            logger = create_logger(experiment_output_dir)
            logger.info(f"Experiment directory created at {experiment_output_dir}")
        else:
            logger = create_logger(None)
            image_output_dirs = None

        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # 模型初始化
        logger.info('Initializing Model')
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

        # 数据集加载
        chair_dataset = CHAIRDataset(
            data_path=args.data_path,
            anno_path=args.anno_path,
            trans=image_processor,
            model=args.model_base
        )

        # 数据子集采样
        total_len = len(chair_dataset)
        subset_size = int(total_len * args.data_ratio)
        indices = torch.randperm(total_len)[:subset_size]
        subset_dataset = torch.utils.data.Subset(chair_dataset, indices)

        # 数据加载器
        chair_loader = DataLoader(
            subset_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            drop_last=False
        )

        # 图像增强字典
        aug_dict = {
            'horizontal flip': v2.RandomHorizontalFlip(p=1),
            'vertical flip': v2.RandomVerticalFlip(p=1),
            'rotation': v2.RandomRotation(degrees=180),
            'color jitter': v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
            'gaussian blur': v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
            'crop': v2.RandomResizedCrop(size=336),
        }
        
        # 增强方法统计
        pos_aug_counter = {k:0 for k in aug_dict}
        pos_aug_counter.update({None: 0})

        # 性能追踪变量
        pred_list, label_list = [], []
        pred_list2 = []
        js_list_first = []

        # 初始化扩散模型管道
        if args.use_diffusion:
            pipe = get_image_generation_pipeline()

        
        # 创建虚拟标签列表
        label_list = [1] * args.num_eval_samples  # 默认所有样本为正类
        
        
        
        # 主评估循环
        logger.info("Start evaluation...")
        for batch_id, data in tqdm(enumerate(chair_loader), total=args.num_eval_samples):
            if batch_id == args.num_eval_samples:
                break
                
            # 提取批次数据
            img_id = data["image_id"]
            image_path = data["image_path"]
            image = data["image"]

            # 默认查询
            qs = "Please describe this image in detail."

            # ZoomEye 子进程调用
            # ZoomEye部分
            try:
                # 创建输出目录
                zoom_eye_output_dir = os.path.join(
                    args.out_path, 
                    f"zoom_eye_outputs_{batch_id}"
                )
                os.makedirs(zoom_eye_output_dir, exist_ok=True)

                # 准备ZoomEye输出路径
                zoom_eye_output_image = f"zoomeye_output_{batch_id}.jpg"
                zoom_eye_output_path = os.path.join(zoom_eye_output_dir, zoom_eye_output_image)

                # 安全地转义特殊字符
                def safe_escape(s):
                    return s.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")

                escaped_question = safe_escape(qs)
                escaped_image_path = safe_escape(image_path[0])
                escaped_model_path = safe_escape(args.model_path)
                escaped_output_path = safe_escape(zoom_eye_output_path)

                zoom_eye_cmd = [
                    "conda", "run", "-n", "zoom_eye",
                    "python", "-c",
                    f'''import sys
import os
import json

# 确保demo目录存在
os.makedirs("demo", exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append("{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")
from ZoomEye.ZoomEye.eval.demo import run_demo

try:
    result = run_demo(
        model_path="{escaped_model_path}",
        input_image="{escaped_image_path}",
        question="{escaped_question}",
        output_image_path="{escaped_output_path}"
    )

    # 确保result是字典
    if not isinstance(result, dict):
        result = {{"description": str(result)}}

    with open("{zoom_eye_output_dir}/zoomeye_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

except Exception as e:
    print(f"Error in ZoomEye processing: {{str(e)}}")
    sys.exit(1)
'''
                ]

                # 执行ZoomEye命令
                try:
                    result = subprocess.run(
                        zoom_eye_cmd,
                        capture_output=True,  
                        text=True,
                        check=True,  
                        timeout=120  
                    )

                    # 尝试读取结果文件
                    result_file = os.path.join(zoom_eye_output_dir, "zoomeye_result.json")
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            zoom_eye_result = json.load(f)
                        zoom_eye_answer = zoom_eye_result.get('description', '')
                        logger.info(f"ZoomEye Answer: {zoom_eye_answer}")
                    else:
                        logger.warning(f"No ZoomEye result file found at {result_file}")
                        zoom_eye_answer = None

                except subprocess.CalledProcessError as e:
                    logger.error(f"ZoomEye subprocess error: {e}")
                    logger.error(f"Command stdout: {e.stdout}")
                    logger.error(f"Command stderr: {e.stderr}")
                    zoom_eye_answer = None

                except subprocess.TimeoutExpired:
                    logger.error("ZoomEye command timed out")
                    zoom_eye_answer = None

                except Exception as e:
                    logger.error(f"Unexpected ZoomEye error: {str(e)}")
                    zoom_eye_answer = None

            # 在异常处理之前定义变量
            except Exception as e:
                logger.error(f"Error preparing ZoomEye command: {str(e)}")
                zoom_eye_output_dir = os.path.join(args.out_path, f"zoom_eye_outputs_{batch_id}")
                os.makedirs(zoom_eye_output_dir, exist_ok=True)
                zoom_eye_answer = None
                # 图像增强和负样本准备
            image_pos = None
            image_neg = None

            # Ritual 图像增强方法
            if args.use_ritual:
                raw_image = Image.open(image_path[0])
                pos_aug = random.choice(list(aug_dict.keys()))

                if pos_aug is not None:
                    raw_image_pos = aug_dict[pos_aug](raw_image)
                    image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0] 
                    image_pos = torch.tensor(image_pos)
                    
                pos_aug_counter[pos_aug] += 1
                logger.info(f"RITUAL Transformation: {pos_aug}")

                # 保存原始和增强图像
                if dist.get_rank() == 0 and image_output_dirs is not None:
                    raw_image.save(os.path.join(image_output_dirs['raw_images'], f'raw_image_{batch_id}.png'))
                    if image_pos is not None:
                        Image.fromarray((image_pos.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
                            os.path.join(image_output_dirs['raw_images'], f'raw_image_pos_{batch_id}.png')
                        )

            # VCD 噪声添加方法
            elif args.use_vcd:
                image_neg = add_diffusion_noise(image, args.noise_step)
            
            # 扩散方法
            elif args.use_diffusion and zoom_eye_answer:
                # 使用ZoomEye生成的描述生成负样本图像
                raw_image_neg = generate_image_stable_diffusion(pipe, zoom_eye_answer)
                
                # 保存负样本图像
                if dist.get_rank() == 0 and image_output_dirs is not None:
                    raw_image_neg.save(os.path.join(image_output_dirs['neg_images'], f'neg_image_{batch_id}.png'))
                
                # 预处理负样本图像
                image_neg = image_processor.preprocess(raw_image_neg, return_tensor='pt')['pixel_values'][0]
                image_neg = torch.tensor(image_neg)

            # 模型推理前的会话设置
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
            
            # 准备输入
            qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv_out.append_message(conv_out.roles[0], qu_out)
            conv_out.append_message(conv_out.roles[1], None)
            prompt_out = conv_out.get_prompt()
            
            # 分词和模型输入准备
            input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

            
            
            # 在主函数开始处添加
            os.makedirs(args.out_path, exist_ok=True)
            out_jsonl_path = os.path.join(args.out_path, f"exp_{args.experiment_index:03d}.jsonl")
            
            
            
            # 图像维度处理
            def ensure_4d_tensor(image):
                """
                确保图像张量是4D的

                Args:
                    image (torch.Tensor): 输入图像张量

                Returns:
                    torch.Tensor: 4D图像张量
                """
                # 如果是3D张量，增加batch维度
                if image.dim() == 3:
                    return image.unsqueeze(0)
                # 如果已经是4D，直接返回
                elif image.dim() == 4:
                    return image
                # 其他情况抛出异常
                else:
                    raise ValueError(f"Expected images to be 3D or 4D, got {image.dim()}D")

            # 在模型推理部分使用
            # 模型推理部分
            
            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=truncated_input_ids,
                        attention_mask=attention_mask,
                        images=image_input,
                        images_pos=image_pos_input,
                        images_neg=image_neg_input,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        max_new_tokens=min(64, context_len),  # 不超过context长度
                        use_cache=True,
                        use_ritual=args.use_ritual,
                        use_vcd=args.use_vcd,
                        use_m3id=args.use_m3id,
                        use_diffusion=args.use_diffusion,
                        degf_alpha_pos=args.degf_alpha_pos,
                        degf_alpha_neg=args.degf_alpha_neg,
                        degf_beta=args.degf_beta
                    )
            except Exception as e:
                logger.error(f"Model generation error: {e}")
                logger.error(f"Input details: input_ids shape {truncated_input_ids.shape}, "
                             f"image shape {image_input.shape}")
                outputs = "Image description generation failed"
                import traceback
                traceback.print_exc()
            
#             try:
#                 with torch.inference_mode():
#                     with torch.no_grad():
#                         # 截断输入序列长度
#                         truncated_input_ids = input_ids[:, :77]
#                         truncated_attention_mask = attention_mask[:, :77] if 'attention_mask' in locals() else None

#                         # 在调用generate()之前，添加attention_mask的创建
#                         attention_mask = torch.ones_like(truncated_input_ids, dtype=torch.long)
                        
#                         output_ids, js_list = model.generate(
#                             truncated_input_ids,
#                             input_ids=truncated_input_ids,
#                             attention_mask=attention_mask,  # 添加attention_mask
#                             images=image.unsqueeze(0).half().cuda(),
#                             images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
#                             images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
#                             do_sample=True,  # 改为True以获得更丰富的输出
#                             temperature=0.7,  # 调整温度
#                             top_p=0.9,
#                             top_k=50,
#                             max_new_tokens=64,  # 调整最大生成token数
#                             use_cache=True,
#                             use_ritual=args.use_ritual,
#                             use_vcd=args.use_vcd,
#                             use_m3id=args.use_m3id,
#                             use_diffusion=args.use_diffusion,
#                             degf_alpha_pos=args.degf_alpha_pos,
#                             degf_alpha_neg=args.degf_alpha_neg,
#                             degf_beta=args.degf_beta,
#                         )

#                 # 处理模型输出
#                 input_token_len = truncated_input_ids.shape[1]
#                 n_diff_input_output = (truncated_input_ids != output_ids[:, :input_token_len]).sum().item()
#                 if n_diff_input_output > 0:
#                     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

#                 outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
#                 outputs = outputs.strip()
#                 if outputs.endswith(stop_str):
#                     outputs = outputs[:-len(stop_str)]
#                 outputs = outputs.strip()

#                 # 如果输出太短，添加默认描述
#                 if len(outputs) < 10:
#                     outputs = "The image shows various elements and details."

#             except Exception as e:
#                 logger.error(f"Model generation error: {e}")
#                 outputs = "Image description generation failed"
#                 import traceback
#                 traceback.print_exc()
            
            
            
#             try:
#                 with torch.inference_mode():
#                     with torch.no_grad():
#                         # 确保图像张量正确
#                         image_4d = ensure_4d_tensor(image)
#                         image_pos_4d = ensure_4d_tensor(image_pos) if image_pos is not None else None
#                         image_neg_4d = ensure_4d_tensor(image_neg) if image_neg is not None else None

#                         output_ids, js_list = model.generate(
#                             input_ids,
#                             images=image_4d.half().cuda(),
#                             images_pos=(image_pos_4d.half().cuda() if image_pos_4d is not None else None),
#                             images_neg=(image_neg_4d.half().cuda() if image_neg_4d is not None else None),
#                             do_sample=False,
#                             temperature=args.temperature,
#                             top_p=args.top_p,
#                             top_k=args.top_k,
#                             max_new_tokens=3,
#                             use_cache=True,
#                             use_ritual=args.use_ritual,
#                             use_vcd=args.use_vcd,
#                             use_m3id=args.use_m3id,
#                             use_diffusion=args.use_diffusion,
#                             degf_alpha_pos=args.degf_alpha_pos,
#                             degf_alpha_neg=args.degf_alpha_neg,
#                             degf_beta=args.degf_beta,
#                         )

            

#                 # 处理模型输出
#                 input_token_len = input_ids.shape[1]
#                 n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
#                 if n_diff_input_output > 0:
#                     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

#                 outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
#                 outputs = outputs.strip()
#                 if outputs.endswith(stop_str):
#                     outputs = outputs[:-len(stop_str)]
#                 outputs = outputs.strip()

#             except Exception as e:
#                 logger.error(f"Model generation error: {e}")
#                 outputs = "Generation failed"
#                 # 可以添加详细的异常追踪
#                 import traceback
#                 traceback.print_exc()

            # 保存结果到jsonl文件
            img_save = {
                "image_id": img_id.item(),
                "caption": outputs
            }
            with open(out_jsonl_path, "a") as f:
                json.dump(img_save, f)
                f.write('\n')

            # 记录预测
            pred_list = recorder(outputs, pred_list)

            # 日志记录
            logger.info(f"[VQA Evaluation]")
            logger.info(f"Image: {image_path}")
            logger.info(f"Question: {qs}")
            logger.info(f"Model Output: {outputs}")

            # 性能统计
            if len(pred_list) > 0 and len(label_list) > 0:
                acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
                
                # 格式化指标
                acc = round(acc*100, 2)
                precision = round(precision*100, 2)
                recall = round(recall*100, 2)
                f1 = round(f1*100, 2)
                yes_ratio = round(yes_ratio*100, 2)
                
                logger.info(
                    f"Current Metrics - "
                    f"acc: {acc}, precision: {precision}, "
                    f"recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
                )

        # 最终性能总结
        if len(pred_list) > 0:
            final_acc, final_precision, final_recall, final_f1, final_yes_ratio = print_acc(pred_list, label_list, logger)
            
            # 格式化最终指标
            final_acc = round(final_acc*100, 2)
            final_precision = round(final_precision*100, 2)
            final_recall = round(final_recall*100, 2)
            final_f1 = round(final_f1*100, 2)
            final_yes_ratio = round(final_yes_ratio*100, 2)
            
            logger.info("\n===== Final Evaluation Results =====")
            logger.info(
                f"Final Metrics - "
                f"acc: {final_acc}, precision: {final_precision}, "
                f"recall: {final_recall}, f1: {final_f1}, yes_ratio: {final_yes_ratio}"
            )

            # 保存结果
            np.save(f"{experiment_output_dir}/pred_list.npy", pred_list)
            np.save(f"{experiment_output_dir}/label_list.npy", label_list)

        # 结束
        logger.info("Evaluation completed.")

    except Exception as e:
        print(f"评估过程发生错误：{e}")
        import traceback
        traceback.print_exc()

def run_evaluation():
    """
    运行评估的主入口函数
    """
    try:
        main()
    except Exception as e:
        print(f"评估执行出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()
