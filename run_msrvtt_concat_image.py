""" running training and evaluation code for Efficient VideoQA

    Created by Chenyang Lyu
"""

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data import TensorDataset
from transformers import CLIPProcessor, CLIPModel, CLIPConfig

import argparse
import sklearn.metrics as metric
import glob
import logging
import os
import random
import numpy as np
import json
import pickle
import codecs
from PIL import Image
from tqdm import tqdm, trange
from sklearn.metrics import top_k_accuracy_score
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from modeling import MSRVTTConcat
import clip

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def train(args, model, train_dataset, preprocess, val_set=None):
    """ Training the model """
    tb_writer = SummaryWriter()

    train_dataset, train_video_names = train_dataset

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = (len(train_dataloader) * args.num_train_epochs) // args.gradient_accumulation_steps

    # Prepare optimizer for training
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_group_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(optimizer_group_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps * t_total),
                                                num_training_steps=t_total)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # Skip past any already trained steps
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1

            batch = tuple(t.to(args.device) for t in batch)

            train_video_ind = list(batch[0].cpu().numpy())

            all_image_frames = []
            for vid in train_video_ind:
                frame = preprocess(
                    Image.open('{}{}.jpg'.format(args.image_dir,
                                                 train_video_names['data'][vid].replace('.mp4', ''))))
                all_image_frames.append(frame.unsqueeze(0))

            all_image_frames = torch.cat(all_image_frames, dim=0).to(args.device)

            inputs = {'image_frames': all_image_frames,
                      'captions': batch[1],
                      }

            loss = model(inputs)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if global_step % args.eval_steps == 0 and val_set is not None:
                        evaluate(args, model, preprocess, val_set)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))

                # if args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

    tb_writer.close()
    global_step = 1 if global_step == 0 else global_step

    return global_step, tr_loss / global_step


def evaluate(args, model, preprocess, eval_dataset, prefix=""):
    eval_dataset, eval_video_names = eval_dataset

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_image_features = []
    all_text_features = []
    all_ans = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            eval_video_ind = list(batch[0].cpu().numpy())

            all_image_frames = []
            for vid in eval_video_ind:
                frame = preprocess(Image.open('{}{}.jpg'.format(args.image_dir,
                                                                eval_video_names['data'][vid].replace('.mp4', ''))))
                all_image_frames.append(frame.unsqueeze(0))

            all_image_frames = torch.cat(all_image_frames, dim=0).to(args.device)

            inputs = {'image_frames': all_image_frames,
                      'captions': batch[1].view(-1, 77),
                      }

            # image encoding without self-attention
            all_image_features.append(model.encode_image(inputs['image_frames']))

            attn_mask = 1 - (inputs['captions'] == 0).long()
            all_text_features.append(model.clip.get_text_features(inputs['captions'],
                                                                  attention_mask=attn_mask))
            all_ans.append(batch[2])

        all_image_features = torch.cat(all_image_features, dim=0)

        all_text_features = torch.cat(all_text_features, dim=0).view(all_image_features.size(0),
                                                                     -1, all_text_features[0].size(-1))

        # all_text_features = model.transform_text_to_inner_feature(all_text_features)
        # r_text_features = all_text_features.unsqueeze(0).repeat(all_text_features.size(0), 1, 1)  # added repeat
        #
        # all_image_features = model.transform_visual_to_inner_feature(all_image_features).transpose(0, 1)
        #
        # all_image_features = model.query_multi_attention(r_text_features.transpose(0, 1),
        #                                                  all_image_features, all_image_features)[0].transpose(0, 1)

        # with l2 norm
        t_video_features = torch.nn.functional.normalize(model.video_to_multimodal(all_image_features), p=2, dim=-1)
        t_text_features = torch.nn.functional.normalize(model.text_to_multimodal(all_text_features), p=2, dim=-1)

        # without l2 norm
        # t_video_features = model.video_to_multimodal(all_image_features)
        # t_text_features = model.text_to_multimodal(all_text_features)

        logit_scale = model.logit_scale.exp()

        # original multiply
        t_video_features = t_video_features.unsqueeze(1)
        logits = torch.bmm(logit_scale * t_video_features, t_text_features.transpose(1, 2)).squeeze(1)

        # text weighted multiply
        # t_text_features = t_text_features.unsqueeze(1)
        # logits = torch.bmm(logit_scale * t_video_features.transpose(0, 1)
        #                    t_text_features.transpose(1, 2)).squeeze(-1)

        logits = logits.cpu().numpy()

    all_ans = torch.cat(all_ans, dim=0).cpu().numpy()

    top_1 = top_k_accuracy_score(all_ans, logits, k=1)

    print('Accuracy: {}'.format(str(top_1 * 100)))
    print()
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, default="avsd",
                        help="the name of the training task (the dataset name)")
    parser.add_argument("--model_size", type=str, default="16",
                        help="the size of pre-trained CLIP model (ViT-16 or ViT-32)")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="the numebr of training epochs")
    parser.add_argument("--do_train", action="store_true",
                        help="whether to train the model or not")
    parser.add_argument("--do_eval", action="store_true",
                        help="whether to evaluate the model or not")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="the weight decay rate")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="the learning rate used to train the model")
    parser.add_argument("--warmup_steps", type=float, default=0.0,
                        help="the warm_up step rate")
    parser.add_argument("--seed", type=int, default=0,
                        help="the random seed used in model initialization and dataloader")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="the batch size used in training")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="the batch size used in evaluation")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="the logging steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="conduct evaluation every eval_steps")
    parser.add_argument("--device", type=int, default=0,
                        help="the device id used for training and evaluation")
    parser.add_argument("--n_gpu", type=int, default=1,
                        help="number of gpus being used")
    parser.add_argument("--attention_heads", type=int, default=8,
                        help="the attention heads used in multi head attention function")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--use_audio", action="store_true",
                        help="whether to use audio features or not")
    parser.add_argument("--search_key", type=str, default="summary",
                        help="the key used in retrieval")
    parser.add_argument("--frames_num", type=str, default="None",
                        help="the number of frames used in training and inference for concat-image schema")

    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda")
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 5.0

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    clip_model_name = 'openai/clip-vit-base-patch16'  # openai/clip-vit-large-patch14-336
    clip_processor_name = "ViT-B/16"
    clip_config = CLIPConfig.from_pretrained(clip_model_name)

    args.transformer_width = clip_config.projection_dim

    dataset_name = 'msrvtt'
    if args.frames_num == 'None':
        args.image_dir = 'data/{}/frames_resize/'.format(dataset_name)
    else:
        args.image_dir = 'data/{}/frames_resize_{}/'.format(dataset_name, args.frames_num)
    print(args.image_dir)
    if args.use_audio:
        model_prefix = 'use_audio'
    else:
        model_prefix = 'without_audio'
    args.output_dir = 'trained_models/audio_aware_dialog_to_video_retrieval/{}_{}_epochs-{}_lr-{}'.format(model_prefix,
                                                                          args.search_key,
                                                                          str(args.num_train_epochs),
                                                                          str(args.learning_rate))
    print(args.output_dir)
    data_dirs = ["data/{}/train.cache".format(dataset_name),
                 "data/{}/val.cache".format(dataset_name)]
    video_names = ["data/{}/train_video_names.json".format(dataset_name),
                   "data/{}/val_video_names.json".format(dataset_name)]

    all_images, all_captions = pickle.load(
        open(data_dirs[0], 'rb'))
    train_dataset = TensorDataset(torch.cat(all_images, dim=0),
                                  torch.cat(all_captions, dim=0))
    train_dataset = (train_dataset, json_load(video_names[0]))

    all_images, all_captions, all_ans = pickle.load(
        open(data_dirs[1], 'rb'))

    all_captions = [cap.unsqueeze(0) for cap in all_captions]
    eval_video_names = json_load(video_names[1])
    val_dataset = TensorDataset(torch.cat(all_images, dim=0),
                                torch.cat(all_captions, dim=0),
                                torch.cat(all_ans, dim=0)
                                )
    val_dataset = (val_dataset, {'data': eval_video_names['data']})

    model = MSRVTTConcat(args, clip_config)
    model.clip = model.clip.from_pretrained(clip_model_name)
    model.to(args.device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(clip_processor_name, device=device)
    del clip_model

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, preprocess, val_set=val_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), args.output_dir + 'model.pt')

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    if args.do_eval:
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            model = MSRVTTConcat(args, clip_config)
            model.load_state_dict(torch.load(checkpoint + 'model.pt'))
            model.eval()
            model.to(args.device)
            evaluate(args, model, preprocess, val_dataset)

    return


if __name__ == '__main__':
    main()
