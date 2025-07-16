import logging
import os
import random
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# from fireredasr.models.fireredasr_aed import FireRedAsrAed

from wenet.firered.adapter import Adapter
from wenet.firered.llm_tokenizer import DEFAULT_SPEECH_TOKEN, IGNORE_TOKEN_ID
from wenet.firered.llm_tokenizer import LlmTokenizerWrapper

def count_model_parameters(model):
    if not isinstance(model, torch.nn.Module):
        return 0, 0
    name = f"{model.__class__.__name__} {model.__class__}"
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size = num * 4.0 / 1024.0 / 1024.0 # float32, MB
    logging.info(f"#param of {name} is {num} = {size:.1f} MB (float32)")
    return num, size

class FireRedAsrLlm(nn.Module):
    @classmethod
    def load_encoder(cls, model_path):
        assert os.path.exists(model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = FireRedAsrAed.from_args(package["args"])
        if "model_state_dict" in package:
            model.load_state_dict(package["model_state_dict"], strict=False)
        encoder = model.encoder
        encoder_dim = encoder.odim
        return encoder, encoder_dim

    @classmethod
    def from_args(cls, args):
        logging.info(args)
        logging.info("Build FireRedAsrLlm")

        # Build Speech Encoder
        # 小红书提供的encoder
        encoder=args.wenet_ori_model.encoder
        encoder_dim=args.encoder_dim
        count_model_parameters(encoder)
        if args.freeze_encoder:
            logging.info(f"Frezee encoder")
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            encoder.eval()

        if args.use_flash_attn:
            attn_implementation = "flash_attention_2"
            if args.use_fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            attn_implementation = "eager"
            if args.use_fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        # 恢复Qwen2的参数
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16
            #torch_dtype=torch_dtype,
        )
        count_model_parameters(llm)

        # LLM Freeze or LoRA
        llm_dim = llm.config.hidden_size
        if args.freeze_llm:
            logging.info(f"Frezee LLM")
            for name, param in llm.named_parameters():
                param.requires_grad = False
            llm.eval()
        else:
            if args.use_lora:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=16,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "up_proj",
                        "gate_proj",
                        "down_proj",
                    ],
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )
                llm = get_peft_model(llm, lora_config)
                llm.print_trainable_parameters()
                if args.freeze_lora:
                    logging.info(f"Frezee Lora")
                    for name, param in llm.named_parameters():
                        param.requires_grad = False

        tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)
        assert tokenizer.pad_token_id == tokenizer.convert_tokens_to_ids("<|endoftext|>")
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN
        )

        # Build projector
        encoder_projector = Adapter(
            encoder_dim, llm_dim, args.encoder_downsample_rate)
        if args.freeze_adapter:
            logging.info(f"Frezee Lora")
            for name, param in encoder_projector.named_parameters():
                param.requires_grad = False
        count_model_parameters(encoder_projector)

        return cls(encoder, llm, encoder_projector,tokenizer,
                   args.freeze_encoder, args.freeze_llm)

    def __init__(self, encoder, llm, encoder_projector,tokenizer,
                 freeze_encoder, freeze_llm):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        # args
        self.freeze_encoder = freeze_encoder
        self.freeze_llm = freeze_llm
        self.llm_config = llm.config
        self.tokenizer = tokenizer

    def transcribe(self, padded_feat, feat_lengths, padded_input_ids, attention_mask,
                   beam_size=1, decode_max_len=0, decode_min_len=0,
                   repetition_penalty=1.0, llm_length_penalty=1.0, temperature=1.0):
        
        # encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)
        encoder_outs,enc_mask = self.encoder(padded_feat, feat_lengths)
        # print("feat_lengths",feat_lengths.shape,feat_lengths)
        # print("encoder_outs",encoder_outs.shape)
        # print("enc_mask",enc_mask.shape,enc_mask)
        enc_lengths = enc_mask.sum(dim=(1, 2))
        # print("enc_lengths",enc_lengths.shape,enc_lengths)
        # 这里第二个参数是需要传递enc_lengths的有效长度
        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)

        inputs_embeds, attention_mask, _ = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype), inputs_embeds, padded_input_ids, attention_mask,
                speech_lens=speech_lens
            )

        max_new_tokens = speech_features.size(1) if decode_max_len < 1 else decode_max_len
        max_new_tokens = max(1, max_new_tokens)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            do_sample=False,
            min_length=decode_min_len,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            length_penalty=llm_length_penalty,
            temperature=temperature,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        return generated_ids

    
        
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ):
        # Encoder
        encoder_out, encoder_mask = self.encoder(
            batch["feats"].to(device), 
            batch["feats_lengths"].to(device)
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        
        # Adapter
        speech_features, speech_lens = self.encoder_projector(encoder_out, encoder_out_lens)
        # Prompt
        input_ids, attention_mask, target_ids, clean_texts = \
            LlmTokenizerWrapper.preprocess_texts(
                batch["text_ori"], 
                tokenizer=self.tokenizer,
                max_len=128,
                decode=False
            )

        # Prompt的Embedding
        inputs_embeds = self.llm.get_input_embeddings()(input_ids.to(device))

        # 合并音频的Embedding和Prompt的Embedding
        (
            inputs_embeds,
            attention_mask,
            labels,
        ) = self._merge_input_ids_with_speech_features(
            speech_features.to(inputs_embeds.dtype), 
            inputs_embeds, 
            input_ids.to(device), 
            attention_mask.to(device), 
            target_ids.to(device), 
            speech_lens=None
        )

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = self.compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
            
        return {
            "loss": model_outputs.loss,
            "th_accuracy": acc,
        }
    
        
    def compute_accuracy(self,pad_outputs, pad_targets, ignore_label):
        """Calculate accuracy.
        Copied from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/utils/metric.py
        Args:
            pad_outputs (LongTensor): Prediction tensors (B, Lmax).
            pad_targets (LongTensor): Target label tensors (B, Lmax).
            ignore_label (int): Ignore label id.
    
        Returns:
            float: Accuracy value (0.0 - 1.0).
    
        """
        mask = pad_targets != ignore_label
        numerator = torch.sum(
            pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
        )
        denominator = torch.sum(mask)
        return numerator.float() / denominator.float()

    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None,
        speech_lens=None
    ):
        """
        Modified from: https://github.com/k2-fsa/icefall/blob/master/egs/speech_llm/ASR_LLM/whisper_llm_zh/model.py
        """
        # speech_lens = None
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.llm.config.pad_token_id)
        )
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == self.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != self.llm.config.default_speech_token_id
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )  # (N,U)
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                IGNORE_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_speech_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_speech_indices
            ]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        if speech_lens is not None:
            speech_pad_position = speech_to_overwrite.cumsum(-1) <= speech_lens[:, None]
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        if speech_lens is not None:
            speech_to_overwrite &= speech_pad_position
        final_attention_mask |= speech_to_overwrite

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == self.llm.config.pad_token_id
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels #, position_ids
