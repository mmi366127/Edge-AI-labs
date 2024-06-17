


CUDA_VISIBLE_DEVICES=1 python lm.py \
    --model_id "meta-llama/Llama-2-7b-hf" \
    --only_search k_proj v_proj o_proj \
    --param_ratio_target 0.8 \
    --use_cache 