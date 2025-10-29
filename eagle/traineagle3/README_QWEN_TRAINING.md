# Training EAGLE-3 for Qwen Models

## Summary of Findings

### Qwen2.5-VL Compatibility Issue

**Result**: ❌ Qwen2.5-VL **cannot** be directly used with current EAGLE-3 training code.

**Reasons**:
1. Qwen2.5-VL is a **multimodal** (vision-language) model with special architecture
2. Uses `mrope` (multimodal RoPE) which is incompatible with standard Qwen2ForCausalLM
3. Cannot be loaded with `Qwen2ForCausalLM.from_pretrained()`
4. EAGLE-3 is designed for text-only language models

### Recommended Solution

Use **Qwen2.5-3B-Instruct** (text-only version) instead:

```bash
# Download the text-only model
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir /home/qianxu/EAGLE/eagle/traineagle3/Qwen2.5-3B-Instruct
```

Or use Qwen3 models which are already supported in the EAGLE codebase.

## Setup Complete

The following modifications have been made to enable Qwen training:

### 1. Modified Files

- **cnets.py**: Added Qwen2ForCausalLM support
  - Added `model_type` parameter to Model class
  - Imports `Qwen2ForCausalLM` from transformers

- **main.py**: Added `--model_type` argument
  - Can specify `--model_type qwen` or `--model_type llama`

- **config_qwen25vl.json**: Created EAGLE config for Qwen2.5
  - Based on Qwen2.5-VL text config
  - Ready to use with text-only Qwen2.5 models

### 2. Test Files Created

- **sample_train.jsonl**: Small training dataset (3 samples)
- **sample_test.jsonl**: Small test dataset (1 sample)
- **test_qwen_setup.py**: Setup verification script

## How to Train EAGLE-3 for Qwen

### Prerequisites

1. **Environment**: Use the `eagle` conda environment
   ```bash
   conda activate eagle
   ```

2. **Model**: Download Qwen2.5-3B-Instruct (text-only)
   ```bash
   cd /home/qianxu/EAGLE/eagle/traineagle3
   huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir Qwen2.5-3B-Instruct
   ```

3. **Data**: Prepare training data in JSONL format:
   ```json
   {"id": "1", "conversations": [
     {"from": "human", "value": "Your question"},
     {"from": "gpt", "value": "Assistant response"}
   ]}
   ```

### Training Command

```bash
cd /home/qianxu/EAGLE/eagle/traineagle3

deepspeed main.py \
  --deepspeed_config ds_config.json \
  --basepath Qwen2.5-3B-Instruct \
  --trainpath /path/to/your/train.jsonl \
  --testpath /path/to/your/test.jsonl \
  --savedir ./output_qwen25_eagle3 \
  --model_type qwen
```

### Configuration Details

**DeepSpeed Config** (ds_config.json):
- ZeRO Stage 2
- Learning rate: 5e-5
- Batch size: 1 per GPU
- Gradient accumulation: 2
- FP16 training

**EAGLE Config** (config_qwen25vl.json):
- Hidden size: 2048
- Num layers: 1 (draft model)
- Vocab size: 151936
- Based on Qwen2.5 architecture

**Training Config** (in main.py):
- Max sequence length: 2048
- Epochs: 40
- Gradient checkpointing: enabled

## Alternative: Use Existing Qwen3 Support

The EAGLE repository already has Qwen3 support. If you have a Qwen3 model, you can use:

1. Use files from `/home/qianxu/EAGLE/eagle/model/modeling_qwen3_kv.py`
2. Follow existing Qwen3 training examples

## Troubleshooting

### Issue: "mrope" error
**Solution**: You're trying to use Qwen2.5-VL. Use text-only Qwen2.5 instead.

### Issue: "Cannot import Qwen2ForCausalLM"
**Solution**: Update transformers: `pip install transformers>=4.37.0`

### Issue: Out of memory
**Solution**:
- Reduce batch size in ds_config.json
- Increase gradient accumulation steps
- Enable ZeRO Stage 3

### Issue: Tokenizer warnings
**Solution**: This is normal for Qwen models. Training will still work.

## Next Steps

1. **Download text-only Qwen2.5 model**
2. **Prepare your training data**
3. **Run a small test training** with sample data to verify setup
4. **Scale up** with your full dataset

## Notes

- For vision-language models like Qwen2.5-VL, EAGLE training would require:
  - Modifying architecture to handle multimodal inputs
  - Supporting mrope (multimodal rope)
  - This is beyond the current EAGLE-3 implementation

- Stick with text-only models for EAGLE training
