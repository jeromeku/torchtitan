TODO
- auxiliary router loss - use HF auxiliary loss func -- need to save router logits from each layer then calculate at lm head level 
    - revert expert_bias
- pipelining_sft - see PP implementation
- FSDP2 + TP + PP benchmarking
- DeepEP
- TE GroupedLinear


--- 

- kv cache inference attention
- 
- model conversion
- lmeval
- throughput benchmarking
- fix apply_fsdp: only fsdp2 shard within EP if ndim == 1 and size() > 1 
- _get_fsdp_state -> _fsdp_param_group -> fsdp_params