# vitis-run --mode hls --csim --config ./llama2/hls_config.cfg --work_dir csim
# vitis-run --mode hls --csim --config ./llama2/mac_config.cfg --work_dir csim

vitis-run --mode hls --csim --config ./llama2/hls_config_mp.cfg --work_dir csim
