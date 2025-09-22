mkdir synth
cd synth

# mkdir quantize
mkdir matmul

# v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../configs/quantize.cfg --work_dir ./quantize
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../configs/matmul.cfg --work_dir ./matmul