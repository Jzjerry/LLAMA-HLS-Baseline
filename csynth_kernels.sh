mkdir synth
cd synth

# mkdir quantize
# mkdir matmul
mkdir mac

# v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../configs/quantize.cfg --work_dir ./quantize
# v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../configs/matmul.cfg --work_dir ./matmul

v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../configs/mac.cfg --work_dir ./mac

cd ..