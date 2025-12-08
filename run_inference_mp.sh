g++ -g -std=c++17 -Wall -O0 ./host_mp/llama2.cpp \
    -o ./host_mp/llama2.exe \
    -I$XILINX_XRT/include/ -L$XILINX_XRT/lib -lxrt_coreutil -pthread


./host_mp/llama2.exe \
    ./weights_Q4.bin \
    -z ./tokenizer.bin \
    -n 256 \
    -i "Once" \
    -k ./hw_end_mp/forward.xclbin