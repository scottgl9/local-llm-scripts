FROM avarok/dgx-vllm-nvfp4-kernel:v23

# Install tools needed for the MTP fix script
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Apply upstream MTP NVFP4 exclusion fix
RUN wget -q https://raw.githubusercontent.com/Avarok-Cybersecurity/dgx-vllm/main/fix_mtp_nvfp4_exclusion.py && \
    python3 $(find / -name fix_mtp_nvfp4_exclusion.py) && \
    echo "Applied fix_mtp_nvfp4_exclusion.py"

# Copy patched vLLM files (tool-call bug fixes — see docs/TOOL_CALL_BUGS.md)
COPY patches/vllm/v23/entrypoints/chat_utils.py /app/vllm/vllm/entrypoints/chat_utils_orig.py
COPY patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.py /app/vllm/vllm/tool_parsers/qwen3coder_tool_parser_orig.py
COPY patches/vllm/v23/entrypoints/chat_utils.patch /tmp/chat_utils.patch
COPY patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.patch /tmp/qwen3coder_tool_parser.patch

RUN patch -o /app/vllm/vllm/entrypoints/chat_utils.py \
        /app/vllm/vllm/entrypoints/chat_utils_orig.py /tmp/chat_utils.patch && \
    patch -o /app/vllm/vllm/tool_parsers/qwen3coder_tool_parser.py \
        /app/vllm/vllm/tool_parsers/qwen3coder_tool_parser_orig.py /tmp/qwen3coder_tool_parser.patch && \
    find /app/vllm/vllm -name '*.pyc' -delete && \
    rm /app/vllm/vllm/entrypoints/chat_utils_orig.py \
       /app/vllm/vllm/tool_parsers/qwen3coder_tool_parser_orig.py \
       /tmp/chat_utils.patch /tmp/qwen3coder_tool_parser.patch && \
    echo "Tool-call patches applied"
