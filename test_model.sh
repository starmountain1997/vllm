#!/bin/bash

# --- 请修改下面两行 ---
IMAGE_PATH="/home/guozr/test.jpg"
SERVER_URL="http://localhost:8000/v1/chat/completions" # 如果你的IP和端口不同，请修改这里
# ---------------------

# 判断图片类型
MIME_TYPE=$(file --mime-type -b "$IMAGE_PATH")
if [[ ! "$MIME_TYPE" =~ ^image/ ]]; then
    echo "错误: 文件 '$IMAGE_PATH' 不是一个有效的图片。"
    exit 1
fi

# 将图片编码为 Base64，并确保没有换行符
# 这是针对 "Non-base64 digit found" 错误的修正
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: 使用 -b 0 参数禁用换行符
    BASE64_IMAGE=$(base64 -i "$IMAGE_PATH" -b 0)
else
    # Linux: 使用 -w 0 参数禁用换行符
    BASE64_IMAGE=$(base64 -w 0 "$IMAGE_PATH")
fi

# 构建 JSON payload
PAYLOAD=$(cat <<EOF
{
    "model": "Qwen3VLMoeForConditionalGeneration",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请描述这张本地图片的内容。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:${MIME_TYPE};base64,${BASE64_IMAGE}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 1024,
    "temperature": 0
}
EOF
)

# 发送 curl 请求
curl "$SERVER_URL" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD"
