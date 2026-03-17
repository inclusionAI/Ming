import copy
from ming_sdk.ming_utils import ThreadSafeCache

"""

usage = {
           "prompt_tokens": 0,
           "generated_tokens": 0,
           "total_tokens": 0,
           "completion_tokens_details": {
               "text_tokens": 0,
               "audio_tokens": 0,
               "image_tokens": 0,
           },
           "prompt_tokens_details": {
               "audio_tokens": 0,
               "image_tokens": 0,
               "video_tokens": 0,
               "text_tokens": 0,
           },
       }

"""

usage_useless_value = [0, None]
extra_key = ["finish_reason"]

def remove_zero_values(usage):
    def check_useless(item):
        for i in usage_useless_value:
            if item == i:
                return True
        return False
    def remove_extra_key(usage):
        for i in extra_key:
            if i in usage:
                del usage[i]

    if usage is None:
        return {}
    new_usage = copy.deepcopy(usage)
    nested_keys = [k for k, v in new_usage.items() if isinstance(v, dict)]
    for key in nested_keys:
        keys_to_remove = [k for k, v in new_usage[key].items() if check_useless(v)]
        for k in keys_to_remove:
            del new_usage[key][k]
        if not new_usage[key]:
            del new_usage[key]

    keys_to_remove = [k for k, v in new_usage.items() if check_useless(v)]
    for key in keys_to_remove:
        del new_usage[key]

    remove_extra_key(new_usage)

    return new_usage


class Usage(object):

    def __init__(self):
        self.cache = ThreadSafeCache(max_size=500, default_ttl=1800)

    def create_usage(self, output):
        usage = {
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {
                "text_tokens": 0,
                "audio_tokens": 0,
                "image_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "image_tokens": 0,
                "video_tokens": 0,
                "text_tokens": 0,
            },
            "finish_reason": None
        }
        prompt_tokens = len(output.prompt_token_ids)
        generated_tokens = len(output.outputs[0].token_ids)
        usage["prompt_tokens"] = prompt_tokens
        usage["generated_tokens"] = generated_tokens
        usage["total_tokens"] = prompt_tokens + usage["generated_tokens"]
        usage["finish_reason"] = output.outputs[0].finish_reason
        return usage

    def create_usage_by_requests_id(self, output, request_id):
        usage = {
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {
                "text_tokens": 0,
                "audio_tokens": 0,
                "image_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "image_tokens": 0,
                "video_tokens": 0,
                "text_tokens": 0,
            },
            "finish_reason": None
        }
        prompt_tokens = len(output.prompt_token_ids)
        generated_tokens = len(output.outputs[0].token_ids)
        usage["prompt_tokens"] = prompt_tokens
        usage["generated_tokens"] = generated_tokens
        usage["total_tokens"] = prompt_tokens + usage["generated_tokens"]
        usage["finish_reason"] = output.outputs[0].finish_reason
        self.cache.set(f"{request_id}", usage, 1800)
        return usage

    def get_stream_usage_by_request_id(self, request_id: int = 0):
        usage = {
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {
                "text_tokens": 0,
                "audio_tokens": 0,
                "image_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "image_tokens": 0,
                "video_tokens": 0,
                "text_tokens": 0,
            },
            "finish_reason": None
        }
        usage_res = self.cache.get(f"{request_id}")
        if usage_res == None:
            return usage
        return copy.deepcopy(usage_res)

    @staticmethod
    def update_audio_usage_by_duration(usage, duration):
        if usage is None:
            usage = {
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {
                    "text_tokens": 0,
                    "audio_tokens": 0,
                    "image_tokens": 0,
                },
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "image_tokens": 0,
                    "video_tokens": 0,
                    "text_tokens": 0,
                },
                "finish_reason": None
            }
        audio_tokens = int(duration * 50)
        text_tokens = usage["generated_tokens"]

        usage["generated_tokens"] += audio_tokens
        usage["total_tokens"] += audio_tokens
        usage["completion_tokens_details"]["audio_tokens"] = audio_tokens
        usage["completion_tokens_details"]["text_tokens"] = text_tokens
        return usage

    @staticmethod
    def update_usage_by_processor(usage, text_token_count=None, image_token_count=None, video_token_count=None, audio_token_count=None):
        if usage is None:
            usage = {
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {
                    "text_tokens": 0,
                    "audio_tokens": 0,
                    "image_tokens": 0,
                },
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "image_tokens": 0,
                    "video_tokens": 0,
                    "text_tokens": 0,
                },
                "finish_reason": None
            }
        input_tokens_sum = 0
        if image_token_count and isinstance(image_token_count, int):
            usage["prompt_tokens_details"]["image_tokens"] = image_token_count
            input_tokens_sum += image_token_count
        if video_token_count and isinstance(video_token_count, int):
            usage["prompt_tokens_details"]["video_tokens"] = video_token_count
            input_tokens_sum += video_token_count
        if audio_token_count and isinstance(audio_token_count, int):
            usage["prompt_tokens_details"]["audio_tokens"] = audio_token_count
            input_tokens_sum += audio_token_count
        if text_token_count and isinstance(text_token_count, int):
            usage["prompt_tokens_details"]["text_tokens"] = text_token_count
            input_tokens_sum += text_token_count
        if input_tokens_sum > 0:
            usage["prompt_tokens"] = input_tokens_sum
        usage["total_tokens"] = usage["prompt_tokens"] + usage["generated_tokens"]
        return usage

    @staticmethod
    def update_image_usage_by_length(usage, image_gen_highres):
        if usage is None:
            usage = {
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {
                    "text_tokens": 0,
                    "audio_tokens": 0,
                    "image_tokens": 0,
                },
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "image_tokens": 0,
                    "video_tokens": 0,
                    "text_tokens": 0,
                },
                "finish_reason": None
            }
        image_tokens = int(image_gen_highres * image_gen_highres / 16 / 16)
        text_tokens = usage["generated_tokens"]
        usage["generated_tokens"] += image_tokens
        usage["total_tokens"] += image_tokens
        usage["completion_tokens_details"]["image_tokens"] = image_tokens
        usage["completion_tokens_details"]["text_tokens"] = text_tokens
        return usage

    @staticmethod
    def create_usage_default(prompt_tokens=0):
        usage = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": 0,
            "total_tokens": prompt_tokens,
            "completion_tokens_details": {
                "text_tokens": 0,
                "audio_tokens": 0,
                "image_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "image_tokens": 0,
                "video_tokens": 0,
                "text_tokens": 0,
            },
            "finish_reason": None
        }
        return usage
