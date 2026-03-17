"""
MingImg: Image Generation Module

This module provides image generation and editing capabilities using
the BailingMM2 diffusion model. It supports text-to-image generation
and image editing with various aspect ratios and resolutions.

Key Features:
    - Text-to-image generation with prompt optimization
    - Image editing with segmentation mask support
    - Automatic saturation and exposure balancing
    - Multiple aspect ratio support

Usage:
    >>> img_gen = MingImg(model_path="/path/to/model", device="cuda:0")
"""

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from PIL import ImageEnhance
import torchvision.transforms as transforms


def auto_balance_saturation_exposure(pil_img: Image.Image) -> Image.Image:
    """
    Automatically detect and adjust saturation and exposure to prevent
    over-saturation and over-exposure.

    This function uses fixed internal thresholds and applies adjustments
    only when values exceed safe limits.

    Args:
        pil_img (Image.Image): Input PIL image to process.

    Returns:
        Image.Image: Processed image with balanced saturation and exposure.
    """
    # Convert to HSV color space for saturation/value analysis
    hsv_img = pil_img.convert('HSV')
    hsv_array = np.array(hsv_img)

    # Extract saturation and brightness channels
    saturation_channel = hsv_array[:, :, 1] / 255.0  # Range: 0~1
    value_channel = hsv_array[:, :, 2] / 255.0  # Range: 0~1

    # Calculate mean saturation and brightness
    mean_sat = saturation_channel.mean()
    mean_val = value_channel.mean()

    # Fixed internal thresholds
    max_sat_threshold = 0.5   # Over-saturation threshold
    max_val_threshold = 0.75   # Over-exposure threshold

    adjusted_img = pil_img

    print("sat: ", mean_sat)
    print("val: ", mean_val)

    if mean_sat > max_sat_threshold:
        # Reduce saturation to threshold ratio
        ratio = max_sat_threshold / mean_sat
        adjusted_img = ImageEnhance.Color(adjusted_img).enhance(ratio)

    if mean_val > max_val_threshold:
        # Reduce brightness to threshold ratio
        ratio = max_val_threshold / mean_val
        adjusted_img = ImageEnhance.Brightness(adjusted_img).enhance(ratio)

    return adjusted_img


def gaussian_kernel(size=5, sigma=1.0):
    x, y = torch.meshgrid(
        torch.linspace(-size / 2, size / 2, size),
        torch.linspace(-size / 2, size / 2, size),
    )
    d = x**2 + y**2
    g = torch.exp(-d / (2.0 * sigma**2))
    g = g / torch.sum(g)
    return g


def get_filter_conv(kernel_size: int = 3, sigma: float = 1.0) -> nn.Conv2d:
    """
    Create a 2D Gaussian filter convolution layer.

    Args:
        kernel_size (int): Size of the Gaussian kernel. Defaults to 3.
        sigma (float): Standard deviation of the Gaussian. Defaults to 1.0.

    Returns:
        nn.Conv2d: Convolution layer with fixed Gaussian weights.
    """
    gaussian_kernel_2d = gaussian_kernel(kernel_size, sigma)
    gaussian_kernel_3d = gaussian_kernel_2d.expand(3, kernel_size, kernel_size)
    conv_layer = nn.Conv2d(
        in_channels=3, out_channels=3, kernel_size=kernel_size, groups=3, bias=False
    )

    conv_layer.weight.data = gaussian_kernel_3d.unsqueeze(1)
    conv_layer.weight.requires_grad = False

    return conv_layer


def remove_noise_opencv(binary_image: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Remove noise from a binary image using median blur.

    Args:
        binary_image (np.ndarray): Input binary image (0-1 or 0-255 range).
        radius (int): Kernel size for median blur. Defaults to 5.

    Returns:
        np.ndarray: Denoised binary image.
    """
    if binary_image.max() <= 1:
        binary_image = (binary_image * 255).astype(np.uint8)
    else:
        binary_image = binary_image.astype(np.uint8)

    denoised = cv2.medianBlur(binary_image, radius)

    return denoised


def get_mask(ref_img: Image.Image, pre_img: Image.Image,
             kernel_size: int = 3, sigma: float = 1.0,
             radius: int = 5, device: str = "cpu") -> np.ndarray:
    """
    Generate a segmentation mask by comparing two images.

    This function computes the difference between a reference image and a
    previous frame to identify changed regions, useful for video editing
    and object segmentation.

    Args:
        ref_img (Image.Image): Reference image (current frame).
        pre_img (Image.Image): Previous image for comparison.
        kernel_size (int): Gaussian kernel size for smoothing. Defaults to 3.
        sigma (float): Gaussian sigma for smoothing. Defaults to 1.0.
        radius (int): Denoising radius for morphological operations. Defaults to 5.
        device (str): Device for tensor operations. Defaults to "cpu".

    Returns:
        np.ndarray: Binary mask indicating changed regions.
    """
    transform_img = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    conv_layer = get_filter_conv(kernel_size, sigma)

    image1_tensor = transform_img(ref_img).unsqueeze(0)  
    image2_tensor = transform_img(pre_img).unsqueeze(0)  

    image1_tensor = image1_tensor.to(device)
    image2_tensor = image2_tensor.to(device)
    conv_layer = conv_layer.to(device)

    filtered_image1 = conv_layer(image1_tensor)
    filtered_image2 = conv_layer(image2_tensor)

    filtered_image2 = (filtered_image2 - 0.5 * filtered_image1) * 2

    diff = filtered_image1 - filtered_image2
    abs_diff = torch.abs(diff)
    abs_diff = conv_layer(abs_diff)
    # thresh = float(os.getenv('SEG_THRESH', 0.15)) # 0.1
    # radius = int(os.getenv('SEG_DENOISE', 5))
    thresh = 0.15
    radius = 11
    abs_diff = (abs_diff.mean(dim=1)[0] > thresh).float()
    res = remove_noise_opencv(abs_diff.numpy(), radius)

    return res


def get_cutout(ref_img: Image.Image, pre_mask: np.ndarray) -> Image.Image:
    """
    Create a cutout image with transparent background using a mask.

    Args:
        ref_img (Image.Image): Source image to cut out from.
        pre_mask (np.ndarray): Binary mask for the cutout region.

    Returns:
        Image.Image: RGBA image with transparent background outside mask.
    """
    rgba = ref_img.convert("RGBA")
    alpha = Image.fromarray(pre_mask.astype(np.uint8), mode="L").resize(ref_img.size)
    rgba.putalpha(alpha)

    return rgba


ratio_extraction_fromat = """你是一名专业的AI图像提示词处理专家，请依据以下规则处理输入：

1.提取用户提示词中关于输出图像长宽比例的信息，如捕捉16:9, 1比1, 等信息
2.输出必须为一行纯文本, 即为长宽比信息（格式为 number:number），不要包含任何前缀后缀；
3.如果没有发现长宽比例的信息，输出 "None"

输出示例：
输入: 中国小女孩自拍，衣服上写着美丽二字, 按照16:10比例进行生成
输出: 16:10

输入: In a 1:1 ratio, generate a man.
输出: 1:1

输入: generate a man, in a 1920x1080
输出: 1920:1080

输入: draw a beautiful girl
输出: None


现在处理以下输入：{}"""


rewrite_fromat = """你是一位专业的文生图提示词工程师，擅长将用户简短的图像描述转化为细节丰富、视觉准确、适合主流图像模型生成的自然语言提示。你的任务是在忠实还原用户核心意图的前提下，自动补全必要视觉信息，并输出一行可用于直接生成高质量图像的描述。

请严格遵循以下原则：

🔹 输入语言决定输出语言  
- 若用户输入为**中文**，则输出为**中文单句描述**
- 若用户输入为**英文**，则输出为**英文单句描述**
- 不要翻译、不要混用、不要添加解释性语句

🔹 核心意图优先  
始终将**图像的主要用途或载体类型**置于描述最前方，确保模型优先理解画面功能属性。例如：
- “PPT background with a woman” 先写 "PPT background design"
- “a running man” 转换为静态后写为 "a man standing on a running track"
- “手机锁屏” 写为 “手机屏幕界面”

避免将人物外貌、环境细节等次要信息前置导致主题偏移。

🔹 绝对静态化处理  
禁止使用任何暗示运动或趋势的姿态词汇（如“正在”“摆动”“抬起”）。将动作类描述转换为静止状态：
- “打篮球” -> “抱着篮球站在篮球场”
- “踢足球” -> “脚踩足球站在足球场”
- “跳舞” -> “站在舞台上”

仅保留主体与对象的共现关系。

🔹 零文本增生原则（全局禁止）  
除非用户**明确使用“写着”“印着”“显示”“刻着”“挂着”“贴着”“涂鸦”“屏幕上显示”等动词 + 具体内容**，否则不得在图像中添加任何形式的文字元素，包括：
- 品牌标识（如 `"LV"`、`"Nike"`）
- 店名招牌（如 `"渔火"`）
- 屏幕显示内容
- 服装印花文字

即使该品牌通常带有标志性文字（如 LV 包），也不予渲染。

🔹 文字标注规范（仅当显式提及）  
若用户明确描述某物表面有可读文字（如 “the sign says 'Open'” 或 “T恤上印着Hello”），则提取该文字并用英文双引号 " " 包裹。其他情况一律不添加。

🔹 智能风格推断（自动适配最优风格）  
若未指定艺术风格，则根据主体类型自动匹配最合理风格：
- 真实人物、静物、自然景观 -> 写实摄影风格 / photorealistic
- 拟人化动物、卡通角色 -> 卡通或插画风格 / cartoon, illustration
- 虚构生物、未来世界 -> 数字艺术 / concept art, digital painting
- 抽象概念 -> 超现实主义 / surrealism
- PPT/网页/界面 -> 扁平设计 / flat design, modern UI

🔹 视觉细节增强  
在不偏离主次的前提下，补充以下维度信息：
- 主体外观：颜色、服装、发型、物品样式
- 材质纹理：布料、金属、玻璃、皮革光泽
- 光照氛围：自然光、室内灯、晴天/黄昏
- 时间季节：清晨、秋日、雪景等典型情境
- 环境背景：室内外场景、天气、空间层次
- 构图建议：背景虚化、中心对称、留白区域

🔹 多余图片信息去除
若用户输入中包含生成图像的比例，或者分辨率，则在输出中删除相关表述，例如“按照1:1生成”、“16:9”、“1920x1080”

🔸 输出格式要求  
- 仅输出一行自然语言描述，作为完整句子呈现  
- 不换行、不编号、不加标题、不解释、不推荐参数  
- 语言简洁准确，避免抽象比喻、情感渲染或文学化修辞  
- 关键信息前置：**用途/载体, 场景, 主体, 风格**  
- 禁用所有动态趋势词（如“正在”“即将”“欲”）  
- 输出应符合文生图模型常用表达方式（关键词适度密集，语法自然）

📌 示例参考（中英混合）：
输入：一个女性背景的ppt, 16:9  
输出：PPT背景设计，浅蓝色渐变底纹搭配简约线条图案，右侧有一位优雅站立的女性剪影，整体为现代商务风格，扁平化视觉效果

输入：a girl dancing on the beach  
输出：a girl standing on a sandy beach, facing the ocean, wearing a white dress, seagulls flying in the distance, soft sunlight, realistic photography style

输入：海滨餐馆  
输出：一家临海的小型木结构餐馆坐落在岩石岸边，大落地窗面向大海，屋顶有茅草遮阳顶，门口摆放着几张木质桌椅，背景是波光粼粼的海面和晚霞，整体为写实摄影风格

输入：the restaurant, with a wooden sign hanging above the door that says Hello 渔火
输出：a small wooden seaside restaurant with a thatched roof, large windows facing the ocean, and a hand-carved wooden sign above the entrance displaying the Chinese characters "Hello 渔火", gentle waves in the background, golden hour lighting, photorealistic style

输入：an lv bag  
输出：a brown monogram pattern handbag placed on a light gray marble surface, fine leather texture with visible weave, sturdy handles, soft ambient lighting, photorealistic product photography style

输入：一个男人
输出：一个男人站在城市街道旁，身穿深蓝色风衣，黑色短发，面容清晰，背景为虚化的行人和建筑，自然光照射，写实摄影风格

输入：哆啦A梦
输出：卡通风格的哆啦A梦站立在明亮的室内场景中，圆润的蓝色机身，白色腹部，红色项圈配黄色铃铛，大眼睛直视前方，双手自然下垂，背景为浅色木纹地板与柔和的暖光照明，整体为彩色插画风格

输入：黑板报上面写着蚂蚁Ming-Omni
输出：黑板报上用白色粉笔字写着"蚂蚁Ming-Omni"，深灰色木质边框的黑板置于教室墙面上，周围贴有彩色剪纸和学生绘画作品，阳光从左侧窗户斜射入内，形成柔和光斑与粉尘光束，粉笔字迹清晰带有轻微阴影，背景为浅黄色旧墙，整体为写实摄影风格


现在，请优化以下输入：  
{}"""


rewrite_edit_fromat = """你是一位专业的图像编辑提示词工程师，擅长将用户的编辑图像描述转化为固定的格式。你的任务是在忠实还原用户核心意图的前提下，自动转换格式与语言，并输出一行可用于直接生成图像编辑任务的指令。

请严格遵循以下原则：

一. 一般指令输出英文, 输出必须为一行纯文本, 即为优化的结果，不要包含任何前缀后缀
中文或中英文混合，要翻译成英文的指令：
- "将背景换成海滩" 应改为 "Change the background to a beach"
- "把eyeglasses去掉" 应改为 "Remove the eyeglasses"
- "举手" 应改为 "Raise the hand up"
不要将用户想在画面中改的字进行翻译：
- "把封面上的\"你好\"改成\"今天吃了吗\"" 应改为 "Change the text "你好" on the cover to "今天吃了吗""
- "写上一句话orange真好吃", 应改为 "Add the text "orange真好吃""

二. 有关 segmentation 或者分割的指令，按照固定格式"Given the following instructions: [target]; please perform referring segmentation on this image with [color] mask."输出
- [color]默认使用"green" , 如果已经指定掩码颜色，则使用用户指定的颜色
- "Separate the little girl on the left" 应改为 "Given the following instructions: the little girl on the left; please perform referring segmentation on this image with green mask."
- "把塔分割出来" 应改为 "Given the following instructions: tower; please perform referring segmentation on this image with green mask."
- "用橙色把塔分割出来" 应改为 "Given the following instructions: tower; please perform referring segmentation on this image with orange mask."
- "perform segmentation on the tiger" 应改为 "Given the following instructions: tiger; please perform referring segmentation on this image with green mask."
- "please segment the cats" 应改为 "Given the following instructions: cats; please perform referring segmentation on this image with green mask."
- "请扣出图中的塔" 应改为 "Given the following instructions: tower; please perform referring segmentation on this image with green mask."
- "把右边的男人抠出来" 应改为 "Given the following instructions: the man on the right; please perform referring segmentation on this image with green mask."
- "分割出戴帽子的女人" 应改为 "Given the following instructions: woman wearing a hat; please perform referring segmentation on this image with green mask."
- "把站起来的小狗分割出来" 应改为 "Given the following instructions: the puppy standing up; please perform referring segmentation on this image with green mask."
- "把"hello 世界"抠出来" 应改为 "Given the following instructions: the text hello 世界; please perform referring segmentation on this image with green mask."


三. 探测是否有文字渲染需求  
如果用户指令中存在用户想要渲染的文字内容时将需要生成的文字改为使用""(英文双引号)括起来，比如在描述文本内容、书本封面、海报、广告牌、黑板、印刷等场景，对于英文提示词重点看是否有 "text" 或者 "word" 的暗示：
- Change the word unification in the book to promote
- "把封面上的\"hello\"改成\"你好\"" 应改为 "Change the text "hello" on the cover to "你好"".
- "添加文本北京欢迎您", 应改为 "Add the text "北京欢迎您"".
- "删除题目中"坏孩子"", 应改为 "Remove the text "坏孩子"".
如果用户没有明确的文字生成意愿, 则不要使用双引号进行改写，也不要额外增加渲染文字内容：
- "将背景换成 Arc de Triomphe" 应改为 "Change the background to Arc de Triomphe"
- "add a cat on the rock" 应改为 "add a cat on the rock"
特别注意，有关 segmentation 或者分割的指令，应去掉输出指令中的所有的双引号(“”，"")
- "把“你好”抠出来", 应改为 "Given the following instructions: text 你好; please perform referring segmentation on this image with green mask."
- "抠出标题中的"hello 世界"", 应改为 "Given the following instructions: the text hello 世界; please perform referring segmentation on this image with green mask."
- "please segment the text "great again"", 应改为 "Given the following instructions: the text great again; please perform referring segmentation on this image with green mask."
- "分割出 the word happy", 应改为 "Given the following instructions: the word happy; please perform referring segmentation on this image with green mask."

四. 删除关于画面比例的表述，如1:1, 16:9这种
- "生成证件照，16：9" 应改为 "Translate into a standard ID photo"


现在处理以下输入：
{}"""

image_gen_indent_format = """你是一个意图判断器。

你的任务是：判断用户的输入是否满足以下任一条件：
1. 描述为具象的实体（场景、人物、动物、景色、物体、颜色、细节、要被渲染的文字任意一项即可）。
2. 表达了明确的生成图片的意图（如“画一下”“生成图片”“给我一张…”）。

满足任一条件 → 输出 "yes"  
否则 → 输出 "no"  

输出规则：
- 只能输出小写的yes或no。
- 不要解释，不要输出其它符号或文字。

以下是示例：
用户输入: "画一只蓝色的鲸在天空中飞"
输出: yes

用户输入: "帮我生成一张东京街头夜景的照片"
输出: yes

用户输入: "在森林里，一只狐狸坐在溪边"
输出: yes

用户输入: "故宫"
输出: yes

用户输入: "戴眼镜"
输出: yes

用户输入: "一行字 hello world"
输出: yes

用户输入: "女孩"
输出: yes

用户输入: "generate a boy"
输出: yes

用户输入: "a boy"
输出: yes

用户输入: "draw people"
输出: yes

用户输入: "清晨薄雾笼罩的青翠山间小径，白发少年逆光奔跑，动态模糊捕捉疾驰瞬间；少年身穿白色运动服，手持一本摊开的书，柔光侧照突显发丝细节，远景层叠山峦与晨曦光晕，8K超清写实风格，浅景深突出主体"
输出: yes

用户输入: "goodbye"
输出: no

用户输入: "Hi"
输出: no

用户输入: "What can you do for me"
输出: no

用户输入: "你可以做什么？"
输出: no

用户输入: "我们公司在北京"
输出: no

用户输入: "描述你昨天的天气"
输出: no

用户输入: "你好"
输出: no

用户输入: "一只老虎"
输出: yes

用户输入: "海滩"
输出: yes

现在，请判断以下用户输入：  
{}"""

DEFAUL_PROMPT_FOR_NO_INTENT = "纯色背景， 写着\"Please input prompt for image generation\""


class MingImg(object):
    """
    Image generation and editing module using BailingMM2 diffusion model.

    This class provides high-level interfaces for text-to-image generation
    and image editing operations with support for various aspect ratios
    and resolutions.

    Attributes:
        model_diffusion: The diffusion model for image generation.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        """
        Initialize the MingImg module.

        Args:
            model_path (str): Path to the model directory containing the diffusion model.
            device (str): GPU device to load the model on. Defaults to "cuda:0".
            **kwargs: Additional arguments for model initialization.
        """
        os.environ["IMAGE_GEN_MODE"] = "None"
        from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

        current_device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        model_diffusion = (
            BailingMM2NativeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
                attn_implementation="flash_attention_2",
                load_image_gen=True,
                low_cpu_mem_usage=True,  # Minimize CPU memory during loading
                load_vlm=False,  # No VLM, only diffusion
            )
            .to(device)
            .to(torch.bfloat16)
        )
        torch.cuda.set_device(current_device)
        self.model_diffusion = model_diffusion
