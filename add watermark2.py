import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

global_image_path="C:/Users/cao wave/Desktop/testwhite.webp"

def extract_watermark_mask(image_path, output_mask_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用边缘检测来检测可能的水印区域
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # 定义颜色阈值 (假设水印的颜色接近白色)
    lower_thresh = np.array([110, 110, 110])  # 定义白色下边界
    upper_thresh = np.array([140, 140, 140])  # 定义白色上边界
    color_mask = cv2.inRange(image, lower_thresh, upper_thresh)
    
    # 定义颜色范围（例如：红色范围）
    # 下限和上限可以根据要提取的颜色调整
    # 这里使用了红色作为示例
    lower_hsv = np.array([0, 120, 70])  # 红色下限
    upper_hsv = np.array([10, 255, 255])  # 红色上限

    # 使用 inRange 函数提取特定颜色范围内的像素
    hsvmask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    # 合并边缘检测结果和颜色阈值掩码
    combined_mask = cv2.bitwise_or(edges, color_mask)

    # 进一步处理：膨胀掩码以使水印区域更明显
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # 保存掩码图像
    # cv2.imwrite(output_mask_path, final_mask)
    cv2.imwrite(output_mask_path, final_mask)
    # 显示原图和掩码
    # plt.figure(figsize=(12, 6))

    # # 原图
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')
    # plt.axis('off')

    # # 掩码图
    # plt.subplot(1, 2, 2)
    # # plt.imshow(final_mask, cmap='gray')
    # plt.imshow(color_mask, cmap='gray')
    # plt.title('Extracted Watermark Mask')
    # plt.axis('off')

    # plt.show()


def dynamic_font_size(image_width, base_size):
    # 根据图片宽度调整字体大小
    return int(base_size * (image_width / 800))  # 假设基准宽度为 800 像素
def dynamic_font_size_image_height(height, base_size):
    # 根据图片高度调整字体大小
    return int(base_size * (height / 550))  # 假设基准宽度为 550 像素
def create_black_background(image_path, output_path):
    # 读取原始图像，获取其尺寸
    original_image = Image.open(image_path)
    width, height = original_image.size

    # 创建一个与原图像尺寸相同的全黑底图
    black_background = Image.new("RGB", (width, height), (0, 0, 0))

    # 保存全黑底图
    black_background.save(output_path)

    # 返回全黑底图
    return black_background
def dynamic_line_spacing_width(image_width, base_spacing):
    # 根据图片宽度动态调整行间距
    return int(base_spacing * (image_width / 800))  # 假设基准宽度为 800 像素
def dynamic_line_spacing_height(image_height, base_spacing):
    # 根据图片高度动态调整行间距
    return int(base_spacing * (image_height / 550))  # 假设基准宽度为 550 像素
def convert_to_mask(image_with_alpha):
    # 确保图像是 RGBA 格式 (包含 Alpha 通道)
    if image_with_alpha.mode != 'RGBA':
        raise ValueError("The image does not have an alpha channel.")
    
    # 提取 alpha 通道
    alpha_channel = image_with_alpha.split()[3]  # 获取 RGBA 的 alpha 通道

    # 创建一个全黑的掩码图，将透明部分设为黑色，不透明部分设为白色
    mask = Image.new("L", image_with_alpha.size, 0)  # 创建单通道灰度图
    mask.paste(255, (0, 0) + image_with_alpha.size, mask=alpha_channel)  # 将不透明部分设置为白色

    return mask


def creat_transparent_watermark_layer(image_path, output_path, text1, text2, text3, font_path, base_size1, base_size2, base_size3, color=(255, 255, 255), transparency=0.5):
    # 检查字体文件是否存在
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")

    # 读取图像 (OpenCV 读取，格式为 BGR)
    image = cv2.imread(image_path)
    
    # 获取图像尺寸
    height, width, _ = image.shape
    original_image = Image.open(image_path)
    width, height = original_image.size
    # if(width>height):
    #     text1='欢迎加入kolunite社区',                           # 第一行水印文本 (中文)
    # else:
    #     text1='BTC-win社区', 
    
    # 动态调整字体大小
    if(width>height):
    
         font_size1 = dynamic_font_size(width, base_size1)
         font_size2 = dynamic_font_size(width, base_size2)
         font_size3 = dynamic_font_size(width, base_size3)
    else:
         font_size1 = dynamic_font_size_image_height(height, base_size1)
         font_size2 = dynamic_font_size_image_height(height, base_size2)
         font_size3 = dynamic_font_size_image_height(height, base_size3)
    

    
    # font_size1 = dynamic_font_size(width, base_size1)
    # font_size2 = dynamic_font_size(width, base_size2)
    # font_size3 = dynamic_font_size(width, base_size3)
    # 将 OpenCV 图像转换为 PIL 图像 (BGR 转 RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 创建一个与原图像尺寸相同的全黑底图
    black_background = Image.new("RGBA", (width, height), (255, 255, 255, 0))
  
    # 创建一个与原始图像大小相同的透明图层
    # txt_layer = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
    
    # 创建绘图对象
    #draw = ImageDraw.Draw(txt_layer)
    draw2 = ImageDraw.Draw(black_background)
    # 加载字体文件
    font1 = ImageFont.truetype(font_path, font_size1)
    font2 = ImageFont.truetype(font_path, font_size2)
    font3 = ImageFont.truetype(font_path, font_size3)
    
    # 计算第一行文本的尺寸
    text_bbox1 = draw2.textbbox((0, 0), text1, font=font1)
    text_width1 = text_bbox1[2] - text_bbox1[0]
    text_height1 = text_bbox1[3] - text_bbox1[1]

    # 处理第二行文本换行
    text2_lines = text2.split('\n')
    text_height2_total = 0
    text_width2 = 0
    for line in text2_lines:
        text_bbox2 = draw2.textbbox((0, 0), line, font=font2)
        text_height2_total += text_bbox2[3] - text_bbox2[1] + 19  # 行间距
        text_width2 = max(text_width2, text_bbox2[2] - text_bbox2[0])

    # 计算第三行文本的尺寸
    text_bbox3 = draw2.textbbox((0, 0), text3, font=font3)
    text_width3 = text_bbox3[2] - text_bbox3[0]
    text_height3 = text_bbox3[3] - text_bbox3[1]
    
    # 计算文本在图像中央的位置（考虑行间距）
    # total_height = text_height1 + text_height2_total + text_height3 + 30  # 30 为总行间距
    if(width>height):
        line_spacing = dynamic_line_spacing_width(width,15)
    else:
        line_spacing = dynamic_line_spacing_height(height,12)
    # 计算文本在图像中央的位置（考虑行间距）
    if(width>height):
        text_y1 = (height// 2-text_height1//2-line_spacing) 
    else:
        text_y1 = (height// 2-text_height3//2-line_spacing)  
    outline_color = (255,0,0)
    if(width>height):
        text_y2 = text_y1 + text_height1 + line_spacing
    else:
        text_y2 = text_y1 + text_height3 + line_spacing
    # 在透明图层上绘制中文文本 (color + 透明度)
    if(width>height):
        draw2.text(((width - text_width1) // 2, text_y1), text1, font=font1, fill=color + (int(255 * transparency),))
    else:
        draw2.text(((width - text_width3) // 2, text_y1), text3, font=font3, fill=color + (int(255 * transparency),))
    # 绘制第二行文本（换行）

    if(width>height):
        for i, line in enumerate(text2_lines):
            draw2.text(((width - text_width2) // 2, text_y2 + i * (font_size2 + 5)+(9 * (width / 800))), line, font=font2, fill=color + (int(255 * transparency),))
    else:
        for i, line in enumerate(text2_lines):
            draw2.text(((width - text_width2) // 2, text_y2 + i * (font_size2 + 5)+(6 * (height / 550))), line, font=font2, fill=color + (int(255 * transparency),))
    # text_y3 = text_y2 + text_height2_total + 10
    # draw.text(((width - text_width3) // 2, text_y3), text3, font=font3, fill=color + (int(255 * transparency),))

    # 将透明图层与原图像合并
    # watermarked_image = Image.alpha_composite(image_pil.convert('RGBA'), txt_layer)
    # watermarked_image = Image.alpha_composite(black_background, txt_layer)
    # mask_image=convert_to_mask(watermarked_image)
    mask_image=convert_to_mask(black_background)
    # 保存掩码图
    mask_image_path = 'C:/Users/cao wave/Pictures/mask_image.png'
    mask_image.save(mask_image_path)
      # 保存全黑底图
    # black_background.save('C:/Users/cao wave/Pictures/result2.png')
    # 输出掩码图保存路径
    # print("Mask image saved at:", mask_image_path)
    # 将 PIL 图像转换回 OpenCV 格式 (RGBA 转 BGR)
    # final_image = cv2.cvtColor(np.array(watermarked_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # 显示结果图像
    # cv2.imshow('Watermarked Image', final_image)
    # final_image_path = 'C:/Users/cao wave/Pictures/Watermarked Image.png'
    # watermarked_image.save(final_image_path)
    # # 保存带水印的图像
    # cv2.imwrite(output_path, final_image)

    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_transparent_watermark(image_path, output_path, text1, text2, text3, font_path, base_size1, base_size2, base_size3, color=(255, 255, 255), transparency=0.5):
    # 检查字体文件是否存在
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")

    # 读取图像 (OpenCV 读取，格式为 BGR)
    image = cv2.imread(image_path)
    
    # 获取图像尺寸
    height, width, _ = image.shape
    original_image = Image.open(image_path)
    width, height = original_image.size
    # if(width>height):
    #     text1='欢迎加入kolunite社区',                           # 第一行水印文本 (中文)
    # else:
    #     text1='BTC-win社区', 
    
    # 动态调整字体大小
    if(width>height):
    
         font_size1 = dynamic_font_size(width, base_size1)
         font_size2 = dynamic_font_size(width, base_size2)
         font_size3 = dynamic_font_size(width, base_size3)
    else:
         font_size1 = dynamic_font_size_image_height(height, base_size1)
         font_size2 = dynamic_font_size_image_height(height, base_size2)
         font_size3 = dynamic_font_size_image_height(height, base_size3)
       # 计算平均颜色
    r, g, b = 0, 0, 0
    for pixel in image:
        r += pixel[0]
        g += pixel[1]
        b += pixel[2]

    total_pixels = len(image)
    avg_color = (r // total_pixels, g // total_pixels, b // total_pixels)

    # font_size1 = dynamic_font_size(width, base_size1)
    # font_size2 = dynamic_font_size(width, base_size2)
    # font_size3 = dynamic_font_size(width, base_size3)
    # 将 OpenCV 图像转换为 PIL 图像 (BGR 转 RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 创建一个与原图像尺寸相同的全黑底图
    # black_background = Image.new("RGBA", (width, height), (0, 0, 0, 255))
  
    # 创建一个与原始图像大小相同的透明图层
    txt_layer = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
    
    # 创建绘图对象
    draw = ImageDraw.Draw(txt_layer)
    #draw2 = ImageDraw.Draw(black_background)
    # 加载字体文件
    font1 = ImageFont.truetype(font_path, font_size1)
    font2 = ImageFont.truetype(font_path, font_size2)
    font3 = ImageFont.truetype(font_path, font_size3)
    
    # 计算第一行文本的尺寸
    text_bbox1 = draw.textbbox((0, 0), text1, font=font1)
    text_width1 = text_bbox1[2] - text_bbox1[0]
    text_height1 = text_bbox1[3] - text_bbox1[1]

    # 处理第二行文本换行
    text2_lines = text2.split('\n')
    text_height2_total = 0
    text_width2 = 0
    for line in text2_lines:
        text_bbox2 = draw.textbbox((0, 0), line, font=font2)
        text_height2_total += text_bbox2[3] - text_bbox2[1] + 19  # 行间距
        text_width2 = max(text_width2, text_bbox2[2] - text_bbox2[0])

    # 计算第三行文本的尺寸
    text_bbox3 = draw.textbbox((0, 0), text3, font=font3)
    text_width3 = text_bbox3[2] - text_bbox3[0]
    text_height3 = text_bbox3[3] - text_bbox3[1]

    # 计算文本在图像中央的位置（考虑行间距）
    # total_height = text_height1 + text_height2_total + text_height3 + 30  # 30 为总行间距
    if(width>height):
        line_spacing = dynamic_line_spacing_width(width,15)
    else:
        line_spacing = dynamic_line_spacing_height(height,12)
    total_height = text_height1 + text_height2_total + line_spacing

    if(width>height):
        text_y1 = (height// 2-text_height1//2-line_spacing) 
    else:
        text_y1 = (height// 2-text_height3//2-line_spacing)  
    # text_y1  = height//2
    # text_y2 = text_y1 + text_height1 + 40
    if(width>height):
        text_y2 = text_y1 + text_height1 + line_spacing
    else:
        text_y2 = text_y1 + text_height3 + line_spacing
    outline_color={0,0,0} #黑色
        # 判断主色调是否为白色
    if (np.array(avg_color) > 200).all():  # 简单判断主色调是否为白色         
    # 在透明图层上绘制中文文本 (color + 透明度)
        if(width>height):
            draw.text(((width - text_width1) // 2, text_y1), text1, font=font1, fill=color + (235,),outline=outline_color)
        else:
            draw.text(((width - text_width3) // 2, text_y1), text3, font=font3, fill=color + (235,),outline=outline_color)
    else:
        if(width>height):
            draw.text(((width - text_width1) // 2, text_y1), text1, font=font1, fill=color + (int(255 * transparency),),outline=outline_color)
        else:
            draw.text(((width - text_width3) // 2, text_y1), text3, font=font3, fill=color + (int(255 * transparency),),outline=outline_color)
    # 绘制第二行文本（换行）
    if(width>height):
        for i, line in enumerate(text2_lines):
            draw.text(((width - text_width2) // 2, text_y2 + i * (font_size2 + 5)+(9 * (width / 800))), line, font=font2, fill=color + (int(255 * transparency),))
    else:
        for i, line in enumerate(text2_lines):
            draw.text(((width - text_width2) // 2, text_y2 + i * (font_size2 + 5)+(6 * (height / 550))), line, font=font2, fill=color + (int(255 * transparency),))
    # text_y3 = text_y2 + text_height2_total + 10
    # draw.text(((width - text_width3) // 2, text_y3), text3, font=font3, fill=color + (int(255 * transparency),))

    # 将透明图层与原图像合并
    watermarked_image = Image.alpha_composite(image_pil.convert('RGBA'), txt_layer)
    # watermarked_image = Image.alpha_composite(black_background, txt_layer)
    # mask_image=convert_to_mask(watermarked_image)
    mask_image=convert_to_mask(txt_layer)
    # 保存掩码图
    mask_image_path = 'C:/Users/cao wave/Pictures/mask_image1.png'
    mask_image.save(mask_image_path)
      # 保存全黑底图
    # black_background.save('C:/Users/cao wave/Pictures/result2.png')
    # 输出掩码图保存路径
    # print("Mask image saved at:", mask_image_path)
    # 将 PIL 图像转换回 OpenCV 格式 (RGBA 转 BGR)
    final_image = cv2.cvtColor(np.array(watermarked_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # 显示结果图像
    cv2.imshow('Watermarked Image', final_image)
    final_image_path = 'C:/Users/cao wave/Pictures/Watermarked Image.png'
    watermarked_image.save(final_image_path)
    # 保存带水印的图像
    cv2.imwrite(output_path, final_image)

    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def remove_watermark(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or unable to load.")
        return

    # 获取图像的尺寸
    height, width = image.shape[:2]

    # 创建掩码，水印位于正中间
    # mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # mask_image = cv2.imread('C:/Users/cao wave/Desktop/mask_image.png')C:\Users\\
    mask_image = cv2.imread('C:/Users/cao wave/Pictures/mask_image_expend.png')
    # gray_image = cv2.cvtColor(mask_image, cv2.IMREAD_GRAYSCALE)
    gray_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    # mask_array = np.array(binary_image)
    mask_array=np.where(binary_image == 255, 1, 0)
    top_left_x = (width - 800) // 2 #426
    top_left_y = (height - 160) // 2 # 260
    # mask[top_left_y:top_left_y + 160, top_left_x:top_left_x + 800] = 50
# 计算裁剪区域的起始坐标 (左上角) 和结束坐标 (右下角)
    crop_width = 1000  # 保留600像素宽
    crop_height = 400  # 保留300像素高

# 计算左上角的坐标
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

# 计算右下角的坐标
    end_x = start_x + crop_width
    end_y = start_y + crop_height

# # 裁剪掩码，只保留中间区域
#     cropped_mask = binary_image[start_y:end_y, start_x:end_x]
#     # 增加修复半径，尝试不同的修复方法
    new_mask = np.zeros_like(binary_image)
    # 将原掩码中间区域复制到新掩码上，其他部分保持为0
    new_mask[start_y:end_y, start_x:end_x] = binary_image[start_y:end_y, start_x:end_x]
    result = cv2.inpaint(image, binary_image, inpaintRadius=20, flags=cv2.INPAINT_NS)
    
    # 锐化图像
    # kernel = np.array([[0, -0.2, 0],
    #                    [-0.2, 2, -0.2],
    #                    [0, -0.2, 0]])
    # result = cv2.filter2D(result, -1, kernel)

    # 保存结果
    cv2.imwrite(output_path, result)

# 使用示例，添加带透明度的中文水印到图片中央
creat_transparent_watermark_layer(
    image_path=global_image_path,
    # image_path='C:/Users/cao wave/Desktop/output_image.jpg',
    output_path='C:/Users/cao wave/Pictures/result1.png',  # 输出图片路径
    # text1='欢迎加入BTC-win社区',                           # 第一行水印文本 (中文)
    text1='欢迎加入kolunite社区',
    text2='Telegram社区：https://t.me/kolunite_notice\nDiscord社区：https://discord.gg/kolunite',     # 第二行水印文本 (中文)
    # text2='Telegram社区：https://t.me/BTC-win_notice\nDiscord社区：https://discord.gg/BTC-win_notice',  
    # text3='Discord社区：https://discord.gg/kolunite',                             # 第三行水印文本 (中文)
    # text3='BTC-win社区',
    text3='kolunite社区',
    font_path='C:/Windows/Fonts/msyh.ttc',               # 字体路径（使用支持中文的字体）
    base_size1=62,                                        # 第一行基准字体大小
    base_size2=13,                                        # 第二行基准字体大小
    base_size3=64,                                        # 第三行基准字体大小
)
extract_watermark_mask('C:/Users/cao wave/Pictures/mask_image.png', 'C:/Users/cao wave/Pictures/mask_image_expend.png')

remove_watermark(global_image_path, 'C:/Users/cao wave/Desktop/output_image.jpg')

add_transparent_watermark(
    image_path='C:/Users/cao wave/Desktop/output_image.jpg',
    output_path='C:/Users/cao wave/Pictures/result2.png',  # 输出图片路径
    text1='欢迎加入BTC-win社区',                           # 第一行水印文本 (中文)
    # text1='欢迎加入kolunite社区',
    # text2='Telegram社区：https://t.me/kolunite_notice\nDiscord社区：https://discord.gg/kolunite',     # 第二行水印文本 (中文)
    text2='Telegram社区：https://t.me/BTC-win_notice\nDiscord社区：https://discord.gg/BTC-win_notice',  
    # text3='Discord社区：https://discord.gg/kolunite',                             # 第三行水印文本 (中文)
    text3='BTC-win社区',
    # text3='kolunite社区',
    font_path='C:/Windows/Fonts/arial.ttf',               # 字体路径（使用支持中文的字体）
    base_size1=62,                                        # 第一行基准字体大小
    base_size2=13,                                        # 第二行基准字体大小
    base_size3=62,                                        # 第三行基准字体大小
)