import os
from PIL import Image, ImageFile

# 允许处理可能被截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop_white_borders(image_path):
    """
    裁剪图片的白色边框 (此函数与前一版本相同，确保其健壮性)
    参数:
        image_path: 图片文件路径
    返回:
        裁剪后的 PIL Image 对象, 或在出错时返回 None,
        或在无需裁剪/边界无效时返回原始图像的副本.
    """
    try:
        with Image.open(image_path) as img:
            original_img_for_return = img.copy()
            # 模式转换
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            elif img.mode in ("LA", "PA"):
                img = img.convert("RGBA")
            elif img.mode not in ("RGB", "L", "RGBA"):  # L=grayscale, RGBA is also fine
                img = img.convert("RGB")

            gray_img = img.convert("L")
            width, height = gray_img.size

            if width == 0 or height == 0:  # 处理空图像
                return original_img_for_return

            white_threshold = 245

            # --- 寻找边界 ---
            left = 0
            for x in range(width):
                is_white_column = True
                for y_scan in range(height):  # 使用 y_scan 避免与外部 y 冲突
                    if gray_img.getpixel((x, y_scan)) <= white_threshold:
                        is_white_column = False
                        break
                if not is_white_column:
                    left = x
                    break
            else:  # 所有列都是白色
                return original_img_for_return

            right = width - 1
            for x in range(width - 1, -1, -1):
                is_white_column = True
                for y_scan in range(height):
                    if gray_img.getpixel((x, y_scan)) <= white_threshold:
                        is_white_column = False
                        break
                if not is_white_column:
                    right = x
                    break

            top = 0
            for y in range(height):
                is_white_row = True
                for x_scan in range(width):  # 使用 x_scan 避免与外部 x 冲突
                    if gray_img.getpixel((x_scan, y)) <= white_threshold:
                        is_white_row = False
                        break
                if not is_white_row:
                    top = y
                    break
            else:  # 所有行都是白色
                return original_img_for_return

            bottom = height - 1
            for y in range(height - 1, -1, -1):
                is_white_row = True
                for x_scan in range(width):
                    if gray_img.getpixel((x_scan, y)) <= white_threshold:
                        is_white_row = False
                        break
                if not is_white_row:
                    bottom = y
                    break

            if left >= right or top >= bottom:  # 边界无效
                return original_img_for_return

            cropped_img = img.crop((left, top, right + 1, bottom + 1))
            return cropped_img
    except FileNotFoundError:
        print(f"错误: 文件未找到 {image_path}")
        return None
    except Exception as e:
        print(f"裁剪 {image_path} 时出错: {e}")
        try:  # 尝试返回原始图像的副本，如果主逻辑失败但文件可读
            with Image.open(image_path) as fallback_img:
                return fallback_img.copy()
        except Exception:
            return None


def process_folder(folder_path, base_output_folder_path):
    """
    处理源文件夹中的所有图片,进行裁剪、放缩。
    PNG图片将转换为JPG。
    处理后的图片将保存在 base_output_folder_path 下，并镜像源文件夹的目录结构。
    每个子文件夹内的图片将从1开始重新编号。

    参数:
        folder_path: 要处理的源文件夹路径 (例如: /path/to/j)
        base_output_folder_path: 保存处理后图片的基准输出文件夹路径 (例如: /path/to/j_processed)
    """
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"]

    for root, dirs, files in os.walk(folder_path):
        if os.path.abspath(root).startswith(os.path.abspath(base_output_folder_path)):
            dirs[:] = [
                d
                for d in dirs
                if not os.path.abspath(os.path.join(root, d)).startswith(
                    os.path.abspath(base_output_folder_path)
                )
            ]
            continue

        relative_path_from_input_root = os.path.relpath(root, folder_path)
        output_dir_for_current_root = os.path.join(
            base_output_folder_path, relative_path_from_input_root
        )

        if not os.path.exists(output_dir_for_current_root):
            os.makedirs(output_dir_for_current_root)

        image_files_in_current_root = sorted(
            [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
        )

        if not image_files_in_current_root:
            if files:
                print(f"注意: 目录 '{root}' 中没有支持的图片文件。")
            elif not dirs and relative_path_from_input_root != ".":
                print(f"注意: 目录 '{root}' 为空或不含图片。")
            elif relative_path_from_input_root == "." and not dirs and not files:
                print(f"注意: 源文件夹 '{root}' 为空。")
            continue

        print(f"处理目录: {root}  ->  输出到: {output_dir_for_current_root}")
        image_counter_for_this_folder = 1

        for file_name in image_files_in_current_root:
            original_file_path = os.path.join(root, file_name)
            original_file_extension = os.path.splitext(file_name)[1].lower()

            try:
                processed_img = crop_white_borders(original_file_path)
                if processed_img is None:
                    print(f"  跳过: {file_name} (裁剪/读取失败)")
                    continue

                try:
                    resized_img = processed_img.resize(
                        (1024, 1024), Image.Resampling.LANCZOS
                    )
                except AttributeError:  # 兼容旧版 Pillow
                    resized_img = processed_img.resize((1024, 1024), Image.LANCZOS)

                # --- 文件名和格式处理 ---
                image_to_save_final = resized_img
                output_target_extension = original_file_extension

                new_numbered_file_name_stem = str(image_counter_for_this_folder)

                if original_file_extension == ".png":
                    output_target_extension = ".jpg"  # PNGs will be saved as JPG

                    # 处理透明度 (转换为RGB，透明部分填充白色)
                    if image_to_save_final.mode == "RGBA":
                        background = Image.new(
                            "RGB", image_to_save_final.size, (255, 255, 255)
                        )
                        # 确保有alpha通道可用作mask
                        if len(image_to_save_final.split()) == 4:
                            alpha_channel = image_to_save_final.split()[3]
                            background.paste(image_to_save_final, (0, 0), alpha_channel)
                        else:  # 如果没有alpha通道（理论上RGBA模式应该有）
                            background.paste(image_to_save_final, (0, 0))  # 直接粘贴
                        image_to_save_final = background
                    elif image_to_save_final.mode == "P":  # 带调色板的PNG也可能有透明度
                        # 先尝试转换为RGBA处理透明度
                        try:
                            img_rgba = image_to_save_final.convert("RGBA")
                            background = Image.new(
                                "RGB", img_rgba.size, (255, 255, 255)
                            )
                            if len(img_rgba.split()) == 4:
                                alpha_channel = img_rgba.split()[3]
                                background.paste(img_rgba, (0, 0), alpha_channel)
                            else:
                                background.paste(img_rgba, (0, 0))
                            image_to_save_final = background
                        except (
                            ValueError
                        ):  # 如果转换RGBA失败（例如，P模式但无透明度信息）
                            image_to_save_final = image_to_save_final.convert(
                                "RGB"
                            )  # 直接转RGB
                    elif image_to_save_final.mode not in (
                        "RGB",
                        "L",
                    ):  # L (grayscale) is fine for JPEG
                        image_to_save_final = image_to_save_final.convert("RGB")

                    pillow_save_format = "JPEG"

                else:  # 非PNG文件，按原格式处理
                    pillow_save_format = original_file_extension.upper().replace(
                        ".", ""
                    )
                    if pillow_save_format == "JPG":  # Pillow bevorzugt "JPEG"
                        pillow_save_format = "JPEG"

                new_file_full_path = os.path.join(
                    output_dir_for_current_root,
                    new_numbered_file_name_stem + output_target_extension,
                )

                # --- 保存逻辑 ---
                if pillow_save_format == "JPEG":
                    # 确保非PNG转JPEG时也是RGB (对于P模式JPG等)
                    if (
                        image_to_save_final.mode == "P"
                        and original_file_extension != ".png"
                    ):  # 检查是否是原始P模式且非已处理的PNG
                        image_to_save_final = image_to_save_final.convert("RGB")
                    elif (
                        image_to_save_final.mode == "RGBA"
                        and original_file_extension != ".png"
                    ):  # 原始RGBA (非PNG)转JPG
                        image_to_save_final = image_to_save_final.convert(
                            "RGB"
                        )  # 简单转换，可能丢失透明度信息

                    # 对于灰度图'L'，Pillow可以直接保存为JPEG
                    if (
                        image_to_save_final.mode not in ("RGB", "L")
                        and original_file_extension != ".png"
                    ):
                        image_to_save_final = image_to_save_final.convert("RGB")

                    image_to_save_final.save(new_file_full_path, "JPEG", quality=95)

                elif pillow_save_format == "BMP":
                    if image_to_save_final.mode not in ("RGB", "L", "P", "1"):
                        image_to_save_final = image_to_save_final.convert("RGB")
                    image_to_save_final.save(new_file_full_path, "BMP")

                elif pillow_save_format == "GIF":
                    if image_to_save_final.mode != "P":
                        image_to_save_final = image_to_save_final.convert(
                            "RGB"
                        ).convert(
                            "P", palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG
                        )
                    save_all_frames = (
                        hasattr(image_to_save_final, "n_frames")
                        and image_to_save_final.n_frames > 1
                        and hasattr(image_to_save_final, "is_animated")
                        and image_to_save_final.is_animated
                    )
                    image_to_save_final.save(
                        new_file_full_path,
                        "GIF",
                        save_all=save_all_frames,
                        duration=image_to_save_final.info.get("duration", 100),
                        loop=image_to_save_final.info.get("loop", 0),
                    )

                elif pillow_save_format == "WEBP":
                    # WEBP支持RGB和RGBA，如果想保留透明度（来自非PNG的WEBP），需要额外处理
                    if image_to_save_final.mode == "P":  # P模式的WEBP转RGB或RGBA
                        if "transparency" in image_to_save_final.info:
                            image_to_save_final = image_to_save_final.convert("RGBA")
                        else:
                            image_to_save_final = image_to_save_final.convert("RGB")
                    image_to_save_final.save(new_file_full_path, "WEBP", quality=90)

                elif pillow_save_format == "TIFF":
                    image_to_save_final.save(new_file_full_path, "TIFF")

                else:  # 其他在 valid_extensions 中的格式
                    try:
                        image_to_save_final.save(new_file_full_path, pillow_save_format)
                    except ValueError:  # Pillow无法识别格式字符串时，尝试让其自动推断
                        image_to_save_final.save(new_file_full_path)

                print(
                    f"    已处理: {file_name}  ->  {new_numbered_file_name_stem + output_target_extension}"
                )
                image_counter_for_this_folder += 1

            except Exception as e:
                print(f"  处理 {original_file_path} 时发生意外错误: {e}")


if __name__ == "__main__":
    input_folder_path_str = input("请输入要处理的源文件夹路径 (例如: j): ")

    if os.path.exists(input_folder_path_str) and os.path.isdir(input_folder_path_str):
        normalized_input_path = os.path.normpath(input_folder_path_str)
        input_folder_name = os.path.basename(normalized_input_path)
        parent_dir_of_input = os.path.dirname(normalized_input_path)
        mirrored_output_root_name = f"{input_folder_name}_processed"
        mirrored_output_root_path = os.path.join(
            parent_dir_of_input, mirrored_output_root_name
        )

        if not parent_dir_of_input or parent_dir_of_input == normalized_input_path:
            mirrored_output_root_path = os.path.join(
                os.getcwd(), mirrored_output_root_name
            )
            print(
                f"输入文件夹 '{normalized_input_path}' 似乎是驱动器根目录或特殊路径。"
            )
            print(f"输出将创建在当前工作目录下的 '{mirrored_output_root_name}' 中。")

        if not os.path.exists(mirrored_output_root_path):
            os.makedirs(mirrored_output_root_path)
            print(f"创建主输出目录: {mirrored_output_root_path}")
        else:
            print(
                f"主输出目录已存在: {mirrored_output_root_path} (将在此基础上创建或更新文件)"
            )

        print(f"\n处理后的图片将镜像源文件夹 '{normalized_input_path}' 的结构,")
        print(f"并将PNG图片转换为JPG格式保存在 '{mirrored_output_root_path}' 中。")
        print("(每个子文件夹内的图片将从1开始重新编号)\n")

        process_folder(normalized_input_path, mirrored_output_root_path)
        print("\n所有图片处理完成！")
    elif not os.path.exists(input_folder_path_str):
        print(f"错误: 指定的源文件夹 '{input_folder_path_str}' 不存在。")
    else:  # 路径存在但不是一个文件夹
        print(f"错误: 指定的源路径 '{input_folder_path_str}' 不是一个文件夹。")
