"""
Backend module - EXACT COPY dari Result_Segmentation_and_Transliteration.ipynb
Tidak ada perubahan pada fungsi-fungsi, hanya copy-paste
UPDATED: Menambahkan dukungan untuk SVM dengan scaler (opsional)
"""

import cv2
import numpy as np
import os
import itertools
import pandas as pd
from skimage import color, measure
from scipy import ndimage
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import glob
from collections import Counter
from pathlib import Path
from skimage.transform import resize
from skimage.measure import label, regionprops
import math
import re
import time
import joblib

# ============================================================
# PREPROCESSING (Exact copy dari notebook)
# ============================================================

def folder_path(output_folder, process, file_name=None):
    if process == "grayscale":
        return os.path.join(output_folder, "Grayscale", file_name)
    elif process == "binary":
        return os.path.join(output_folder, "Binary", file_name)
    elif process == "rotated":
        return os.path.join(output_folder, "Orientation_Correct", file_name)
    elif process == "denoised":
        return os.path.join(output_folder, "Noise_Reduct", file_name)
    elif process == "cropping":
        return os.path.join(output_folder, "Cropping", file_name)
    elif process == "projection_profile":
        return os.path.join(output_folder, "Projection_Profile", file_name)
    elif process == "four_dfs":
        return os.path.join(output_folder, "Four_DFS")


def grayScale_method(input_image, output_image_path=None):
    original_img = cv2.imread(input_image)

    if original_img is None:
        raise ValueError("Failed to load image")

    print(original_img.shape)

    height, width, _ = original_img.shape

    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            b, g, r = original_img[y, x]
            grayScale_value = int(0.229 * r + 0.587 * g + 0.114 * b)
            grayscale_image[y, x] = grayScale_value

    if output_image_path:
        output_folder = os.path.dirname(output_image_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(output_image_path, grayscale_image)
    print("Grayscale created")
    return grayscale_image


def binary_method(gray_image, output_image_path=None, ws=25, k=0.2):
    window_size = ws
    k = k

    thresh_sauvola = threshold_sauvola(gray_image, window_size=window_size, k=k)

    binary_image = (gray_image > thresh_sauvola).astype(np.uint8) * 255

    print(binary_image.shape)

    if output_image_path:
        output_folder = os.path.dirname(output_image_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(output_image_path, binary_image)
    print("Binary created")
    return binary_image


def rotate_method(binary_image, output_image_path=None):
    moments = measure.moments(binary_image, order=2)

    mu20 = moments[2, 0]
    mu02 = moments[0, 2]
    mu11 = moments[1, 1]

    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    print(theta)

    (height, width) = binary_image.shape
    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, -theta, 1.0)

    rotated_image = cv2.warpAffine(binary_image, rotation_matrix, (width, height),
                                   flags=cv2.INTER_LINEAR, borderValue=255)

    if output_image_path:
        output_folder = os.path.dirname(output_image_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        rotated_binary_image = binary_method(rotated_image, None, ws=75, k=0.2)
        cv2.imwrite(output_image_path, rotated_binary_image)
    print("Rotate created")
    return rotated_binary_image


def denoise_median(binary_image, output_image_path=None, kernel_size=5):
    median_result = cv2.medianBlur(binary_image, kernel_size)
    print(median_result.shape)
    if output_image_path:
        output_folder = os.path.dirname(output_image_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        cv2.imwrite(output_image_path, median_result)
    print("Noise Reduction created")
    return median_result


def crop_text_region(nr_binary_image, output_image_path=None, padding_threshold=10):
    if np.max(nr_binary_image) == 255:
        nr_binary_image = nr_binary_image // 255

    h_hist = np.sum(nr_binary_image == 0, axis=1)
    v_hist = np.sum(nr_binary_image == 0, axis=0)

    h_thresh = np.median(h_hist)
    v_thresh = np.median(v_hist)

    rows = np.where(h_hist >= h_thresh)[0]
    cols = np.where(v_hist >= v_thresh)[0]

    if len(rows) == 0 or len(cols) == 0:
        print("No text detected. Returning original image.")
        return nr_binary_image

    row_min, row_max = rows[0], rows[-1]
    col_min, col_max = cols[0], cols[-1]

    padding_row = int((padding_threshold / 100) * (row_max - row_min))
    padding_col = int((padding_threshold / 100) * (col_max - col_min))

    row_min = max(0, row_min - padding_row)
    row_max = min(nr_binary_image.shape[0], row_max + padding_row)
    col_min = max(0, col_min - padding_col)
    col_max = min(nr_binary_image.shape[1], col_max + padding_col)

    cropped_image = nr_binary_image[row_min:row_max, col_min:col_max]

    image_with_box = cv2.cvtColor(nr_binary_image * 255, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image_with_box, (col_min, row_min), (col_max, row_max), (0, 0, 255), 2)

    if output_image_path:
        output_folder = os.path.dirname(output_image_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        cv2.imwrite(output_image_path, cropped_image * 255)
    print("Cropped created")
    return cropped_image, (row_min, row_max, col_min, col_max)


def projection_profile(binary_image, smooth_sigma=20, row=False, column=False, output_image_path=None, draw_lines=True, segmentation_row=False, segmentation_col=False):
    if segmentation_row == True:
        pp = np.sum(binary_image == 0, axis=1)
    elif segmentation_col == True:
        pp = np.sum(binary_image == 0, axis=0)
    else:
        raise ValueError("Please specify either segmentation_row or segmentation_col as True.")
    pp_smooth = gaussian_filter1d(pp, sigma=smooth_sigma)
    peaks, _ = find_peaks(pp_smooth)
    valleys, _ = find_peaks(-pp_smooth)

    if len(valleys) == 0:
        print("Tidak ditemukan valleys, segmentasi kosong.")
        return []

    if len(peaks) == 0 or peaks[0] > valleys[0]:
        if pp_smooth[0] > pp_smooth[valleys[0]]:
            peaks = np.insert(peaks, 0, 0)

    if valleys[0] > peaks[0]:
        valleys = np.insert(valleys, 0, 0)

    if valleys[-1] < peaks[-1]:
        valleys = np.append(valleys, len(pp_smooth) - 1)

    segment = list(zip(valleys[:-1], valleys[1:]))

    image_with_lines = cv2.cvtColor((binary_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if draw_lines:
        for start, end in segment:
            cv2.line(image_with_lines, (0, start), (image_with_lines.shape[1], start), (255, 0, 0), 2)
            cv2.line(image_with_lines, (0, end), (image_with_lines.shape[1], end), (255, 0, 0), 2)

    if output_image_path:
        output_folder = os.path.dirname(output_image_path)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        cv2.imwrite(output_image_path, image_with_lines)

        segmented_folder = os.path.join(output_folder, f"segmented_lines_{len(segment)}")
        os.makedirs(segmented_folder, exist_ok=True)

        for i, (start, end) in enumerate(segment):
            line_image = binary_image[start:end, :]
            output_path = os.path.join(segmented_folder, f"baris_{i + 1}.png")
            cv2.imwrite(output_path, line_image * 255)
    print(f"Berhasil membagi citra menjadi {len(segment)} segmen.")
    return segment


def segmentation_row(binary_image, output_folder, sigma=10):
    return projection_profile(binary_image, output_image_path=None, smooth_sigma=sigma, row=True, column=False, draw_lines=False, segmentation_row=True)


def segmentation_col(binary_image, output_folder, sigma=10):
    return projection_profile(binary_image, output_image_path=None, smooth_sigma=sigma, row=False, column=True, draw_lines=False, segmentation_col=True)


def segmentation_pp_result_df(cropped_image, binary_image, output_folder, sigma_row=10, sigma_col=10):
    segmentation_data = []

    row_segments = segmentation_row(cropped_image, output_folder, sigma=sigma_row)

    for i, (start_row, end_row) in enumerate(row_segments, start=1):
        print(i, start_row, end_row)
        baris_image = cropped_image[start_row:end_row, :]
        process_image = binary_image[start_row:end_row, :]

        if baris_image.size == 0:
            print(f"Warning: Baris {i} is empty after segmentation.")
            continue

        col_segments = segmentation_col(baris_image, output_folder, sigma=sigma_col)

        for j, (start_col, end_col) in enumerate(col_segments, start=1):
            col_image = process_image[:, start_col:end_col]

            segmentation_data.append({
                "row_id": i,
                "col_id": j,
                "start_row": int(start_row),
                "end_row": int(end_row),
                "start_col": int(start_col),
                "end_col": int(end_col),
                "binary_image": col_image.copy()
            })

            row_folder = os.path.join(output_folder, f"row_{i}")
            os.makedirs(row_folder, exist_ok=True)
            col_image = process_image[:, start_col:end_col]
            if col_image.size > 0:
                cv2.imwrite(os.path.join(row_folder, f"col_{j}.png"), col_image * 255)

        for start_col, end_col in col_segments:
            cv2.line(process_image, (start_col, 0), (start_col, process_image.shape[0]), (255, 0, 0), 2)
            cv2.line(process_image, (end_col, 0), (end_col, process_image.shape[0]), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_folder, f"row_{i}_with_lines.png"), process_image * 255)

    df_segments = pd.DataFrame(segmentation_data)
    return df_segments


def process_image(input_path=None, output_base_folder=None, name=None, sigma_row=10, sigma_col=10):
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    if os.path.isfile(input_path):
        image_files = [input_path]
    elif os.path.isdir(input_path):
        image_files = glob.glob(os.path.join(input_path, "*.jpg")) + glob.glob(os.path.join(input_path, "*.png"))
    else:
        print("⌠Path tidak ditemukan atau bukan file/folder gambar yang valid.")
        return

    for idx, image_path in enumerate(image_files, start=1):
        file_name = os.path.basename(image_path)
        file_name = os.path.splitext(file_name)[0] + ".png"
        output_folder = os.path.join(output_base_folder, f"Result {idx}{name if name else ''}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Processing {image_path} -> {output_folder}")

        # Grayscale
        grayScale_path = folder_path(output_folder=output_folder, process="grayscale", file_name=f"grayscale_{file_name}")
        result_grayScale = grayScale_method(image_path, grayScale_path)

        # Binary
        ws = 75
        k = 0.2
        binary_path = folder_path(output_folder=output_folder, process="binary", file_name=f"binary_img_{file_name}")
        result_binary = binary_method(result_grayScale, binary_path, ws, k)

        # Orientation Correction
        rotate_path = folder_path(output_folder=output_folder, process="rotated", file_name=f"rotationCorection_{file_name}")
        result_orientationCorection = rotate_method(result_binary, rotate_path)

        # Noise Reduction
        binary_image = binary_method(result_orientationCorection, None, ws=75, k=0.2)
        ks = 5
        result_noise_path = folder_path(output_folder=output_folder, process="denoised", file_name=f"noise_reduction_{file_name}")
        result_noise_reduction = denoise_median(binary_image, result_noise_path, kernel_size=ks)

        # Cropping
        th = 10
        cropped_path = folder_path(output_folder=output_folder, process="cropping", file_name=f"cropping_{file_name}")
        result_cropped, crop_bounds = crop_text_region(result_noise_reduction, cropped_path, padding_threshold=th)
        row_min, row_max, col_min, col_max = crop_bounds

        # Projection Profile
        binary_cropped = binary_image[row_min:row_max, col_min:col_max]
        binary_cropped = (binary_cropped > 0).astype(np.uint8)
        pp_path = folder_path(output_folder=output_folder, process="projection_profile", file_name=f"smooth_{file_name}")
        result_pp = segmentation_pp_result_df(result_cropped, binary_cropped, pp_path, sigma_row=sigma_row, sigma_col=sigma_col)

        print(result_pp.head())
        result_pp['start_row'] += row_min
        result_pp['end_row'] += row_min
        result_pp['start_col'] += col_min
        result_pp['end_col'] += col_min
        result_pp.to_csv(os.path.join(output_folder, "segmented_projection_profile.csv"), index=False)
        print("Process End.")
    return result_pp


# ============================================================
# FILTERING (Exact copy dari notebook)
# ============================================================

def four_dfs_method(binary_image):
    rows, cols = binary_image.shape
    visited = np.zeros((rows, cols), dtype=bool)
    all_objects = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(rows):
        for col in range(cols):
            if binary_image[row][col] == 0 and not visited[row][col]:
                stack = [(row, col)]
                coor_obj = []

                while stack:
                    cur_row, cur_col = stack.pop()

                    if visited[cur_row][cur_col]:
                        continue

                    visited[cur_row][cur_col] = True
                    coor_obj.append((cur_row, cur_col))

                    for d_row, d_col in directions:
                        new_row, new_col = cur_row + d_row, cur_col + d_col
                        if (0 <= new_row < rows and 0 <= new_col < cols and
                                binary_image[new_row][new_col] == 0 and
                                not visited[new_row][new_col]):
                            stack.append((new_row, new_col))

                all_objects.append({
                    "Object": len(all_objects) + 1,
                    "Coordinates": coor_obj
                })

    df_all_objects = pd.DataFrame(all_objects)
    return df_all_objects


def filter_objects(binary_image, method="median", keep="larger", th=20):
    df_objects = four_dfs_method(binary_image)

    if len(df_objects) == 0:
        return binary_image.copy(), pd.DataFrame()

    df_objects["Size"] = df_objects["Coordinates"].apply(len)

    if method == "median":
        threshold = df_objects["Size"].median()
    elif method == "mean":
        threshold = df_objects["Size"].mean()
    elif method == "manual":
        threshold = th
    else:
        raise ValueError("Method harus 'median' atau 'mean'")

    if keep == "larger":
        df_objects["Status"] = df_objects["Size"].apply(
            lambda x: "kept" if x >= threshold else "removed"
        )
    elif keep == "smaller":
        df_objects["Status"] = df_objects["Size"].apply(
            lambda x: "kept" if x < threshold else "removed"
        )
    else:
        raise ValueError("keep harus 'larger' atau 'smaller'")

    filtered_objects = df_objects[df_objects["Status"] == "kept"]

    cleaned_binary = np.ones_like(binary_image, dtype=np.uint8)

    for coords in filtered_objects["Coordinates"]:
        for (r, c) in coords:
            cleaned_binary[r, c] = 0

    return cleaned_binary, df_objects


def save_binary_images(df_results, output_folder="output_images", save_original=True):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    saved_files = []

    for idx, row in df_results.iterrows():
        base_filename = f"row{row['row_id']}_col{row['col_id']}"

        if save_original:
            original_img = row['original_binary_image'] * 255
            original_path = os.path.join(output_folder, f"{base_filename}_original.png")
            cv2.imwrite(original_path, original_img)

        cleaned_img = row['cleaned_binary_image'] * 255
        cleaned_path = os.path.join(output_folder, f"{base_filename}_cleaned.png")
        cv2.imwrite(cleaned_path, cleaned_img)

        saved_files.append({
            'row_id': row['row_id'],
            'col_id': row['col_id'],
            'original_path': original_path if save_original else None,
            'cleaned_path': cleaned_path,
            'total_objects': row['total_objects'],
            'kept_objects': row['kept_objects'],
            'removed_objects': row['removed_objects']
        })

    print(f"✅ Berhasil menyimpan {len(saved_files)} gambar ke folder '{output_folder}'")

    return pd.DataFrame(saved_files)


def process_and_save(df, output_folder="output_images", method="median", keep="larger", save_original=True, th=20):
    results = []

    for idx, row in df.iterrows():
        binary_img = row['binary_image']

        cleaned_img, df_objects = filter_objects(
            binary_img,
            method=method,
            keep=keep,
            th=th
        )

        if len(df_objects) > 0:
            total_objs = len(df_objects)
            kept_objs = len(df_objects[df_objects["Status"] == "kept"])
            removed_objs = total_objs - kept_objs
            threshold = df_objects["Size"].median() if method == "median" else df_objects["Size"].mean()
        else:
            total_objs = kept_objs = removed_objs = threshold = 0

        results.append({
            'row_id': row.get('row_id', idx),
            'col_id': row['col_id'],
            'start_row': row['start_row'],
            'end_row': row['end_row'],
            'start_col': row['start_col'],
            'end_col': row['end_col'],
            'original_binary_image': binary_img,
            'cleaned_binary_image': cleaned_img,
            'total_objects': total_objs,
            'kept_objects': kept_objs,
            'removed_objects': removed_objs,
            'threshold': threshold,
            'objects_detail': df_objects
        })

    df_results = pd.DataFrame(results)

    df_saved = save_binary_images(df_results, output_folder, save_original)

    return df_results, df_saved


# ============================================================
# CROPPING & NORMALIZATION (Exact copy dari notebook)
# ============================================================

def crop_white_space_segment(image):
    if image is None:
        raise ValueError("Input image is None")

    img = np.asarray(image)

    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype == bool or np.max(gray) <= 1:
        gray = (gray.astype(np.uint8) * 255)

    unique_vals = np.unique(gray)
    if len(unique_vals) > 2:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = gray.copy()

    fg_mask = (binary < 128)
    if not fg_mask.any():
        return (binary > 128).astype(np.uint8)

    rows = np.where(fg_mask.any(axis=1))[0]
    cols = np.where(fg_mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    cropped = binary[top:bottom + 1, left:right + 1]

    cropped_bin01 = (cropped > 128).astype(np.uint8)

    return cropped_bin01


def make_square(image, background=1):
    if image.ndim != 2:
        raise ValueError("Input harus citra 2D (binary/grayscale)")

    h, w = image.shape
    size = max(h, w)
    square_img = np.full((size, size), background, dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return square_img


def rescale_image(binary_image, output_size=(90, 90)):
    resized = cv2.resize(binary_image.astype(np.uint8), output_size,
                         interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.uint8)


def process_image_binary_1x1(df, binary_column, output_folder="output_1x1"):
    processed_images = []
    cropped_images = []
    errors = []

    os.makedirs(output_folder, exist_ok=True)

    for index, row in df.iterrows():
        try:
            binary_image = row[binary_column]
            if not isinstance(binary_image, np.ndarray):
                raise ValueError(f"Image at index {index} is not a numpy array")
            if binary_image.size == 0:
                raise ValueError(f"Image at index {index} is empty")

            cropped_image = crop_white_space_segment(binary_image)
            if cropped_image.size == 0:
                raise ValueError(f"Cropping resulted in an empty image at index {index}")

            cropped_path = os.path.join(output_folder, f"cropped_{index}.png")
            cv2.imwrite(cropped_path, (cropped_image * 255).astype(np.uint8))

            square_image = make_square(cropped_image, background=1)
            square_path = os.path.join(output_folder, f"square_{index}.png")
            cv2.imwrite(square_path, (square_image * 255).astype(np.uint8))

            cropped_images.append(cropped_image)
            processed_images.append(square_image)

        except Exception as e:
            print(f"Error processing image at index {index}: {str(e)}")
            errors.append((index, str(e)))
            cropped_images.append(None)
            processed_images.append(None)

    df['Cropped_image_array'] = cropped_images
    df['Square_image_array'] = processed_images

    print(f"Process complete. {len(df) - len(errors)} sukses, {len(errors)} gagal.")
    return df


def rescale_image_90x90(df, name_column, output_size=(90, 90),
                        output_path="output_90x90", save_prefix="rescaled"):
    rescaled_images = []
    errors = []

    os.makedirs(output_path, exist_ok=True)

    for index, row in df.iterrows():
        try:
            binary_image = row[name_column]
            if binary_image is None or not isinstance(binary_image, np.ndarray):
                raise ValueError(f"Invalid image data at index {index}")

            rescaled_image = rescale_image(binary_image, output_size)
            rescaled_images.append(rescaled_image)

            save_path = os.path.join(output_path, f"{save_prefix}_{index}.png")
            cv2.imwrite(save_path, (rescaled_image * 255).astype(np.uint8))

        except Exception as e:
            print(f"Error processing image at index {index}: {str(e)}")
            errors.append((index, str(e)))
            rescaled_images.append(None)

    df['Processed_image_array_90X90'] = rescaled_images
    return df


# ============================================================
# FEATURE EXTRACTION (Exact copy dari notebook)
# ============================================================

def preprocess_bin(img, out_size=(90, 90), thresh=127):
    img = np.asarray(img).copy()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.max() > 1:
        _, imgb = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        imgb = (img * 255).astype(np.uint8)
        imgb = cv2.bitwise_not(imgb)

    ys, xs = np.where(imgb == 255)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(out_size, dtype=np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    roi = imgb[y0:y1 + 1, x0:x1 + 1]

    h, w = roi.shape
    scale = min((out_size[0] - 4) / h, (out_size[1] - 4) / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    roi_rs = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros(out_size, dtype=np.uint8)
    y_off = (out_size[0] - new_h) // 2
    x_off = (out_size[1] - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = roi_rs

    canvas = (canvas > 127).astype(np.uint8)

    return canvas


def feature_pixel_density(img):
    total = img.size
    fg = img.sum()
    return float(fg) / total


def feature_bounding_box_stats(img):
    lbl = label(img)
    props = regionprops(lbl)
    if not props:
        return {
            "bbox_area": 0.0,
            "bbox_aspect": 0.0,
            "solidity": 0.0,
            "eccentricity": 0.0,
            "perimeter_area_ratio": 0.0
        }
    props = sorted(props, key=lambda p: p.area, reverse=True)[0]
    minr, minc, maxr, maxc = props.bbox
    bbox_area = (maxr - minr) * (maxc - minc)
    bbox_aspect = (maxc - minc) / (maxr - minr + 1e-8)
    area = props.area
    solidity = props.solidity if hasattr(props, "solidity") else (area / (bbox_area + 1e-8))
    eccentricity = props.eccentricity if hasattr(props, "eccentricity") else 0.0
    perimeter = props.perimeter if hasattr(props, "perimeter") else 0.0
    per_area_ratio = perimeter / (area + 1e-8)
    return {
        "bbox_area": float(bbox_area),
        "bbox_aspect": float(bbox_aspect),
        "solidity": float(solidity),
        "eccentricity": float(eccentricity),
        "perimeter_area_ratio": float(per_area_ratio)
    }


def feature_projection_profiles(img, bins=16):
    h, w = img.shape
    hpp = img.sum(axis=1)
    vpp = img.sum(axis=0)

    def agg(vec, n_bins):
        L = len(vec)
        if L == 0:
            return [0.0] * n_bins
        chunk = int(math.ceil(L / n_bins))
        out = []
        for i in range(0, L, chunk):
            out.append(float(vec[i:i + chunk].sum()))
        while len(out) < n_bins:
            out.append(0.0)
        return out[:n_bins]

    return agg(hpp, bins) + agg(vpp, bins)


def feature_zoning(img, grid=(8, 8)):
    h, w = img.shape
    gy, gx = grid
    zone_h = h // gy
    zone_w = w // gx
    feats = []
    for i in range(gy):
        for j in range(gx):
            y0 = i * zone_h
            x0 = j * zone_w
            y1 = (i + 1) * zone_h if i < gy - 1 else h
            x1 = (j + 1) * zone_w if j < gx - 1 else w
            block = img[y0:y1, x0:x1]
            feats.append(float(block.sum()) / (block.size + 1e-8))
    return feats


def feature_hu_moments(img):
    img_255 = (img * 255).astype(np.uint8)
    moments = cv2.moments(img_255, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = []
    for v in hu:
        if v == 0:
            hu_log.append(0.0)
        else:
            hu_log.append(-1.0 * math.copysign(1.0, v) * math.log10(abs(v) + 1e-30))
    return hu_log


def extract_features_from_binary_image(img, out_size=(90, 90), zoning_grid=(8, 8), proj_bins=16):
    imgp = preprocess_bin(img, out_size=out_size)
    features = {}
    features["pixel_density"] = feature_pixel_density(imgp)
    bbox_stats = feature_bounding_box_stats(imgp)
    features.update(bbox_stats)
    proj = feature_projection_profiles(imgp, bins=proj_bins)
    for i, v in enumerate(proj):
        features[f"proj_{i}"] = float(v)
    zones = feature_zoning(imgp, grid=zoning_grid)
    for i, v in enumerate(zones):
        features[f"zone_{i}"] = float(v)
    hu = feature_hu_moments(imgp)
    for i, v in enumerate(hu):
        features[f"hu_{i}"] = float(v)
    features["height"] = float(imgp.shape[0])
    features["width"] = float(imgp.shape[1])
    return features


def batch_extract_to_dataframe(list_of_images, labels=None, out_size=(90, 90), zoning_grid=(8, 8), proj_bins=16):
    rows = []
    for img in list_of_images:
        rows.append(extract_features_from_binary_image(img, out_size=out_size, zoning_grid=zoning_grid, proj_bins=proj_bins))
    df = pd.DataFrame(rows)
    if labels is not None:
        df['label'] = labels
    return df


def load_images_from_folders(base_path):
    images = []
    labels = []
    for label_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, label_name)
        if not os.path.isdir(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(folder_path, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label_name)
    return images, labels


# ============================================================
# PREDICTION & TRANSLITERATION (UPDATED FOR SVM)
# ============================================================

def predict_image(features, model_path, scaler_path=None):
    """
    Predict using SVM model (with optional scaler for normalization)
    
    Parameters:
    -----------
    features : numpy.ndarray or pandas.DataFrame
        Feature matrix to predict
    model_path : str
        Path to the trained model (.pkl file) - supports both KNN and SVM
    scaler_path : str, optional
        Path to the fitted scaler (.pkl file)
        If provided, features will be normalized before prediction
        CRITICAL for SVM models!
        
    Returns:
    --------
    list : Predicted class labels
    """
    if not isinstance(features, np.ndarray):
        features = np.array(features)

    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    results = []
    try:
        # Load model
        model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        print(f"  - Model type: {type(model).__name__}")
        
        # Load scaler if provided (IMPORTANT for SVM!)
        if scaler_path:
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                features_original = features.copy()
                features = scaler.transform(features)
                print(f"✓ Features normalized using scaler from: {scaler_path}")
                print(f"  - Feature range before: [{features_original.min():.2f}, {features_original.max():.2f}]")
                print(f"  - Feature range after:  [{features.min():.2f}, {features.max():.2f}]")
            else:
                print(f"⚠️  Warning: Scaler path provided but file not found: {scaler_path}")
                print(f"    Proceeding without normalization - accuracy may be severely affected!")
        else:
            print("ℹ️  No scaler provided (OK for KNN, BAD for SVM)")
        
        # Predict
        start_time = time.time()
        predictions = model.predict(features)
        end_time = time.time()
        prediction_time = end_time - start_time
        
        # Convert predictions to list
        for prediction in predictions:
            results.append(f"{prediction}")

        print(f"✓ Prediction completed!")
        print(f"  - Prediction shape: {predictions.shape}")
        print(f"  - Number of predictions: {len(results)}")
        print(f"  - Prediction time: {prediction_time:.6f} seconds")
        print(f"  - Average time per sample: {(prediction_time/len(features)*1000):.3f} ms")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        results = ["Error" for _ in range(len(features))]

    return results


def combine_latin_transliteration(latin_text):
    latin_text = " ".join(latin_text)

    print(latin_text)

    def taling_tarung(match):
        suku_kata = match.group(1)
        suku_kata_ganti = re.sub(r'[a]', 'o', suku_kata)
        return f"{suku_kata_ganti}"

    def taling_e(match):
        suku_kata = match.group(1)
        suku_kata_ganti = re.sub(r'[a]', 'e', suku_kata)
        return f"{suku_kata_ganti}"

    latin_text = re.sub(r'taling\s(\w+)\starung', taling_tarung, latin_text)

    latin_text = re.sub(r'taling\s(\w+)', taling_e, latin_text)

    latin_text = latin_text.replace("  ", " ").strip()

    return latin_text
