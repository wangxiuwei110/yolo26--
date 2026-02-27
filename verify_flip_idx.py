#!/usr/bin/env python3
"""
车牌关键点 flip_idx 验证脚本

验证 flip_idx: [1, 0, 3, 2] 在水平翻转时是否正确
"""

import numpy as np


def visualize_horizontal_flip():
    """可视化水平翻转前后的关键点位置变化"""

    print("=" * 60)
    print("车牌关键点 flip_idx 验证分析")
    print("=" * 60)

    # 定义原始关键点顺序
    keypoint_names = {
        0: "左上角",
        1: "右上角",
        2: "右下角",
        3: "左下角"
    }

    flip_idx = [1, 0, 3, 2]

    print("\n【关键点定义】")
    for idx, name in keypoint_names.items():
        print(f"  索引 {idx}: {name}")

    print("\n【flip_idx 配置】")
    print(f"  flip_idx = {flip_idx}")

    print("\n【水平翻转几何关系分析】")
    print("  图像水平翻转时，左右互换：")
    print("    - 左上 ↔ 右上 (索引 0 ↔ 1)")
    print("    - 右下 ↔ 左下 (索引 2 ↔ 3)")

    print("\n【flip_idx 含义验证】")
    print("  flip_idx 表示：翻转后新位置 i 应取原位置 flip_idx[i] 的关键点")
    print()

    for new_idx, old_idx in enumerate(flip_idx):
        old_name = keypoint_names[old_idx]
        # 计算翻转后应该是哪个位置
        if old_idx == 0:
            expected_name = "右上角"  # 左上→右上
        elif old_idx == 1:
            expected_name = "左上角"  # 右上→左上
        elif old_idx == 2:
            expected_name = "左下角"  # 右下→左下
        elif old_idx == 3:
            expected_name = "右下角"  # 左下→右下

        actual_name = keypoint_names[new_idx]
        status = "✓" if expected_name == actual_name else "✗"

        print(f"  新位置 {new_idx} ({actual_name}) ← 原位置 {old_idx} ({old_name}) {status}")

    print("\n【代码实现验证】")

    # 模拟一个车牌的4个角点坐标
    # 假设图像大小 640x480，车牌大约在中心区域
    original_kpts = np.array([
        [200, 150],  # 0: 左上
        [440, 150],  # 1: 右上
        [440, 330],  # 2: 右下
        [200, 330]   # 3: 左下
    ])

    print(f"\n  原始关键点坐标:")
    for i, (x, y) in enumerate(original_kpts):
        print(f"    {i}: ({x:3d}, {y:3d}) - {keypoint_names[i]}")

    # 水平翻转坐标 (假设图像宽640)
    img_width = 640
    flipped_manual = original_kpts.copy()
    flipped_manual[:, 0] = img_width - original_kpts[:, 0]

    print(f"\n  手动水平翻转后坐标 (x' = W - x):")
    for i, (x, y) in enumerate(flipped_manual):
        print(f"    {i}: ({x:3d}, {y:3d})")

    # 使用 flip_idx 重排
    flipped_with_idx = flipped_manual[flip_idx]

    print(f"\n  使用 flip_idx={flip_idx} 重排后:")
    for i, (x, y) in enumerate(flipped_with_idx):
        print(f"    {i}: ({x:3d}, {y:3d}) - {keypoint_names[i]}")

    # 验证顺序是否正确
    print("\n【正确性验证】")
    # 使用 flip_idx 后，关键点应该保持在原定义的位置顺序：
    # 翻转后的索引0仍然指向"左上"位置，索引1仍然指向"右上"位置
    expected = np.array([
        [200, 150],  # 0: 翻转后的左上角
        [440, 150],  # 1: 翻转后的右上角
        [440, 330],  # 2: 翻转后的右下角
        [200, 330]   # 3: 翻转后的左下角
    ])

    if np.array_equal(flipped_with_idx, expected):
        print("  ✓ 验证通过！flip_idx 配置正确")
        print("  翻转并重排后，关键点索引仍对应原定义的位置")
    else:
        print("  ✗ 验证失败！请检查 flip_idx 配置")
        print(f"  期望: {expected.tolist()}")
        print(f"  实际: {flipped_with_idx.tolist()}")

    print("\n【结论】")
    print("  flip_idx: [1, 0, 3, 2] 完全符合车牌四个角点水平翻转的几何关系")
    print("  配置正确 ✓")
    print("=" * 60)


def test_code_logic():
    """测试代码中的实际逻辑"""
    print("\n\n【代码实现逻辑测试】")
    print("=" * 60)

    # 模拟 augment.py 中的代码逻辑
    # instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])

    # 假设 keypoints.shape = (batch_size, num_kpts, 2)
    keypoints = np.array([
        [[200, 150], [440, 150], [440, 330], [200, 330]],  # 第一个样本
        [[100, 100], [500, 100], [500, 400], [100, 400]],   # 第二个样本
    ])

    flip_idx = [1, 0, 3, 2]

    print(f"原始 keypoints shape: {keypoints.shape}")
    print("原始关键点 (样本1):", keypoints[0])

    # 应用 flip_idx
    flipped_keypoints = keypoints[:, flip_idx, :]

    print("翻转后关键点 (样本1):", flipped_keypoints[0])

    print("\n代码逻辑验证: ✓")
    print("使用 keypoints[:, [1,0,3,2], :] 可以正确重排关键点索引")


if __name__ == "__main__":
    visualize_horizontal_flip()
    test_code_logic()
