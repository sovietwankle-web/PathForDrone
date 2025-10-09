#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础功能测试脚本
验证地形处理器的核心功能
"""

import sys
from pathlib import Path
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_functionality():
    """测试基础功能"""
    print("开始测试地形处理器基础功能...")
    print("=" * 50)
    
    try:
        # 1. 导入模块
        print("1. 导入模块...")
        from terrain_processor import TerrainProcessor
        print("   模块导入成功")
        
        # 2. 创建处理器实例
        print("\n2. 创建处理器实例...")
        tif_file = "test_terrain.tif"
        
        if not Path(tif_file).exists():
            print(f"   错误: 测试文件不存在: {tif_file}")
            return False
        
        processor = TerrainProcessor(tif_file)
        print("   处理器创建成功")
        
        # 3. 加载地形数据
        print("\n3. 加载地形数据...")
        terrain_data = processor.load_terrain_data()
        print(f"   数据形状: {terrain_data.shape}")
        print(f"   数据类型: {terrain_data.dtype}")
        
        # 4. 获取文件信息
        print("\n4. 获取文件信息...")
        info = processor.get_terrain_info()
        print(f"   文件大小: {info['file_size_mb']:.2f} MB")
        print(f"   坐标系统: {info['coordinate_system']}")
        
        # 5. 获取统计信息
        print("\n5. 获取统计信息...")
        stats = processor.get_terrain_statistics()
        print(f"   有效像素: {stats['valid_pixels']:,}")
        print(f"   最低高程: {stats['min_elevation']:.2f} m")
        print(f"   最高高程: {stats['max_elevation']:.2f} m")
        print(f"   平均高程: {stats['mean_elevation']:.2f} m")
        
        # 6. 保存数组
        print("\n6. 保存三维数组...")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        npy_path = processor.save_array(output_dir / "terrain_data.npy", "npy")
        print(f"   NPY文件已保存: {npy_path}")
        
        # 验证保存的数据
        loaded_data = np.load(npy_path)
        if np.array_equal(terrain_data, loaded_data, equal_nan=True):
            print("   数据验证成功")
        else:
            print("   数据验证失败")
        
        # 7. 生成等高线图
        print("\n7. 生成等高线图...")
        try:
            contour_path = processor.generate_contour_image(
                output_dir / "contour_map.png",
                levels=15,
                colormap='terrain'
            )
            print(f"   等高线图已保存: {contour_path}")
        except Exception as e:
            print(f"   等高线图生成失败: {e}")
        
        # 8. 测试点高程查询
        print("\n8. 测试点高程查询...")
        center_x = terrain_data.shape[1] // 2
        center_y = terrain_data.shape[0] // 2
        elevation = processor.get_elevation_at_point(center_x, center_y)
        print(f"   中心点 ({center_x}, {center_y}) 高程: {elevation:.2f} m")
        
        print("\n" + "=" * 50)
        print("基础功能测试完成 - 所有功能正常!")
        return True
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_line_tool():
    """测试命令行工具"""
    print("\n测试命令行工具...")
    print("-" * 30)
    
    import subprocess
    
    try:
        # 测试显示帮助信息
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("命令行工具可用")
            return True
        else:
            print(f"命令行工具测试失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"命令行工具测试出错: {e}")
        return False

def main():
    """主函数"""
    print("地形处理器功能验证")
    print("=" * 60)
    
    # 测试基础功能
    basic_success = test_basic_functionality()
    
    # 测试命令行工具
    cli_success = test_command_line_tool()
    
    print("\n" + "=" * 60)
    print("验证结果:")
    print(f"基础功能: {'通过' if basic_success else '失败'}")
    print(f"命令行工具: {'通过' if cli_success else '失败'}")
    
    if basic_success and cli_success:
        print("\n所有测试通过! 地形处理器功能正常。")
        return True
    else:
        print("\n部分测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)