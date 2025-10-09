#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地形处理器测试脚本
验证所有核心功能是否正常工作
"""

import sys
import unittest
import numpy as np
from pathlib import Path
import tempfile
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from terrain_processor import TerrainProcessor
from terrain_processor.reader import TIFReader
from terrain_processor.processor import DataProcessor
from terrain_processor.visualizer import ContourGenerator
from terrain_processor.utils import TerrainUtils


class TestTerrainProcessor(unittest.TestCase):
    """地形处理器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建测试数据文件路径
        cls.test_tif_path = Path(__file__).parent.parent / "test_terrain.tif"
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # 检查测试数据是否存在
        if not cls.test_tif_path.exists():
            # 如果测试数据不存在，创建一个简单的测试数据
            cls._create_test_data()
    
    @classmethod
    def _create_test_data(cls):
        """创建简单的测试数据"""
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS
            
            # 创建简单的测试地形数据
            width, height = 100, 100
            data = np.random.rand(height, width) * 100 + 50  # 50-150米高程
            
            # 添加一些地形特征
            center_x, center_y = width // 2, height // 2
            y, x = np.ogrid[:height, :width]
            
            # 添加一个山峰
            peak_mask = ((x - center_x)**2 + (y - center_y)**2) < 400
            data[peak_mask] += 50
            
            # 保存为GeoTIFF
            transform = from_bounds(116.2, 40.0, 116.3, 40.1, width, height)
            crs = CRS.from_epsg(4326)
            
            with rasterio.open(
                cls.test_tif_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform
            ) as dst:
                dst.write(data, 1)
                
        except ImportError:
            # 如果rasterio不可用，跳过测试
            raise unittest.SkipTest("rasterio不可用，跳过测试")
    
    def setUp(self):
        """每个测试前的设置"""
        if not self.test_tif_path.exists():
            self.skipTest("测试数据文件不存在")
        
        self.processor = TerrainProcessor(self.test_tif_path)
    
    def test_terrain_processor_initialization(self):
        """测试地形处理器初始化"""
        self.assertIsInstance(self.processor, TerrainProcessor)
        self.assertEqual(self.processor.tif_file_path, self.test_tif_path)
    
    def test_load_terrain_data(self):
        """测试加载地形数据"""
        data = self.processor.load_terrain_data()
        
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(len(data.shape), 2)  # 应该是2D数组
        self.assertGreater(data.size, 0)  # 应该有数据
    
    def test_get_terrain_info(self):
        """测试获取地形信息"""
        info = self.processor.get_terrain_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('file_path', info)
        self.assertIn('data_shape', info)
        self.assertIn('data_type', info)
        self.assertIn('coordinate_system', info)
    
    def test_get_terrain_statistics(self):
        """测试获取统计信息"""
        stats = self.processor.get_terrain_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('min_elevation', stats)
        self.assertIn('max_elevation', stats)
        self.assertIn('mean_elevation', stats)
        self.assertIn('valid_pixels', stats)
        
        # 检查统计值的合理性
        self.assertLessEqual(stats['min_elevation'], stats['max_elevation'])
        self.assertGreaterEqual(stats['valid_pixels'], 0)
    
    def test_save_array(self):
        """测试保存数组"""
        # 测试NPY格式
        npy_path = self.temp_dir / "test_array.npy"
        saved_path = self.processor.save_array(npy_path, format='npy')
        
        self.assertTrue(Path(saved_path).exists())
        
        # 验证保存的数据
        loaded_data = np.load(saved_path)
        original_data = self.processor.load_terrain_data()
        np.testing.assert_array_equal(loaded_data, original_data)
    
    def test_generate_contour_image(self):
        """测试生成等高线图"""
        contour_path = self.temp_dir / "test_contour.png"
        saved_path = self.processor.generate_contour_image(
            contour_path,
            levels=10,
            colormap='terrain'
        )
        
        self.assertTrue(Path(saved_path).exists())
        
        # 检查文件大小（应该不为空）
        file_size = Path(saved_path).stat().st_size
        self.assertGreater(file_size, 1000)  # 至少1KB
    
    def test_get_elevation_at_point(self):
        """测试获取指定点高程"""
        data = self.processor.load_terrain_data()
        height, width = data.shape
        
        # 测试中心点
        center_x, center_y = width // 2, height // 2
        elevation = self.processor.get_elevation_at_point(center_x, center_y)
        
        self.assertIsInstance(elevation, (int, float))
        self.assertFalse(np.isnan(elevation))
    
    def test_get_elevation_profile(self):
        """测试获取高程剖面"""
        data = self.processor.load_terrain_data()
        height, width = data.shape
        
        start_point = (0, height // 2)
        end_point = (width - 1, height // 2)
        
        distances, elevations = self.processor.get_elevation_profile(
            start_point, end_point, num_points=50
        )
        
        self.assertEqual(len(distances), 50)
        self.assertEqual(len(elevations), 50)
        self.assertGreater(distances[-1], 0)  # 总距离应该大于0


class TestTIFReader(unittest.TestCase):
    """TIF读取器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_tif_path = Path(__file__).parent.parent / "test_terrain.tif"
        if not self.test_tif_path.exists():
            self.skipTest("测试数据文件不存在")
        
        self.reader = TIFReader(self.test_tif_path)
    
    def test_reader_initialization(self):
        """测试读取器初始化"""
        self.assertIsInstance(self.reader, TIFReader)
        self.assertEqual(self.reader.file_path, self.test_tif_path)
    
    def test_read_tif(self):
        """测试读取TIF文件"""
        data, metadata = self.reader.read_tif()
        
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(metadata, dict)
        self.assertIn('driver', metadata)
        self.assertIn('width', metadata)
        self.assertIn('height', metadata)
    
    def test_get_file_info(self):
        """测试获取文件信息"""
        info = self.reader.get_file_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('file_path', info)
        self.assertIn('file_size_mb', info)
        self.assertIn('width', info)
        self.assertIn('height', info)
    
    def test_validate_data_integrity(self):
        """测试数据完整性验证"""
        validation = self.reader.validate_data_integrity()
        
        self.assertIsInstance(validation, dict)
        self.assertIn('file_readable', validation)
        self.assertTrue(validation['file_readable'])


class TestDataProcessor(unittest.TestCase):
    """数据处理器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = DataProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试数据
        self.test_data = np.random.rand(50, 50) * 100
    
    def test_save_and_load_array(self):
        """测试保存和加载数组"""
        # 测试NPY格式
        npy_path = self.temp_dir / "test.npy"
        saved_path = self.processor.save_array(self.test_data, npy_path, 'npy')
        loaded_data = self.processor.load_array(saved_path)
        
        np.testing.assert_array_equal(self.test_data, loaded_data)
    
    def test_resample_data(self):
        """测试数据重采样"""
        resampled = self.processor.resample_data(self.test_data, 25, 25, 'bilinear')
        
        self.assertEqual(resampled.shape, (25, 25))
        self.assertIsInstance(resampled, np.ndarray)
    
    def test_apply_filter(self):
        """测试应用滤波器"""
        filtered = self.processor.apply_filter(self.test_data, 'gaussian', sigma=1.0)
        
        self.assertEqual(filtered.shape, self.test_data.shape)
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_normalize_data(self):
        """测试数据归一化"""
        normalized = self.processor.normalize_data(self.test_data, 'minmax')
        
        self.assertEqual(normalized.shape, self.test_data.shape)
        self.assertGreaterEqual(np.nanmin(normalized), 0)
        self.assertLessEqual(np.nanmax(normalized), 1)


class TestContourGenerator(unittest.TestCase):
    """等高线生成器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.generator = ContourGenerator()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试数据
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        self.test_data = 100 + 50 * np.sin(X) * np.cos(Y)
    
    def test_generate_contour_plot(self):
        """测试生成等高线图"""
        output_path = self.temp_dir / "test_contour.png"
        
        saved_path = self.generator.generate_contour_plot(
            self.test_data,
            output_path,
            levels=10,
            colormap='terrain'
        )
        
        self.assertTrue(Path(saved_path).exists())
        self.assertGreater(Path(saved_path).stat().st_size, 1000)
    
    def test_generate_hillshade_plot(self):
        """测试生成山体阴影图"""
        output_path = self.temp_dir / "test_hillshade.png"
        
        saved_path = self.generator.generate_hillshade_plot(
            self.test_data,
            output_path
        )
        
        self.assertTrue(Path(saved_path).exists())


class TestTerrainUtils(unittest.TestCase):
    """地形工具测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.utils = TerrainUtils()
        
        # 创建测试数据
        self.test_data = np.random.rand(50, 50) * 100 + 50
    
    def test_calculate_statistics(self):
        """测试计算统计信息"""
        stats = self.utils.calculate_statistics(self.test_data)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('min_elevation', stats)
        self.assertIn('max_elevation', stats)
        self.assertIn('mean_elevation', stats)
        self.assertIn('valid_pixels', stats)
    
    def test_calculate_slope_aspect(self):
        """测试计算坡度和坡向"""
        slope, aspect = self.utils.calculate_slope_aspect(self.test_data)
        
        self.assertEqual(slope.shape, self.test_data.shape)
        self.assertEqual(aspect.shape, self.test_data.shape)
        self.assertGreaterEqual(np.nanmin(slope), 0)
        self.assertGreaterEqual(np.nanmin(aspect), 0)
        self.assertLessEqual(np.nanmax(aspect), 360)
    
    def test_create_hillshade(self):
        """测试创建山体阴影"""
        hillshade = self.utils.create_hillshade(self.test_data)
        
        self.assertEqual(hillshade.shape, self.test_data.shape)
        self.assertEqual(hillshade.dtype, np.uint8)
        self.assertGreaterEqual(np.min(hillshade), 0)
        self.assertLessEqual(np.max(hillshade), 255)
    
    def test_detect_peaks_valleys(self):
        """测试检测山峰和山谷"""
        peaks_valleys = self.utils.detect_peaks_valleys(self.test_data)
        
        self.assertIsInstance(peaks_valleys, dict)
        self.assertIn('peaks', peaks_valleys)
        self.assertIn('valleys', peaks_valleys)
    
    def test_validate_data_quality(self):
        """测试数据质量验证"""
        report = self.utils.validate_data_quality(self.test_data)
        
        self.assertIsInstance(report, dict)
        self.assertIn('quality_score', report)
        self.assertIn('quality_level', report)
        self.assertGreaterEqual(report['quality_score'], 0)
        self.assertLessEqual(report['quality_score'], 100)


def run_all_tests():
    """运行所有测试"""
    print("开始运行地形处理器测试套件...")
    print("=" * 60)
    
    # 创建测试套件
    test_classes = [
        TestTerrainProcessor,
        TestTIFReader,
        TestDataProcessor,
        TestContourGenerator,
        TestTerrainUtils
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)