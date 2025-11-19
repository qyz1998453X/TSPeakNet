"""
系统吞吐量压力测试
专注于测试系统最大吞吐量（请求/秒）

用于 Computers in Industry 期刊补充材料
"""

import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading

class ThroughputTester:
    def __init__(self, base_url="http://localhost:8003"):
        self.base_url = base_url
        self.results = []
        self.lock = threading.Lock()
    
    def make_request(self, endpoint):
        """执行单个请求"""
        try:
            start = time.time()
            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            elapsed = time.time() - start
            
            server_time = float(response.headers.get('X-Server-Time', '0'))
            
            return {
                'success': response.status_code == 200,
                'elapsed': elapsed,
                'server_time': server_time,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'success': False,
                'elapsed': 0,
                'server_time': 0,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def test_throughput(self, endpoint, concurrent_users, duration_seconds=30):
        """
        测试指定并发数下的吞吐量
        
        Args:
            endpoint: API端点
            concurrent_users: 并发用户数
            duration_seconds: 测试持续时间（秒）
        """
        print(f"\n{'='*70}")
        print(f"并发用户: {concurrent_users}")
        print(f"测试时长: {duration_seconds}秒")
        print(f"端点: {endpoint}")
        print(f"{'='*70}")
        
        start_time = time.time()
        request_count = 0
        success_count = 0
        results = []
        
        def worker():
            """工作线程 - 持续发送请求"""
            nonlocal request_count, success_count
            
            while time.time() - start_time < duration_seconds:
                result = self.make_request(endpoint)
                
                with self.lock:
                    request_count += 1
                    if result['success']:
                        success_count += 1
                        results.append(result)
                    
                    # 每秒报告一次进度
                    if request_count % 100 == 0:
                        elapsed = time.time() - start_time
                        current_throughput = request_count / elapsed
                        print(f"  进度: {elapsed:.1f}s | 请求: {request_count} | "
                              f"成功: {success_count} | 吞吐量: {current_throughput:.1f} req/s")
        
        # 启动并发线程
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_users)]
            
            # 等待所有线程完成
            for future in as_completed(futures):
                pass
        
        # 计算结果
        total_duration = time.time() - start_time
        throughput = request_count / total_duration
        success_rate = (success_count / request_count * 100) if request_count > 0 else 0
        
        # 计算响应时间统计
        if results:
            elapsed_times = [r['elapsed'] * 1000 for r in results]  # 转换为毫秒
            server_times = [r['server_time'] * 1000000 for r in results]  # 转换为微秒
            
            stats = {
                'concurrent_users': concurrent_users,
                'duration': round(total_duration, 2),
                'total_requests': request_count,
                'successful_requests': success_count,
                'failed_requests': request_count - success_count,
                'success_rate': round(success_rate, 2),
                'throughput': round(throughput, 2),
                'response_time_ms': {
                    'mean': round(statistics.mean(elapsed_times), 2),
                    'median': round(statistics.median(elapsed_times), 2),
                    'min': round(min(elapsed_times), 2),
                    'max': round(max(elapsed_times), 2),
                    'p95': round(sorted(elapsed_times)[int(len(elapsed_times) * 0.95)], 2),
                    'p99': round(sorted(elapsed_times)[int(len(elapsed_times) * 0.99)], 2)
                },
                'server_time_us': {
                    'mean': round(statistics.mean(server_times), 2),
                    'median': round(statistics.median(server_times), 2)
                }
            }
        else:
            stats = {
                'concurrent_users': concurrent_users,
                'duration': round(total_duration, 2),
                'total_requests': request_count,
                'successful_requests': success_count,
                'failed_requests': request_count - success_count,
                'success_rate': 0,
                'throughput': 0
            }
        
        # 打印结果
        print(f"\n{'='*70}")
        print(f"测试结果 - {concurrent_users} 并发用户")
        print(f"{'='*70}")
        print(f"总请求数: {stats['total_requests']}")
        print(f"成功请求: {stats['successful_requests']}")
        print(f"失败请求: {stats['failed_requests']}")
        print(f"成功率: {stats['success_rate']}%")
        print(f"测试时长: {stats['duration']}秒")
        print(f"\n⭐ 吞吐量: {stats['throughput']} req/s")
        
        if 'response_time_ms' in stats:
            print(f"\n响应时间 (端到端):")
            print(f"  平均值: {stats['response_time_ms']['mean']:.2f}ms")
            print(f"  中位数: {stats['response_time_ms']['median']:.2f}ms")
            print(f"  最小值: {stats['response_time_ms']['min']:.2f}ms")
            print(f"  最大值: {stats['response_time_ms']['max']:.2f}ms")
            print(f"  P95: {stats['response_time_ms']['p95']:.2f}ms")
            print(f"  P99: {stats['response_time_ms']['p99']:.2f}ms")
            
            print(f"\n服务器处理时间:")
            print(f"  平均值: {stats['server_time_us']['mean']:.2f}μs")
            print(f"  中位数: {stats['server_time_us']['median']:.2f}μs")
        
        return stats
    
    def run_full_test(self):
        """运行完整的吞吐量测试"""
        print("\n" + "="*70)
        print("系统吞吐量压力测试")
        print("For Computers in Industry Journal")
        print("="*70)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # 测试端点（使用缓存命中的端点，测试最大吞吐量）
        endpoint = "/api/regional-warning-data"
        
        # 预热 - 确保缓存已加载
        print("\n[预热阶段] 加载缓存...")
        for _ in range(5):
            requests.get(f"{self.base_url}{endpoint}")
        print("✓ 缓存预热完成\n")
        
        # 测试不同的并发级别
        test_scenarios = [
            {'users': 200, 'duration': 20},
            {'users': 500, 'duration': 20},
            {'users': 1000, 'duration': 20},
        ]
        
        all_results = []
        
        for scenario in test_scenarios:
            result = self.test_throughput(
                endpoint=endpoint,
                concurrent_users=scenario['users'],
                duration_seconds=scenario['duration']
            )
            all_results.append(result)
            time.sleep(2)  # 测试间隔
        
        # 汇总报告
        print("\n" + "="*70)
        print("吞吐量测试总结")
        print("="*70)
        print(f"\n{'并发数':<10} {'吞吐量':<15} {'成功率':<10} {'平均响应时间':<15}")
        print("-" * 70)
        
        for result in all_results:
            throughput_str = f"{result['throughput']:.1f} req/s"
            success_str = f"{result['success_rate']:.1f}%"
            
            if 'response_time_ms' in result:
                response_str = f"{result['response_time_ms']['mean']:.2f}ms"
            else:
                response_str = "N/A"
            
            print(f"{result['concurrent_users']:<10} {throughput_str:<15} {success_str:<10} {response_str:<15}")
        
        # 找出最佳吞吐量
        best = max(all_results, key=lambda x: x['throughput'])
        print(f"\n⭐ 最佳性能:")
        print(f"   并发数: {best['concurrent_users']}")
        print(f"   吞吐量: {best['throughput']:.1f} req/s")
        print(f"   成功率: {best['success_rate']:.1f}%")
        
        if 'response_time_ms' in best:
            print(f"   平均响应: {best['response_time_ms']['mean']:.2f}ms")
            print(f"   P95响应: {best['response_time_ms']['p95']:.2f}ms")
        
        print("\n" + "="*70)
        print("测试完成！")
        print("="*70)
        
        return all_results


def main():
    # 检查服务器连接
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        print("✓ 服务器连接成功\n")
    except:
        print("✗ 无法连接到服务器 (http://localhost:8003)")
        print("  请先启动Flask服务器: python app_flask.py\n")
        return
    
    # 运行测试
    tester = ThroughputTester()
    tester.run_full_test()


if __name__ == "__main__":
    main()
