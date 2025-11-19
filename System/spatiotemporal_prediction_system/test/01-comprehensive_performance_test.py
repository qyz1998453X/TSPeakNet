"""
Comprehensive Performance Testing Suite
For Computers in Industry Journal - Supplementary Material

Tests:
1. API Response Time (Single User)
2. Concurrent Performance (Multiple Users)  
3. System Resource Utilization
4. Data Processing Performance
5. System Stability Test
6. Throughput Analysis

Author: AgriGuard Team
Date: November 2024
"""

import requests
import time
import psutil
import statistics
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

class PerformanceTestSuite:
    def __init__(self, base_url="http://localhost:8003"):
        self.base_url = base_url
        self.results = {
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self._get_system_info(),
            "api_response_times": {},
            "concurrent_tests": {},
            "resource_utilization": {},
            "data_processing": {},
            "stability_test": {},
            "throughput_analysis": {}
        }
    
    def _get_system_info(self):
        """Collect system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": f"{psutil.cpu_freq().current:.0f}" if psutil.cpu_freq() else "N/A",
            "total_memory_gb": f"{psutil.virtual_memory().total / (1024**3):.2f}",
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        }
    
    def test_api_response_time(self):
        """Test 1: API Response Time (Server-side only)"""
        print("\n" + "="*70)
        print("Test 1: API Response Time Analysis")
        print("="*70)
        
        endpoints = [
            "/api/raw-data",
            "/api/weather-relationship", 
            "/api/regional-warning-data",
            "/api/model-stats?model=LSTM",
            "/api/compare-models"
        ]
        
        for endpoint in endpoints:
            print(f"\nTesting: {endpoint}")
            
            # Clear cache
            requests.post(f"{self.base_url}/api/cache/clear")
            time.sleep(0.5)
            
            # Test without cache (10 iterations)
            cold_times = []
            for i in range(10):
                response = requests.get(f"{self.base_url}{endpoint}", timeout=15)
                server_time = float(response.headers.get('X-Server-Time', '0')) * 1000000  # 转换为微秒
                cold_times.append(server_time)
                if (i + 1) % 5 == 0:
                    print(f"  Cold cache: {i+1}/10 complete")
                time.sleep(0.5)
            
            # Test with cache (50 iterations for better precision)
            warm_times = []
            for i in range(50):
                response = requests.get(f"{self.base_url}{endpoint}", timeout=15)
                server_time = float(response.headers.get('X-Server-Time', '0')) * 1000000  # 转换为微秒
                warm_times.append(server_time)
                if (i + 1) % 10 == 0:
                    print(f"  Warm cache: {i+1}/50 complete")
                time.sleep(0.1)
            
            result = {
                "endpoint": endpoint,
                "cold_cache": {
                    "mean_us": round(statistics.mean(cold_times), 2),
                    "median_us": round(statistics.median(cold_times), 2),
                    "std_dev_us": round(statistics.stdev(cold_times), 2) if len(cold_times) > 1 else 0,
                    "min_us": round(min(cold_times), 2),
                    "max_us": round(max(cold_times), 2)
                },
                "warm_cache": {
                    "mean_us": round(statistics.mean(warm_times), 2),
                    "median_us": round(statistics.median(warm_times), 2),
                    "std_dev_us": round(statistics.stdev(warm_times), 2) if len(warm_times) > 1 else 0,
                    "min_us": round(min(warm_times), 2),
                    "max_us": round(max(warm_times), 2)
                }
            }
            
            self.results["api_response_times"][endpoint] = result
            
            print(f"  Cold cache - Mean: {result['cold_cache']['mean_us']:.2f}μs, "
                  f"Median: {result['cold_cache']['median_us']:.2f}μs")
            print(f"  Warm cache - Mean: {result['warm_cache']['mean_us']:.2f}μs, "
                  f"Median: {result['warm_cache']['median_us']:.2f}μs")
    
    def test_concurrent_performance(self):
        """Test 2: Concurrent User Performance"""
        print("\n" + "="*70)
        print("Test 2: Concurrent Performance Analysis")
        print("="*70)
        
        endpoint = "/api/regional-warning-data"
        test_scenarios = [5, 10, 20, 30]
        
        for num_users in test_scenarios:
            print(f"\nTesting {num_users} concurrent users...")
            
            requests_per_user = 10
            total_requests = num_users * requests_per_user
            
            def make_request():
                try:
                    start = time.time()
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=15)
                    elapsed = time.time() - start
                    server_time = float(response.headers.get('X-Server-Time', '0'))
                    return {
                        "success": response.status_code == 200,
                        "total_time": elapsed * 1000,  # 毫秒
                        "server_time": server_time * 1000000  # 微秒
                    }
                except:
                    return {"success": False, "total_time": 0, "server_time": 0}
            
            start_time = time.time()
            results = []
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(make_request) for _ in range(total_requests)]
                for future in as_completed(futures):
                    results.append(future.result())
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            successful = [r for r in results if r["success"]]
            success_count = len(successful)
            
            if successful:
                server_times = [r["server_time"] for r in successful]
                total_times = [r["total_time"] for r in successful]
                
                test_result = {
                    "concurrent_users": num_users,
                    "total_requests": total_requests,
                    "successful_requests": success_count,
                    "failed_requests": total_requests - success_count,
                    "success_rate": round(success_count / total_requests * 100, 2),
                    "total_duration_s": round(total_duration, 3),
                    "throughput_req_per_s": round(total_requests / total_duration, 2),
                    "server_processing": {
                        "mean_us": round(statistics.mean(server_times), 2),
                        "median_us": round(statistics.median(server_times), 2),
                        "std_dev_us": round(statistics.stdev(server_times), 2) if len(server_times) > 1 else 0
                    },
                    "end_to_end": {
                        "mean_ms": round(statistics.mean(total_times), 3),
                        "median_ms": round(statistics.median(total_times), 3),
                        "max_ms": round(max(total_times), 3)
                    }
                }
                
                self.results["concurrent_tests"][f"{num_users}_users"] = test_result
                
                print(f"  Success rate: {test_result['success_rate']}%")
                print(f"  Throughput: {test_result['throughput_req_per_s']} req/s")
                print(f"  Server processing mean: {test_result['server_processing']['mean_us']:.2f}μs")
                print(f"  End-to-end mean: {test_result['end_to_end']['mean_ms']:.3f}ms")
    
    def test_resource_utilization(self):
        """Test 3: System Resource Utilization"""
        print("\n" + "="*70)
        print("Test 3: System Resource Utilization")
        print("="*70)
        
        duration = 30
        print(f"Monitoring for {duration} seconds...")
        
        cpu_samples = []
        memory_samples = []
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            
            cpu_samples.append(cpu)
            memory_samples.append(memory)
            sample_count += 1
            
            if sample_count % 10 == 0:
                print(f"  Sample {sample_count}: CPU={cpu:.1f}%, Memory={memory:.1f}%")
        
        self.results["resource_utilization"] = {
            "duration_seconds": duration,
            "sample_count": sample_count,
            "cpu_percent": {
                "mean": round(statistics.mean(cpu_samples), 2),
                "median": round(statistics.median(cpu_samples), 2),
                "min": round(min(cpu_samples), 2),
                "max": round(max(cpu_samples), 2),
                "std_dev": round(statistics.stdev(cpu_samples), 2) if len(cpu_samples) > 1 else 0
            },
            "memory_percent": {
                "mean": round(statistics.mean(memory_samples), 2),
                "median": round(statistics.median(memory_samples), 2),
                "min": round(min(memory_samples), 2),
                "max": round(max(memory_samples), 2),
                "std_dev": round(statistics.stdev(memory_samples), 2) if len(memory_samples) > 1 else 0
            }
        }
        
        print(f"\n  CPU: Mean={self.results['resource_utilization']['cpu_percent']['mean']}%, "
              f"Max={self.results['resource_utilization']['cpu_percent']['max']}%")
        print(f"  Memory: Mean={self.results['resource_utilization']['memory_percent']['mean']}%, "
              f"Max={self.results['resource_utilization']['memory_percent']['max']}%")
    
    def test_data_processing(self):
        """Test 4: Data Processing Performance"""
        print("\n" + "="*70)
        print("Test 4: Data Processing Performance")
        print("="*70)
        
        endpoints = [
            "/api/raw-data",
            "/api/compare-models",
            "/api/district-model-comparison"
        ]
        
        for endpoint in endpoints:
            print(f"\nTesting: {endpoint}")
            
            try:
                # Clear cache
                requests.post(f"{self.base_url}/api/cache/clear")
                time.sleep(0.5)
                
                response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
                
                if response.status_code == 200:
                    data_size = len(response.content)
                    server_time_us = float(response.headers.get('X-Server-Time', '0')) * 1000000  # 微秒
                    
                    result = {
                        "endpoint": endpoint,
                        "data_size_bytes": data_size,
                        "data_size_kb": round(data_size / 1024, 2),
                        "server_processing_us": round(server_time_us, 2),
                        "throughput_kb_per_s": round((data_size / 1024) / (server_time_us / 1000000), 2) if server_time_us > 0 else 0
                    }
                    
                    self.results["data_processing"][endpoint] = result
                    
                    # 根据时间大小选择合适的单位显示
                    if server_time_us >= 1000:
                        print(f"  Data size: {result['data_size_kb']} KB")
                        print(f"  Server processing: {server_time_us/1000:.2f}ms ({server_time_us:.0f}μs)")
                        print(f"  Throughput: {result['throughput_kb_per_s']} KB/s")
                    else:
                        print(f"  Data size: {result['data_size_kb']} KB")
                        print(f"  Server processing: {server_time_us:.2f}μs")
                        print(f"  Throughput: {result['throughput_kb_per_s']} KB/s")
                else:
                    print(f"  Failed with status code: {response.status_code}")
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    def test_stability(self):
        """Test 5: System Stability"""
        print("\n" + "="*70)
        print("Test 5: System Stability Test")
        print("="*70)
        
        duration_minutes = 3
        duration_seconds = duration_minutes * 60
        
        print(f"Testing for {duration_minutes} minutes...")
        
        endpoints = [
            "/api/raw-data",
            "/api/regional-warning-data",
            "/api/weather-data"
        ]
        
        start_time = time.time()
        request_count = 0
        success_count = 0
        server_times = []
        
        while time.time() - start_time < duration_seconds:
            endpoint = endpoints[request_count % len(endpoints)]
            
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    success_count += 1
                    server_time = float(response.headers.get('X-Server-Time', '0')) * 1000000  # 微秒
                    server_times.append(server_time)
                
                request_count += 1
                
                if request_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Progress: {int(elapsed)}s, Requests: {request_count}, "
                          f"Success: {success_count}/{request_count}")
                
                time.sleep(0.5)
                
            except:
                request_count += 1
        
        total_time = time.time() - start_time
        
        self.results["stability_test"] = {
            "duration_seconds": round(total_time, 2),
            "total_requests": request_count,
            "successful_requests": success_count,
            "failed_requests": request_count - success_count,
            "success_rate": round(success_count / request_count * 100, 2) if request_count > 0 else 0,
            "requests_per_minute": round(request_count / (total_time / 60), 2),
            "server_processing": {
                "mean_us": round(statistics.mean(server_times), 2) if server_times else 0,
                "median_us": round(statistics.median(server_times), 2) if server_times else 0,
                "std_dev_us": round(statistics.stdev(server_times), 2) if len(server_times) > 1 else 0
            }
        }
        
        print(f"\n  Total requests: {request_count}")
        print(f"  Success rate: {self.results['stability_test']['success_rate']}%")
        print(f"  Mean server processing: {self.results['stability_test']['server_processing']['mean_us']:.2f}μs")
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE TESTING SUITE")
        print("For Computers in Industry Journal")
        print("="*70)
        print(f"Test Date: {self.results['test_date']}")
        print(f"System: {self.results['system_info']}")
        print("="*70)
        
        # Run all tests
        self.test_api_response_time()
        self.test_concurrent_performance()
        self.test_resource_utilization()
        self.test_data_processing()
        self.test_stability()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
    
    def generate_report(self):
        """Generate JSON and Markdown reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = f"performance_test_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Markdown report
        md_file = f"performance_test_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Performance Test Report\n\n")
            f.write(f"**Date**: {self.results['test_date']}\\n")
            f.write(f"**Purpose**: Supplementary Material for Computers in Industry Journal\\n\\n")
            
            f.write("## System Configuration\n\n")
            f.write("| Parameter | Value |\n|-----------|-------|\n")
            for key, value in self.results['system_info'].items():
                f.write(f"| {key} | {value} |\n")
            f.write("\n")
            
            f.write("## Test Results Summary\n\n")
            f.write("### 1. API Response Time (Server-side)\n\n")
            f.write("| Endpoint | Cold Cache (μs) | Warm Cache (μs) | Std Dev (μs) |\n")
            f.write("|----------|-----------------|-----------------|-------------|\n")
            for endpoint, data in self.results['api_response_times'].items():
                f.write(f"| {endpoint} | {data['cold_cache']['mean_us']:.2f} | "
                       f"{data['warm_cache']['mean_us']:.2f} | "
                       f"{data['warm_cache']['std_dev_us']:.2f} |\n")
            f.write("\n")
            
            f.write("### 2. Concurrent Performance\n\n")
            f.write("| Users | Throughput (req/s) | Success Rate | Server Mean (μs) | End-to-End (ms) |\n")
            f.write("|-------|-------------------|--------------|------------------|----------------|\n")
            for key, data in self.results['concurrent_tests'].items():
                f.write(f"| {data['concurrent_users']} | {data['throughput_req_per_s']} | "
                       f"{data['success_rate']}% | "
                       f"{data['server_processing']['mean_us']:.2f} | "
                       f"{data['end_to_end']['mean_ms']:.2f} |\n")
            f.write("\n")
            
            f.write("### 3. Resource Utilization\n\n")
            res = self.results['resource_utilization']
            f.write(f"- CPU: Mean {res['cpu_percent']['mean']}%, Max {res['cpu_percent']['max']}%\\n")
            f.write(f"- Memory: Mean {res['memory_percent']['mean']}%, Max {res['memory_percent']['max']}%\\n\\n")
            
            f.write("### 4. System Stability\n\n")
            stab = self.results['stability_test']
            f.write(f"- Duration: {stab['duration_seconds']}s\\n")
            f.write(f"- Total Requests: {stab['total_requests']}\\n")
            f.write(f"- Success Rate: {stab['success_rate']}%\\n")
            f.write(f"- Server Processing Mean: {stab['server_processing']['mean_us']:.2f}μs\\n\\n")
        
        print(f"\n  Reports generated:")
        print(f"    JSON: {json_file}")
        print(f"    Markdown: {md_file}")


if __name__ == "__main__":
    # Check server connection
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        print("✓ Server connection successful\n")
    except:
        print("✗ Cannot connect to server (http://localhost:8003)")
        print("  Please start Flask server first: python app_flask.py\n")
        sys.exit(1)
    
    # Run tests
    suite = PerformanceTestSuite()
    suite.run_all_tests()
