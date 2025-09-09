#!/usr/bin/env python3
"""
Production Testing Suite for MMM Application
Comprehensive end-to-end testing in production environment
"""
import asyncio
import csv
import io
import json
import time
import httpx
import websockets
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class ProductionTestSuite:
    """Comprehensive production testing suite"""
    
    def __init__(self, base_url: str, websocket_url: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.websocket_url = websocket_url or base_url.replace('http', 'ws')
        self.client = httpx.AsyncClient(timeout=300)
        self.results = []
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_result(self, test_name: str, passed: bool, duration: float, details: str = ""):
        """Log test result"""
        result = {
            'test_name': test_name,
            'passed': passed,
            'duration_seconds': round(duration, 2),
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.2f}s) {details}")
    
    async def test_health_endpoints(self):
        """Test basic health and info endpoints"""
        start_time = time.time()
        
        try:
            # Test root endpoint
            response = await self.client.get(f"{self.base_url}/")
            assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
            
            root_data = response.json()
            assert "message" in root_data, "Root response missing message"
            assert "version" in root_data, "Root response missing version"
            
            self.log_result("health_endpoints", True, time.time() - start_time, 
                          f"Version: {root_data.get('version', 'unknown')}")
        except Exception as e:
            self.log_result("health_endpoints", False, time.time() - start_time, str(e))
    
    async def test_api_documentation(self):
        """Test API documentation endpoints"""
        start_time = time.time()
        
        try:
            # Test OpenAPI docs
            response = await self.client.get(f"{self.base_url}/docs")
            assert response.status_code == 200, f"Docs endpoint failed: {response.status_code}"
            assert "swagger" in response.text.lower(), "Swagger UI not loaded"
            
            # Test OpenAPI spec
            response = await self.client.get(f"{self.base_url}/openapi.json")
            assert response.status_code == 200, f"OpenAPI spec failed: {response.status_code}"
            
            spec = response.json()
            assert "openapi" in spec, "Invalid OpenAPI spec"
            
            self.log_result("api_documentation", True, time.time() - start_time,
                          f"OpenAPI version: {spec.get('openapi', 'unknown')}")
        except Exception as e:
            self.log_result("api_documentation", False, time.time() - start_time, str(e))
    
    def create_test_dataset(self, days: int = 365, channels: List[str] = None) -> pd.DataFrame:
        """Create realistic test dataset for MMM"""
        if channels is None:
            channels = ['search_brand', 'search_nonbrand', 'social', 'display', 'video']
        
        # Generate date range
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        
        # Create realistic marketing data with trends and seasonality
        np.random.seed(42)  # For reproducible results
        
        data = {'date': dates.strftime('%Y-%m-%d')}
        
        # Base profit trend
        base_profit = 50000 + np.cumsum(np.random.normal(200, 500, days))
        
        # Add seasonality (higher in Q4)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * dates.dayofyear / 365 + np.pi/2)
        base_profit *= seasonal_factor
        
        # Generate channel spend data
        total_spend = []
        for i, channel in enumerate(channels):
            # Different spend patterns per channel
            if 'search_brand' in channel:
                spend = np.random.gamma(2, 5000) * np.ones(days) * (1 + 0.1 * np.sin(2 * np.pi * dates.dayofyear / 365))
            elif 'search_nonbrand' in channel:
                spend = np.random.gamma(3, 8000) * np.ones(days) * (1 + 0.2 * np.sin(2 * np.pi * dates.dayofyear / 365))
            elif 'social' in channel:
                spend = np.random.gamma(2, 6000) * np.ones(days) * (1 + 0.15 * np.cos(2 * np.pi * dates.dayofyear / 365))
            else:
                spend = np.random.gamma(2.5, 7000) * np.ones(days) * (1 + 0.1 * np.random.randn(days))
            
            # Add some noise and ensure non-negative
            spend = np.maximum(0, spend + np.random.normal(0, spend.std() * 0.1, days))
            data[channel] = spend.round(2)
            total_spend.append(spend)
        
        # Generate profit based on spend with diminishing returns
        total_spend_array = np.array(total_spend).T
        contribution = np.sum(total_spend_array ** 0.7 * np.array([0.3, 0.25, 0.2, 0.15, 0.1]), axis=1)
        
        data['profit'] = (base_profit + contribution * 0.5 + np.random.normal(0, 2000, days)).round(2)
        data['profit'] = np.maximum(10000, data['profit'])  # Ensure minimum profit
        
        # Add optional columns
        data['is_holiday'] = np.random.choice([0, 1], days, p=[0.95, 0.05])
        
        return pd.DataFrame(data)
    
    async def test_data_upload(self):
        """Test data upload functionality with realistic data"""
        start_time = time.time()
        
        try:
            # Create test dataset
            test_data = self.create_test_dataset(days=400)  # ~13 months of data
            
            # Convert to CSV string
            csv_content = test_data.to_csv(index=False)
            csv_file = io.BytesIO(csv_content.encode())
            
            # Upload data
            files = {"file": ("test_data.csv", csv_file, "text/csv")}
            response = await self.client.post(f"{self.base_url}/api/data/upload", files=files)
            
            assert response.status_code == 200, f"Upload failed: {response.status_code} - {response.text}"
            
            upload_result = response.json()
            assert "upload_id" in upload_result, "Upload ID not returned"
            assert upload_result.get("status") == "completed", f"Upload status: {upload_result.get('status')}"
            
            # Store upload_id for subsequent tests
            self.upload_id = upload_result["upload_id"]
            
            self.log_result("data_upload", True, time.time() - start_time,
                          f"Upload ID: {self.upload_id}, Rows: {len(test_data)}")
        except Exception as e:
            self.log_result("data_upload", False, time.time() - start_time, str(e))
    
    async def test_model_training_with_websocket(self):
        """Test model training with real-time WebSocket updates"""
        start_time = time.time()
        
        if not hasattr(self, 'upload_id'):
            self.log_result("model_training_websocket", False, time.time() - start_time,
                          "Skipped: No upload_id from previous test")
            return
        
        try:
            # Start training via REST API
            training_request = {"upload_id": self.upload_id}
            response = await self.client.post(f"{self.base_url}/api/model/train", json=training_request)
            
            assert response.status_code == 200, f"Training start failed: {response.status_code}"
            
            training_result = response.json()
            run_id = training_result.get("run_id")
            assert run_id, "Run ID not returned"
            
            # Connect to WebSocket for real-time updates
            websocket_uri = f"{self.websocket_url}/ws/training/{run_id}"
            
            updates_received = 0
            final_status = None
            
            async with websockets.connect(websocket_uri, timeout=180) as websocket:
                async for message in websocket:
                    try:
                        update = json.loads(message)
                        updates_received += 1
                        
                        if update.get("status") in ["completed", "failed"]:
                            final_status = update.get("status")
                            break
                            
                        # Log progress updates
                        if "progress" in update:
                            print(f"   Training progress: {update['progress']}")
                            
                    except json.JSONDecodeError:
                        continue
            
            assert final_status == "completed", f"Training failed with status: {final_status}"
            assert updates_received > 0, "No WebSocket updates received"
            
            # Store run_id for subsequent tests
            self.run_id = run_id
            
            self.log_result("model_training_websocket", True, time.time() - start_time,
                          f"Run ID: {run_id}, Updates: {updates_received}")
        except Exception as e:
            self.log_result("model_training_websocket", False, time.time() - start_time, str(e))
    
    async def test_model_results(self):
        """Test model results retrieval"""
        start_time = time.time()
        
        if not hasattr(self, 'run_id'):
            self.log_result("model_results", False, time.time() - start_time,
                          "Skipped: No run_id from previous test")
            return
        
        try:
            # Get model results
            response = await self.client.get(f"{self.base_url}/api/model/{self.run_id}/results")
            assert response.status_code == 200, f"Results failed: {response.status_code}"
            
            results = response.json()
            assert "model_fit" in results, "Model fit results missing"
            assert "channel_performance" in results, "Channel performance missing"
            
            # Validate model fit metrics
            model_fit = results["model_fit"]
            assert "mape" in model_fit, "MAPE missing from model fit"
            assert "r_squared" in model_fit, "R-squared missing from model fit"
            
            mape = model_fit["mape"]
            r_squared = model_fit["r_squared"]
            
            self.log_result("model_results", True, time.time() - start_time,
                          f"MAPE: {mape:.2f}%, R²: {r_squared:.3f}")
        except Exception as e:
            self.log_result("model_results", False, time.time() - start_time, str(e))
    
    async def test_response_curves(self):
        """Test response curve generation"""
        start_time = time.time()
        
        if not hasattr(self, 'run_id'):
            self.log_result("response_curves", False, time.time() - start_time,
                          "Skipped: No run_id from previous test")
            return
        
        try:
            response = await self.client.get(f"{self.base_url}/api/model/{self.run_id}/response-curves")
            assert response.status_code == 200, f"Response curves failed: {response.status_code}"
            
            curves = response.json()
            assert isinstance(curves, dict), "Response curves should be a dictionary"
            assert len(curves) > 0, "No response curves returned"
            
            # Validate curve structure
            first_channel = next(iter(curves.values()))
            assert "spend_points" in first_channel, "Spend points missing"
            assert "response_points" in first_channel, "Response points missing"
            
            self.log_result("response_curves", True, time.time() - start_time,
                          f"Channels: {len(curves)}")
        except Exception as e:
            self.log_result("response_curves", False, time.time() - start_time, str(e))
    
    async def test_budget_optimization(self):
        """Test budget optimization functionality"""
        start_time = time.time()
        
        if not hasattr(self, 'run_id'):
            self.log_result("budget_optimization", False, time.time() - start_time,
                          "Skipped: No run_id from previous test")
            return
        
        try:
            # Test optimization with realistic constraints
            optimization_request = {
                "run_id": self.run_id,
                "total_budget": 100000,  # $100k monthly budget
                "current_spend": {
                    "search_brand": 20000,
                    "search_nonbrand": 30000,
                    "social": 25000,
                    "display": 15000,
                    "video": 10000
                },
                "constraints": {
                    "search_brand": {"min": 15000, "max": 40000},
                    "social": {"min": 10000, "max": 50000}
                }
            }
            
            response = await self.client.post(f"{self.base_url}/api/optimization/run", 
                                            json=optimization_request)
            assert response.status_code == 200, f"Optimization failed: {response.status_code}"
            
            optimization = response.json()
            assert "optimized_spend" in optimization, "Optimized spend missing"
            assert "expected_profit_lift" in optimization, "Profit lift missing"
            
            # Validate optimization results
            optimized_spend = optimization["optimized_spend"]
            total_optimized = sum(optimized_spend.values())
            assert abs(total_optimized - 100000) < 100, f"Budget constraint violated: {total_optimized}"
            
            profit_lift = optimization["expected_profit_lift"]
            
            self.log_result("budget_optimization", True, time.time() - start_time,
                          f"Profit lift: {profit_lift:.1f}%")
        except Exception as e:
            self.log_result("budget_optimization", False, time.time() - start_time, str(e))
    
    async def test_performance_under_load(self):
        """Test performance with concurrent requests"""
        start_time = time.time()
        
        try:
            # Create multiple concurrent requests to test performance
            tasks = []
            for _ in range(10):  # 10 concurrent requests
                task = self.client.get(f"{self.base_url}/")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            for response in responses:
                if isinstance(response, httpx.Response) and response.status_code == 200:
                    success_count += 1
            
            success_rate = success_count / len(tasks)
            assert success_rate >= 0.8, f"Success rate too low: {success_rate}"
            
            self.log_result("performance_load", True, time.time() - start_time,
                          f"Success rate: {success_rate:.1%} ({success_count}/{len(tasks)})")
        except Exception as e:
            self.log_result("performance_load", False, time.time() - start_time, str(e))
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Starting Production Test Suite for MMM Application")
        print(f"   Target URL: {self.base_url}")
        print(f"   WebSocket URL: {self.websocket_url}")
        print()
        
        # Run tests in order
        await self.test_health_endpoints()
        await self.test_api_documentation()
        await self.test_data_upload()
        await self.test_model_training_with_websocket()
        await self.test_model_results()
        await self.test_response_curves()
        await self.test_budget_optimization()
        await self.test_performance_under_load()
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        total_time = sum(r['duration_seconds'] for r in self.results)
        
        print()
        print("=" * 60)
        print("🏁 PRODUCTION TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏱️ Total Time: {total_time:.1f} seconds")
        print(f"🎯 Success Rate: {passed_tests/total_tests:.1%}")
        print()
        
        if failed_tests > 0:
            print("❌ Failed Tests:")
            for result in self.results:
                if not result['passed']:
                    print(f"   • {result['test_name']}: {result['details']}")
            print()
        
        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"production_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Your MMM application is production-ready!")
        else:
            print("⚠️ Some tests failed. Please review and fix issues before proceeding.")

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MMM Application Production Test Suite")
    parser.add_argument("--url", required=True, help="Base URL of the MMM application")
    parser.add_argument("--websocket-url", help="WebSocket URL (defaults to derived from base URL)")
    
    args = parser.parse_args()
    
    async with ProductionTestSuite(args.url, args.websocket_url) as test_suite:
        await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())