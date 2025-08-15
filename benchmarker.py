#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    request_id: int
    success: bool
    latency: float  # Total time in seconds
    ttft: float     # Time to first token in seconds
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    itl: list = None  # Inter-token latencies in seconds
    error: str = None
    
    def __post_init__(self):
        if self.itl is None:
            self.itl = []
    
    @property
    def tpot(self) -> float:
        """Time per output token in seconds (averaged from inter-token latencies)"""
        if self.itl and len(self.itl) > 0:
            return sum(self.itl) / len(self.itl)  # Keep in seconds
        elif self.completion_tokens > 0 and self.success and self.ttft < self.latency:
            # Fallback calculation if no ITL data
            return (self.latency - self.ttft) / self.completion_tokens
        return 0

class DeepSeekBenchmarker:
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        
        # DeepSeek R1 specific reasoning prompts of varying complexity
        self.reasoning_prompts = {
            'simple_math': {
                'content': """<thinking>
I need to solve this step by step.
Let me break down the problem carefully.
</thinking>

Solve this problem step by step: If a train travels 120 km in 2 hours, and then travels 180 km in 3 hours, what is the average speed for the entire journey? Show your reasoning.""",
                'expected_tokens': 300
            },
            
            'logic_puzzle': {
                'content': """<thinking>
This is a logic problem that requires careful analysis of the given conditions.
I need to work through each constraint systematically.
</thinking>

Solve this logic puzzle: Three friends Alice, Bob, and Charlie have different favorite colors (red, blue, green) and different ages (25, 30, 35). Given these clues:
1. Alice is not 30 years old
2. The person who likes blue is older than Alice
3. Charlie doesn't like red
4. The 25-year-old likes green
Determine each person's age and favorite color.""",
                'expected_tokens': 500
            },
            
            'complex_reasoning': {
                'content': """<thinking>
This is a complex reasoning problem that involves multiple steps and considerations.
I need to analyze the situation from different angles and consider various factors.
Let me think through this systematically.
</thinking>

Analyze this scenario: A company has 100 employees. Due to budget constraints, they need to reduce costs by 20%. They can either:
Option A: Lay off 20 employees (each earning $50,000/year)
Option B: Reduce everyone's salary by 20%
Option C: Implement a 4-day work week (reducing salary proportionally)

Consider the pros and cons of each option from perspectives of: employee morale, productivity, company culture, legal implications, and long-term business impact. What would you recommend and why?""",
                'expected_tokens': 800
            },
            
            'technical_problem': {
                'content': """<thinking>
This requires both technical understanding and systematic problem-solving.
I need to consider the technical constraints and provide a practical solution.
</thinking>

Design a solution for this technical problem: You're building a distributed system that needs to handle 1 million concurrent users. The system has the following requirements:
- Sub-100ms response time for 95% of requests
- 99.9% uptime
- Global deployment (users from US, Europe, Asia)
- Data consistency across regions
- Cost optimization

Describe your architecture, including: database design, caching strategy, load balancing, CDN usage, monitoring, and disaster recovery. Explain your reasoning for each choice.""",
                'expected_tokens': 1000
            },
            
            'long_context': {
                'content': """<thinking>
This requires processing a longer context and maintaining coherence throughout the response.
I need to consider all the information provided and give a comprehensive analysis.
</thinking>

Given this comprehensive business case study: 
TechCorp is a mid-sized software company (500 employees) that has been successful in the traditional desktop software market for 15 years. However, they're facing declining revenue as customers shift to cloud-based solutions. Current situation:

Financial: $50M annual revenue (down 15% from last year), $35M operating costs, $5M in cash reserves
Products: 3 main desktop applications, established customer base of 10,000 businesses, strong brand recognition in niche market
Competition: New SaaS startups offering similar functionality at lower prices with better user experience
Technology: Legacy codebase, skilled but aging development team, limited cloud experience
Market: Cloud adoption accelerating, customer demands for mobile access increasing, subscription model becoming standard

The board has approved a 3-year transformation budget of $20M. Develop a comprehensive digital transformation strategy that addresses: product migration to cloud, organizational changes, financial projections, risk mitigation, timeline, and success metrics. Consider multiple scenarios and provide detailed reasoning for your recommendations.""",
                'expected_tokens': 1500
            }
        }
    
    async def check_server_health(self) -> bool:
        """Check if the server is responding"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{self.base_url}/v1/models', 
                                     timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        logger.info("✓ Server is healthy and responding")
                        return True
                    else:
                        logger.error(f"Server health check failed: HTTP {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    async def send_request(self, session: aiohttp.ClientSession, request_id: int, 
                          prompt_content: str, max_tokens: int, temperature: float = 0.7) -> TestResult:
        """Send a single request and measure performance using vLLM's streaming approach"""
        
        payload = {
            "model": "/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746",
            "messages": [{"role": "user", "content": prompt_content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            }
        }
        
        start_time = time.perf_counter()
        ttft = 0.0
        inter_token_latencies = []
        most_recent_timestamp = start_time
        generated_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        try:
            async with session.post(self.endpoint, json=payload) as response:
                
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        
                        chunk_bytes = chunk_bytes.decode("utf-8")
                        # Skip SSE comments (pings) that start with colon
                        if chunk_bytes.startswith(":"):
                            continue
                        
                        chunk = chunk_bytes.removeprefix("data: ")
                        if chunk == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(chunk)
                            timestamp = time.perf_counter()
                            
                            # Process choices (content tokens)
                            if choices := data.get("choices"):
                                if choices and len(choices) > 0:
                                    choice = choices[0]
                                    delta = choice.get("delta", {})
                                    content = delta.get("content")
                                    
                                    if content is not None:  # Content can be empty string
                                        # Record TTFT on first content token
                                        if ttft == 0.0:
                                            ttft = timestamp - start_time
                                            logger.debug(f"Request {request_id}: TTFT = {ttft:.3f}s")
                                        else:
                                            # Record inter-token latency
                                            itl = timestamp - most_recent_timestamp
                                            inter_token_latencies.append(itl)
                                        
                                        generated_text += content
                                        most_recent_timestamp = timestamp
                            
                            # Extract usage info (usually in final chunk)
                            elif usage := data.get("usage"):
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                                total_tokens = usage.get("total_tokens", total_tokens)
                                logger.debug(f"Request {request_id}: Got usage - completion_tokens={completion_tokens}")
                        
                        except json.JSONDecodeError as e:
                            logger.debug(f"Request {request_id}: Failed to parse chunk: {e}")
                            continue
                    
                    end_time = time.perf_counter()
                    total_latency = end_time - start_time
                    
                    # Validate measurements
                    if ttft == 0.0:
                        logger.warning(f"Request {request_id}: No TTFT measured (no content received)")
                        ttft = total_latency  # Fallback
                    
                    if completion_tokens == 0 and generated_text:
                        # Estimate if server didn't provide usage
                        completion_tokens = max(1, len(generated_text) // 4)
                        logger.warning(f"Request {request_id}: Estimated {completion_tokens} tokens from {len(generated_text)} chars")
                    
                    logger.debug(f"Request {request_id}: Total latency={total_latency:.3f}s, TTFT={ttft:.3f}s, Tokens={completion_tokens}, ITL_count={len(inter_token_latencies)}")
                    
                    return TestResult(
                        request_id=request_id,
                        success=True,
                        latency=total_latency,
                        ttft=ttft,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        itl=inter_token_latencies
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        request_id=request_id,
                        success=False,
                        latency=0,
                        ttft=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        itl=[],
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    

        except Exception as e:
            return TestResult(
                request_id=request_id,
                success=False,
                latency=0,
                ttft=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                itl=[],
                error=str(e)
            )
    
    async def run_test_scenario(self, scenario_name: str, total_requests: int, 
                               max_concurrent: int) -> List[TestResult]:
        """Run a specific test scenario"""
        
        logger.info(f"Starting scenario: {scenario_name}")
        logger.info(f"Total requests: {total_requests}, Max concurrent: {max_concurrent}")
        logger.info(f"Sending all {total_requests} requests as fast as possible...")
        
        async def send_immediate_request(request_id: int):
            # Choose a random prompt type
            prompt_type = random.choice(list(self.reasoning_prompts.keys()))
            prompt_data = self.reasoning_prompts[prompt_type]
            
            return await self.send_request(
                session, request_id, 
                prompt_data['content'], 
                prompt_data['expected_tokens']
            )
        
        # Execute all requests immediately without throttling
        # Create connector with unlimited connections for stress testing
        connector = aiohttp.TCPConnector(
            limit=0,  # No total connection limit
            limit_per_host=0,  # No per-host connection limit
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        # Measure actual test duration
        test_start_time = time.perf_counter()
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                send_immediate_request(i) 
                for i in range(total_requests)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        test_end_time = time.perf_counter()
        test_duration = test_end_time - test_start_time
        
        # Filter out exceptions and convert to TestResult objects
        test_results = []
        for result in results:
            if isinstance(result, TestResult):
                test_results.append(result)
            else:
                logger.error(f"Exception in request: {result}")
        
        logger.info(f"Completed {len(test_results)} requests in {test_duration:.1f} seconds")
        
        # Add test duration to results for accurate throughput calculation
        for result in test_results:
            result.test_duration = test_duration
        
        return test_results
    
    def analyze_results(self, results: List[TestResult], scenario_name: str) -> Dict:
        """Analyze test results and compute metrics"""
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results:
            logger.error("No successful requests to analyze!")
            return {}
        
        # Calculate metrics
        latencies = [r.latency for r in successful_results]  # Keep in seconds
        ttfts = [r.ttft for r in successful_results]  # Keep in seconds
        tpots = [r.tpot for r in successful_results if r.completion_tokens > 0]
        
        total_tokens = sum(r.completion_tokens for r in successful_results)
        # Use actual test duration instead of max latency
        if successful_results and hasattr(successful_results[0], 'test_duration'):
            total_time = successful_results[0].test_duration
        else:
            # Fallback: use max latency (less accurate)
            total_time = max(r.latency for r in successful_results) if successful_results else 0
        
        analysis = {
            'scenario': scenario_name,
            'timestamp': datetime.now().isoformat(),
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100 if results else 0,
            
            # Latency metrics (seconds)
            'avg_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'max_latency': np.max(latencies),
            'min_latency': np.min(latencies),  # Shows fastest possible request
            
            # TTFT metrics (seconds)
            'avg_ttft': np.mean(ttfts),
            'median_ttft': np.median(ttfts),
            'p95_ttft': np.percentile(ttfts, 95),
            
            # TPOT metrics (seconds)
            'avg_tpot': np.mean(tpots) if tpots else 0,
            'median_tpot': np.median(tpots) if tpots else 0,
            
            # Throughput metrics
            'requests_per_second': len(successful_results) / total_time if total_time > 0 else 0,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            
            # Token metrics
            'avg_prompt_tokens': np.mean([r.prompt_tokens for r in successful_results]),
            'avg_completion_tokens': np.mean([r.completion_tokens for r in successful_results]),
            'total_tokens_generated': total_tokens,
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print formatted analysis results"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {analysis['scenario']}")
        print(f"{'='*60}")
        print(f"Timestamp: {analysis['timestamp']}")
        print(f"Total Requests: {analysis['total_requests']}")
        print(f"Successful: {analysis['successful_requests']} ({analysis['success_rate']:.1f}%)")
        print(f"Failed: {analysis['failed_requests']}")
        print()
        
        print("LATENCY METRICS (seconds):")
        print(f"  Min:     {analysis['min_latency']:.3f}  ← Fastest request (no queue)")
        print(f"  Average: {analysis['avg_latency']:.3f}")
        print(f"  Median:  {analysis['median_latency']:.3f}")
        print(f"  P95:     {analysis['p95_latency']:.3f}")
        print(f"  P99:     {analysis['p99_latency']:.3f}")
        print(f"  Max:     {analysis['max_latency']:.3f}  ← Slowest request (max queue)")
        print()
        
        print("TIME TO FIRST TOKEN (seconds):")
        print(f"  Average: {analysis['avg_ttft']:.3f}")
        print(f"  Median:  {analysis['median_ttft']:.3f}")
        print(f"  P95:     {analysis['p95_ttft']:.3f}")
        print()
        
        print("TIME PER OUTPUT TOKEN (seconds):")
        print(f"  Average: {analysis['avg_tpot']:.4f}")
        print(f"  Median:  {analysis['median_tpot']:.4f}")
        print()
        
        print("THROUGHPUT METRICS:")
        print(f"  Requests/sec: {analysis['requests_per_second']:.2f}")
        print(f"  Tokens/sec:   {analysis['tokens_per_second']:.1f}")
        print()
        
        print("TOKEN STATISTICS:")
        print(f"  Avg Prompt Tokens:     {analysis['avg_prompt_tokens']:.0f}")
        print(f"  Avg Completion Tokens: {analysis['avg_completion_tokens']:.0f}")
        print(f"  Total Tokens Generated: {analysis['total_tokens_generated']}")
        print()
    
    def save_results(self, all_analyses: List[Dict], filename: str = None):
        """Save results to JSON and CSV files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deepseek_benchmark_{timestamp}"
        
        # Save as JSON
        with open(f"{filename}.json", "w") as f:
            json.dump(all_analyses, f, indent=2)
        
        # Save as CSV for analysis
        df = pd.DataFrame(all_analyses)
        df.to_csv(f"{filename}.csv", index=False)
        
        logger.info(f"Results saved to {filename}.json and {filename}.csv")
        return df
    
    async def run_comprehensive_benchmark(self):
        """Run a comprehensive benchmark with multiple scenarios"""
        
        # Check server health first
        if not await self.check_server_health():
            logger.error("Server is not responding. Please ensure your model is running on port 8003")
            return
        
        # Define test scenarios - (name, total_requests, max_concurrent)
        scenarios = [
            ('light_load', 50, 5),       # 50 requests, max 5 concurrent
            ('medium_load', 100, 10),    # 100 requests, max 10 concurrent  
            ('heavy_load', 200, 20),     # 200 requests, max 20 concurrent
            ('burst_load', 300, 50),     # 300 requests, max 50 concurrent
            ('stress_test', 500, 0),     # 500 requests, NO concurrency limit (unlimited)
        ]
        
        all_analyses = []
        
        for scenario_name, total_requests, max_concurrent in scenarios:
            print(f"\n🚀 Starting {scenario_name}...")
            
            # Run the test
            results = await self.run_test_scenario(scenario_name, total_requests, max_concurrent)
            
            # Analyze results
            analysis = self.analyze_results(results, scenario_name)
            if analysis:
                self.print_analysis(analysis)
                all_analyses.append(analysis)
            
            # No wait between scenarios - run continuously for faster benchmarking
        
        # Save all results
        if all_analyses:
            df = self.save_results(all_analyses)
            self.create_comparison_chart(df)
        
        return all_analyses
    
    def create_comparison_chart(self, df: pd.DataFrame):
        """Create comparison charts from the results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DeepSeek R1 Performance Analysis', fontsize=16)
        
        scenarios = df['scenario'].tolist()
        
        # Throughput
        axes[0,0].bar(scenarios, df['requests_per_second'])
        axes[0,0].set_title('Throughput (req/s)')
        axes[0,0].set_ylabel('Requests/second')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Token throughput
        axes[0,1].bar(scenarios, df['tokens_per_second'])
        axes[0,1].set_title('Token Throughput')
        axes[0,1].set_ylabel('Tokens/second')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Average latency
        axes[0,2].bar(scenarios, df['avg_latency'])
        axes[0,2].set_title('Average Latency (seconds)')
        axes[0,2].set_ylabel('Seconds')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # P95 latency
        axes[1,0].bar(scenarios, df['p95_latency'])
        axes[1,0].set_title('P95 Latency (seconds)')
        axes[1,0].set_ylabel('Seconds')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # TTFT
        axes[1,1].bar(scenarios, df['avg_ttft'])
        axes[1,1].set_title('Time to First Token (seconds)')
        axes[1,1].set_ylabel('Seconds')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Success rate
        axes[1,2].bar(scenarios, df['success_rate'])
        axes[1,2].set_title('Success Rate (%)')
        axes[1,2].set_ylabel('Percentage')
        axes[1,2].set_ylim(0, 105)
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'deepseek_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Performance chart saved as deepseek_performance_{timestamp}.png")


async def main():
    parser = argparse.ArgumentParser(description="DeepSeek R1 Performance Benchmarker")
    parser.add_argument("--url", default="http://localhost:8003", 
                       help="Base URL of the vLLM server (default: http://localhost:8003)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test instead of comprehensive benchmark")
    
    args = parser.parse_args()
    
    benchmarker = DeepSeekBenchmarker(args.url)
    
    if args.quick:
        # Quick test - single scenario
        print("🏃 Running quick test...")
        results = await benchmarker.run_test_scenario("quick_test", 20, 5)  # 20 requests, max 5 concurrent
        analysis = benchmarker.analyze_results(results, "quick_test")
        if analysis:
            benchmarker.print_analysis(analysis)
    else:
        # Full comprehensive benchmark
        print("🔬 Running comprehensive benchmark...")
        await benchmarker.run_comprehensive_benchmark()

if __name__ == "__main__":
    asyncio.run(main())